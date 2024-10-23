import base64
from typing import Optional
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse, JSONResponse
import os
from dotenv import load_dotenv
import copy

import json
from fastapi.responses import StreamingResponse
from PIL import Image
import io
from pydantic import BaseModel
from vmbpy import *
import time
import uvicorn
import threading
import numpy as np
import cv2
import app_settings
from models.qr_detection import *
import random
import pandas as pd
#add cors to fastapi
from fastapi.middleware.cors import CORSMiddleware
from utils.calib_camera import CalibrationTool,ColorCorrectionTool
from utils.process_qrcode import qr_decoder_fn
import requests
global image_captured_data
global current_image
global connected_camera

image_captured_data = None
current_image = None
connected_camera = None
capture_event =threading.Event()
capture_event.clear()
image_captured_event = threading.Event()
image_captured_event.clear()
origins = [
'*'
] 
load_dotenv()
analyze_url= os.getenv('AI_API_URL')   
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
calibration_tool = CalibrationTool()
color_correction_tool = ColorCorrectionTool()
df = pd.read_csv('data.csv')
class CalibrationCorner(BaseModel):
    data : str

def get_camera(camera_id: Optional[str]) -> Camera:
    with VmbSystem.get_instance() as vmb:
        if camera_id:
            try:
                return vmb.get_camera_by_id(camera_id)

            except VmbCameraError:
                print('Failed to access Camera \'{}\'. Abort.'.format(camera_id))
                return None

        else:
            cams = vmb.get_all_cameras()
            if not cams:
                print('No Cameras accessible. Abort.')
                return None

            return cams[0]

def find_first_match(lst, condition):
    for item in lst:
        if condition(item):
            return item
    return None  # Return None if no match is found
def get_qrcode_data(img):
    # detect qrcode
    qrcode_detection_results, pred_qrcode_scores, _ = predict(
        qrcode_detector, img, resize=(640, 640), postprocess=pp
    )
    print(qrcode_detection_results)
    print(pred_qrcode_scores)
    qrcode_detection_results_filterd =[]
    pred_qrcode_scores_filterd = []
    image_size= img.shape[:2]
    max_w = image_size[1]//9
    max_h = image_size[0]//9
    for i in range(len(pred_qrcode_scores)):
        bbox = qrcode_detection_results[i]
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        if w < max_w and h < max_h:
            qrcode_detection_results_filterd.append(bbox)
            pred_qrcode_scores_filterd.append(pred_qrcode_scores[i])
    try:
        pred_qrcode_scores_idx = np.argmax(np.asarray(pred_qrcode_scores_filterd), axis=0)
        print(pred_qrcode_scores_idx)
        qrcode_detection_results = qrcode_detection_results_filterd[pred_qrcode_scores_idx]
    except Exception as e:
        print(e)
        return None
    x1, y1, x2, y2 = qrcode_detection_results
    offset = 0
    print(x1, y1, x2, y2)
    cropped_qrcode_img = img[
        int(y1 - offset) : int(y2 + offset), int(x1 - offset) : int(x2 + offset)
    ]
    # decode qrcode
    qrcode_data = qr_decoder_fn(img=cropped_qrcode_img, qr_reader=qrcode_reader)
    if qrcode_data is not None:       
        return {
            "qrcode_data": qrcode_data,
            "qr_image": cropped_qrcode_img,
        }
    return {
            "qrcode_data": 'qrcode_data',
            "qr_image": cropped_qrcode_img,
        }

@app.get("/connected_camera_info")
async def get_connected_camera_info():
    global connected_camera
    if connected_camera is not None:
        return {"camera_id": connected_camera.get_id(), "camera_name": connected_camera.get_name()}
    else:
        return {"status": "disconnected"}
def divide_image(image, num_divide):
    height, width = image.shape[0:2]
    h = height // num_divide
    w = width // num_divide
    images = []
    for i in range(num_divide):
        for j in range(num_divide):
            images.append(image[i*h:(i+1)*h, j*w:(j+1)*w])
    return images
@app.get("/read_qrcode")
async def read_qrcode():
    global current_image
    if current_image is None:
        return Response(status_code=404, content="Camera not found")
    rgb_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB)
    qrcode_data = get_qrcode_data(rgb_image)
    if qrcode_data is not None:
        qrcode_data['qr_image'] = cv2.cvtColor(qrcode_data['qr_image'], cv2.COLOR_BGR2RGB)
        _, qr_image_encoded = cv2.imencode('.jpg', qrcode_data['qr_image'])
        qrcode_data['qr_image'] = base64.b64encode(qr_image_encoded.tobytes()).decode('utf-8')
        return qrcode_data
    else:
        raise Response(status_code=404, content="QRCode not found")
    #raise HTTPException(status_code=404, detail="Camera not found")
@app.get("/capture")
async def capture_image(test_mode: Optional[bool] = False):
    global current_image
    global image_captured_data
    print(f"test_mode {test_mode}")
    if test_mode:
        test_image_url = df.iloc[random.randint(0, len(df) - 1), 0]+"?raw=true"
        print(test_image_url)
        response = requests.get(test_image_url)
        large_image = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
        #large_image = cv2.imread('test_img.png')
        global current_image
        current_image = large_image
        display_image = cv2.resize(current_image, (current_image.shape[1]//4, current_image.shape[0]//4))
        _, img_encoded = cv2.imencode('.jpg', display_image)
        return Response(content=img_encoded.tobytes(), media_type="image/jpeg")
    if not SoftwareTrigger():
        return Response(status_code=404, content="Cannot trigger camera")
        return {"status": False, "message": "Cannot trigger camera"}
    if image_captured_event.wait(10):
        current_image = image_captured_data
    else:
        test_image_url = df.iloc[random.randint(0, len(df) - 1), 0]+"?raw=true"
        print(test_image_url)
        response = requests.get(test_image_url)
        large_image = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
        #large_image = cv2.imread('test_img.png')
        
        current_image = large_image
    image_captured_event.clear()
    #resize image for faster processing
    display_image = cv2.resize(current_image, (current_image.shape[1]//4, current_image.shape[0]//4))
    _, img_encoded = cv2.imencode('.jpg', display_image)
    #large_image = cv2.resize(image_captured_data, (1000, int(image_captured_data.shape[0] * 1000 / image_captured_data.shape[1])))
    return Response(content=img_encoded.tobytes(), media_type="image/jpeg")

@app.get("/analyze_image")
async def analyze_image(qr_code: str):
    if qr_code == 'undefined':
        return {"status": False, "message": "QRCode not found"}
    global current_image
    # URL of the FastAPI server
    url = f"{analyze_url}/api/v2/inspection_t_shirt/"
    resize_img = cv2.resize(current_image, (current_image.shape[1]//4, current_image.shape[0]//4))
    resize_img = cv2.cvtColor(resize_img, cv2.COLOR_BGR2RGB)
    _, encoded_image = cv2.imencode('.jpg', resize_img)
    image_bytes = encoded_image.tobytes()
    
    data = {
        'camera_matrix': calibration_tool.camera_matrix.tolist(),
        'dist_coeffs': calibration_tool.distortion_coeffs.tolist(),
        'tvecs': calibration_tool.tvecs.tolist(),
        'rvecs': calibration_tool.rvecs.tolist(),
        'M_R': color_correction_tool.M_R.tolist(),
        'M_T': color_correction_tool.M_T.tolist(),
        'qr_code': qr_code
    }
    # Prepare the files for multipart/form-data request
    files={'file': image_bytes}
    # files = {
    #     "image": ("image.jpg", image_bytes, "image/jpeg")
    # }
    # Send the request to FastAPI
    response = requests.post(url, data= {"data":json.dumps(data)}, files=files)
    try:
        content = response.json()
        return JSONResponse(content=content, status_code=response.status_code)
    except Exception as e:
        return JSONResponse(content={"status": False, "message": "Error"}, status_code=500)
@app.post("/add_calibration_points")
async def add_calibration_points(item: CalibrationCorner):
    corners = json.loads(item.data)
    calibration_tool.add_calibrated_points(corners)
    return {"status": True}
@app.post("/calibrate_camera")
async def calibrate_camera():
    camera_matrix, distortion_coeffs, rvecs, tvecs = calibration_tool.calibrate_camera()
    return {"status": True}
@app.get("/add_calibration_image")
async def add_calibration_image():
    global image_captured_data
    global current_image
    global image_captured_event
    if not SoftwareTrigger():
        return {"status": False, "message": "Cannot trigger camera"}
    if image_captured_event.wait(10):
        current_image = image_captured_data
    else:
        # random select image from calibration_images folder
        list_of_files = os.listdir('calibration_images')
        if not os.path.exists('calibration_images'):
            os.mkdir('calibration_images')
        selected_image = random.choice(list_of_files)
        large_image = cv2.imread(os.path.join('calibration_images', selected_image))
        current_image = large_image
    image_captured_event.clear()
    #capture_event.clear()
    
    #large_image = cv2.imread('test_img.png')
    image_calib,conners = calibration_tool.add_calibrate_image(current_image)
    if image_calib is not None:
        _,img_encoded  = cv2.imencode('.jpg', image_calib)
        b64Image = base64.b64encode(img_encoded).decode('utf-8')
        return {"image": b64Image, "conners": conners,'status': True}
    else:
        _,img_encoded  = cv2.imencode('.jpg', current_image)
        b64Image = base64.b64encode(img_encoded).decode('utf-8')
        return {"image": b64Image, "conners": conners,'status': False}
        raise HTTPException(status_code=404, detail="Image not not good for calibration")
@app.get("/add_color_corection_image")
async def add_color_corection_image():
    global image_captured_data
    global current_image
    global image_captured_event
    if not SoftwareTrigger():
        return {"status": False, "message": "Cannot trigger camera"}
    if image_captured_event.wait(10):
        
        current_image = image_captured_data
    else:
        # random select image from calibration_images folder
        list_of_files = os.listdir('color_corection_images')
        if not os.path.exists('color_corection_images'):
            os.mkdir('color_corection_images')
        selected_image = random.choice(list_of_files)
        large_image = cv2.imread(os.path.join('color_corection_images', selected_image))
        current_image = large_image
    image_captured_event.clear()
    #capture_event.clear()
    image_result = current_image.copy()
    
    #large_image = cv2.imread('test_img.png')
    try:
        mt,mr= color_correction_tool.add_calibrate_image(image_result)
        _,img_encoded  = cv2.imencode('.jpg', image_result)
        b64Image = base64.b64encode(img_encoded).decode('utf-8')
        return {"image": b64Image, 'status': True}
    except Exception as e:
        _,img_encoded  = cv2.imencode('.jpg', image_result)
        b64Image = base64.b64encode(img_encoded).decode('utf-8')
        print(e)
        return {"image": b64Image, 'status': False}

class Handler:
    def __init__(self):
        self.shutdown_event = threading.Event()

    def __call__(self, cam: Camera, stream: Stream, frame: Frame):
        global image_captured_data
        global image_captured_event
        if frame.get_status() == FrameStatus.Complete:
            print('{} acquired {}'.format(cam, frame), flush=True)
            # Convert frame if it is not already the correct format
            if frame.get_pixel_format() == PixelFormat.Bgr8:
                display = frame
            else:
                # This creates a copy of the frame. The original `frame` object can be requeued
                # safely while `display` is used
                display = frame.convert_pixel_format(PixelFormat.Bgr8)
            image_captured_data = display.as_opencv_image()
            image_captured_event.set()
        cam.queue_frame(frame)
def SoftwareTrigger():
    global connected_camera
    global image_captured_event
    image_captured_event.clear()
    try:
        connected_camera.TriggerSoftware.run()
        return True
    except Exception as e:
        print("Cannot trigger camera")
        return False
def camera_thread_func():
    global connected_camera               
    global image_captured_data
    global capture_event
    with VmbSystem.get_instance():
        while True:
            with get_camera(None) as cam:
                print('Camera connected', flush=True)
                if cam is None:
                    print('Waiting for camera to be connected...', flush=True)
                    time.sleep(10)
                    continue
                #set camera to software trigger mode
                cam.TriggerMode.set('On')
                cam.TriggerSource.set('Software')
                connected_camera = cam
                handler = Handler()
                try:
                    # Start Streaming with a custom a buffer of 10 Frames (defaults to 5)
                    cam.start_streaming(handler=handler, buffer_count=10)
                    handler.shutdown_event.wait()

                finally:
                    cam.stop_streaming()
                # while capture_event.wait():               
                #     for frame in cam.get_frame_generator(limit=1, timeout_ms=5000):
                #         print('Got {}'.format(frame), flush=True)
                #         bgr_frame = frame.convert_pixel_format(PixelFormat.Bgr8)
                #         image_captured_data = bgr_frame.as_opencv_image()
                #     image_captured_event.set()
                #     capture_event.clear()
                #     print('Image captured', flush=True)
            print('Camera disconnected. Reconnecting...', flush=True)
            time.sleep(10)
def uvicorn_run():
    uvicorn.run(app, host="localhost", port=8000)
if __name__ == "__main__":
    threading.Thread(target=camera_thread_func).start()
    uvicorn_run()
    
