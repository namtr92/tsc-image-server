import numpy as np
import cv2
import glob
import os
import json
import colour
from colour_checker_detection import detect_colour_checkers_segmentation
def get_color_matrix(img):
    COLOUR_CHECKER_IMAGES = [colour.cctf_decoding(img / 255.)]

    SWATCHES = []
    for image in COLOUR_CHECKER_IMAGES:
        for colour_checker_data in detect_colour_checkers_segmentation(
            image, additional_data=True):
            
            swatch_colours, swatch_masks, colour_checker_image = (
                colour_checker_data.values)
            #print(colour_checker_image)
            SWATCHES.append(swatch_colours)
            # #print(colour_checker_image.shape)
            # for corner in colour_checker_image:
            #     print(corner.shape)
            #     cv2.circle(img, (int(corner[0]), int(corner[3])), 5, (0, 255, 0), -1)

    D65 = colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D65']
    REFERENCE_COLOUR_CHECKER = colour.CCS_COLOURCHECKERS['ColorChecker24 - After November 2014']

    REFERENCE_SWATCHES = colour.XYZ_to_RGB(colour.xyY_to_XYZ(list(REFERENCE_COLOUR_CHECKER.data.values())), REFERENCE_COLOUR_CHECKER.illuminant, D65, colour.RGB_COLOURSPACES['sRGB'].matrix_XYZ_to_RGB)

    return SWATCHES[0], REFERENCE_SWATCHES

def draw_chessboard_corners(img, chessboard_size, corners):
    # Draw lines with a specific thickness between corners
    thickness = 3  # Set the desired line thickness
    color = (0, 255, 0)  # Green color for the lines

    # Draw the chessboard with customized lines between the corners
    for i in range(chessboard_size[0] - 1):
        for j in range(chessboard_size[1] - 1):
            start_corner = tuple(map(int, corners[i * chessboard_size[0] + j][0]))
            end_corner = tuple(map(int, corners[i * chessboard_size[0] + j + 1][0]))
            cv2.line(img, start_corner, end_corner, color, thickness)

            start_corner_vert = tuple(map(int, corners[j * chessboard_size[0] + i][0]))
            end_corner_vert = tuple(map(int, corners[(j + 1) * chessboard_size[0] + i][0]))
            cv2.line(img, start_corner_vert, end_corner_vert, color, thickness)
class ColorCorrectionTool:
    def __init__(self):
        self.M_T= None
        self.M_R = None
        self.tvecs=None
        self.save_dir = "calibration_data"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.load_calibration_data()
    def load_calibration_data(self):
        try:
            self.M_T = np.load(os.path.join(self.save_dir,"M_T.npy"))
            self.M_R = np.load(os.path.join(self.save_dir,"M_R.npy"))
            return self.M_T, self.M_R
        except:
            pass
    def add_calibrate_image(self,img:np.ndarray)->np.ndarray:
        
        M_T,M_R = get_color_matrix(img)
        self.M_T = M_T
        self.M_R = M_R
        np.save(os.path.join(self.save_dir,"M_T.npy"), M_T)
        np.save(os.path.join(self.save_dir,"M_R.npy"), M_R)
        return M_T,M_R
    
class CalibrationTool:
    def __init__(self, pattern_size=(6, 6), resize_ratio=4):
        self.pattern_size = pattern_size
        self.resize_ratio = resize_ratio
        self.objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
        self.objpoints = []
        self.imgpoints = []
        self.image_size=None
        self.camera_matrix = None
        self.distortion_coeffs= None
        self.rvecs= None
        self.tvecs=None
        self.save_dir = "calibration_data"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.load_calibration_data()
    def load_calibration_data(self):
        try:
            self.camera_matrix = np.load(os.path.join(self.save_dir,"camera_matrix.npy"))
            self.distortion_coeffs = np.load(os.path.join(self.save_dir,"distortion_coeffs.npy"))
            self.rvecs = np.load(os.path.join(self.save_dir,"rvecs.npy"))
            self.tvecs = np.load(os.path.join(self.save_dir,"tvecs.npy"))
            return self.camera_matrix, self.distortion_coeffs, self.rvecs, self.tvecs
        except:
            pass
    def draw_chessboard_corners(self, img, corners, ret):
        cv2.drawChessboardCorners(img, self.pattern_size, corners, ret)

    def add_calibrated_points(self, corners):
        corners = np.array(corners).reshape(-1,36, 2)
        print(corners.shape)
        for corner in corners:
            self.objpoints.append(self.objp)
            print(corner.shape)
            self.imgpoints.append(corner.astype(np.float32))
    def add_calibrate_image(self,img:np.ndarray)->np.ndarray:
        
        img = cv2.resize(img, (img.shape[1]//self.resize_ratio, img.shape[0]//self.resize_ratio))  # Resize to speed up detection
        print(img.shape)
        self.image_size = (img.shape[1], img.shape[0])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        ret, corners = cv2.findChessboardCorners(gray, self.pattern_size, None) #conners shape (36,1,2)
        print(corners.shape)
        if ret:
            draw_chessboard_corners(img, self.pattern_size, corners)
            #cv2.drawChessboardCorners(gray, self.pattern_size, corners, ret)
            return img, corners.tolist()
        return None,None
    def calibrate_camera(self):
        ret, camera_matrix, distortion_coeffs, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints,self.image_size, None, None)
        self.camera_matrix = camera_matrix
        self.distortion_coeffs = distortion_coeffs
        self.rvecs = rvecs
        self.tvecs = tvecs
        np.save(os.path.join(self.save_dir,"camera_matrix.npy"), camera_matrix)
        np.save(os.path.join(self.save_dir,"distortion_coeffs.npy"), distortion_coeffs)
        np.save(os.path.join(self.save_dir,"rvecs.npy"), rvecs)
        np.save(os.path.join(self.save_dir,"tvecs.npy"), tvecs)
        print("Camera calibration successful.")
        self.load_calibration_data()
        return camera_matrix, distortion_coeffs, rvecs, tvecs

# # Define the size of the calibration pattern (number of inner corners)
# # pattern_size = (6, 6)  # Change this to match your calibration pattern
# pattern_size = (6, 6)  # Change this to match your calibration pattern
# resize_ratio = 4
# # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)
# objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
# objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

# def draw_chessboard_corners(img, pattern_size, corners, ret):
#     cv2.drawChessboardCorners(img, pattern_size, corners, ret)
#     cv2.imshow('img', img)
#     cv2.waitKey(500)  # Adjust the delay as needed


# # Arrays to store object points and image points from all the images.
# objpoints = []  # 3d points in real world space
# imgpoints = []  # 2d points in image plane.

# # Load calibration images
# images = glob.glob('calibration_images/*.jpg')
# print(f"Found {len(images)} calibration images.")
# for fname in images:
#     img = cv2.imread(fname)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     gray = cv2.resize(gray, (gray.shape[1]//resize_ratio, gray.shape[0]//resize_ratio))  # Resize to speed up detection
#     print(f"Processing {fname}...")
#     # Find the chessboard corners
#     ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

#     # If corners are found, add object points, image points
#     if ret:
#         objpoints.append(objp)
#         imgpoints.append(corners)

#         # Draw and display the corners
#         # cv2.drawChessboardCorners(img, pattern_size, corners, ret)
#         cv2.drawChessboardCorners(gray, pattern_size, corners, ret)
#         # cv2.imshow('img', img)
#         cv2.imshow('img', gray)
#         cv2.waitKey(500)  # Adjust the delay as needed

#cv2.destroyAllWindows()

# # Calibrate camera
# ret, camera_matrix, distortion_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# # Save calibration data
# np.save("camera_matrix.npy", camera_matrix)
# np.save("distortion_coeffs.npy", distortion_coeffs)
# np.save("rvecs.npy", rvecs)
# np.save("tvecs.npy", tvecs)

# print("Camera calibration successful.")
