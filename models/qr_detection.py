import numpy as np
import onnxruntime
from qreader import QReader
import torchvision.transforms as transforms
import cv2
QRCODE_DETECTOR_PATH = "models/qr_model_v2.onnx"
MODEL_TYPE = "m"
MIN_CONF= 0.5
NUM_CLASSES_FOR_OD_MODEL = 4
CONF_THRESHOLD_FOR_OD = 0.2
NMS_THRESHOLD_FOR_OD = 0.1
providers = [
        ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}),
        "CPUExecutionProvider"
]
# load qrcode detection model
qrcode_detector = onnxruntime.InferenceSession(
    QRCODE_DETECTOR_PATH, providers=providers
)
def nms(bboxes, scores, nms_thresh):
    """"Pure Python NMS."""
    x1 = bboxes[:, 0]  # xmin
    y1 = bboxes[:, 1]  # ymin
    x2 = bboxes[:, 2]  # xmax
    y2 = bboxes[:, 3]  # ymax

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        # compute iou
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(1e-10, xx2 - xx1)
        h = np.maximum(1e-10, yy2 - yy1)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-14)
        inds = np.where(iou <= nms_thresh)[0]
        order = order[inds + 1]

    return keep
def multiclass_nms_class_agnostic(scores, labels, bboxes, nms_thresh):
    # nms
    keep = nms(bboxes, scores, nms_thresh)

    scores = scores[keep]
    labels = labels[keep]
    bboxes = bboxes[keep]

    return scores, labels, bboxes
class PostProcessor(object):
    def __init__(self, img_size=640, strides=[8,16,32], num_classes=4, conf_thresh=0.7, nms_thresh=0.1):
        self.img_size = img_size
        self.num_classes = num_classes
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.strides = strides

    def __call__(self, predictions):
        """
        Input:
            predictions: (ndarray) [n_anchors_all, 4+1+C]
        """
        bboxes = predictions[..., :4]
        scores = predictions[..., 4:]

        labels = np.argmax(scores, axis=1)
        scores = scores[(np.arange(scores.shape[0]), labels)]

        # threshold
        keep = np.where(scores >= self.conf_thresh)
        bboxes = bboxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        # nms
        scores, labels, bboxes = multiclass_nms(
            scores, labels, bboxes, self.nms_thresh, self.num_classes, False)

        return bboxes, scores, labels
def multiclass_nms(scores, labels, bboxes, nms_thresh, num_classes, class_agnostic=False):
    if class_agnostic:
        return multiclass_nms_class_agnostic(scores, labels, bboxes, nms_thresh)
    else:
        return multiclass_nms_class_aware(scores, labels, bboxes, nms_thresh, num_classes)
def multiclass_nms_class_aware(scores, labels, bboxes, nms_thresh, num_classes):
    # nms
    keep = np.zeros(len(bboxes), dtype=np.int32)
    for i in range(num_classes):
        inds = np.where(labels == i)[0]
        if len(inds) == 0:
            continue
        c_bboxes = bboxes[inds]
        c_scores = scores[inds]
        c_keep = nms(c_bboxes, c_scores, nms_thresh)
        keep[inds[c_keep]] = 1

    keep = np.where(keep > 0)
    scores = scores[keep]
    labels = labels[keep]
    bboxes = bboxes[keep]

    return scores, labels, bboxes
def adjustResultCoordinates(bboxes, ratio_w, ratio_h, height, width):
    if len(bboxes) > 0:
        bboxes[..., [0, 2]] *= ratio_w
        bboxes[..., [1, 3]] *= ratio_h
        bboxes[..., [0, 2]] = np.clip(bboxes[..., [0, 2]], 0, width)
        bboxes[..., [1, 3]] = np.clip(bboxes[..., [1, 3]], 0, height)
    return bboxes

def resize_aspect_ratio(image, resize):
    height, width = image.shape[:2]
    target_height, target_width = resize

    aspect_ratio = height / float(width)

    target_aspect_ratio = target_height / float(target_width)

    if aspect_ratio > target_aspect_ratio:
        new_height = target_height
        new_width = int(new_height / aspect_ratio)
    else:
        new_width = target_width
        new_height = int(new_width * aspect_ratio)

    resized_image = cv2.resize(
        image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    top = 0
    bottom = target_height - new_height
    left = 0
    right = target_width - new_width
    resized_image = cv2.copyMakeBorder(
        resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, None, value=0)

    height_ratio = new_height / float(height)
    width_ratio = new_width / float(width)
    target_ratio = (height_ratio, width_ratio)

    return resized_image, target_ratio
def normalization(in_img):
    # convert ndarray into tensor
    image_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    in_img = np.array(in_img, dtype=np.uint8)

    return image_transform(in_img)
def predict(model, image, resize, postprocess):
    heigh, width = image.shape[:-1]

    img_resized, target_ratio = resize_aspect_ratio(image, resize)
    ratio_h, ratio_w = 1 / target_ratio[0], 1 / target_ratio[1]

    img_resized = normalization(img_resized).numpy()
    img_resized = img_resized[np.newaxis, ...]

    ort_inputs = {'input': img_resized}
    ort_outputs = model.run(None, ort_inputs)

    bboxes, scores, labels = postprocess(ort_outputs[0])

    boxes = adjustResultCoordinates(bboxes, ratio_h, ratio_w, heigh, width)

    return boxes, scores, labels
pp = PostProcessor(
    img_size=640, 
    strides=[8, 16, 32], 
    num_classes=NUM_CLASSES_FOR_OD_MODEL, 
    conf_thresh=CONF_THRESHOLD_FOR_OD, 
    nms_thresh=NMS_THRESHOLD_FOR_OD
)
# load qrcode reader
qrcode_reader = QReader(model_size=MODEL_TYPE, min_confidence=MIN_CONF)