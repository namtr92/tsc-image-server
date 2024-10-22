from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Tuple

class AppSettings():
    APP_NAME: str = 'Printify APIs'
    
    # deep learning models settings
    OD_MODEL_PATH: str
    QRCODE_DETECTOR_PATH: str
    SLEEVES_MODEL_PATH: str

    NUM_CLASSES_FOR_OD_MODEL: int
    CONF_THRESHOLD_FOR_OD: float
    NMS_THRESHOLD_FOR_OD: float

    GARMENT_SEGMENTATION_MODEL_PAHT: str
    MASK_THRESHOLD: float

    EDGE_MODEL_PATH: str

    # qrcode reader settings
    MODEL_TYPE: str
    MIN_CONF: float

    # kornia setting
    
    METHOD_TYPE_FOR_ALIGN_IMG = str
    DESIGN_MATCHING_RESIZE = int

    # camera info settings
    CAMERA_MATRIX_PATH: str
    DIST_COEFFS: str    

    # template base url for download templates
    TEMPLATE_BASE_URL: str
    API_KEY_FOR_DOWNLOAD: str
    IMG_RESIZE: Tuple[int, int]
    

    class Config:
        env_file = '.env'

app_settings = AppSettings()