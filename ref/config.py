
## BASIC DATASET PARAMETERS
DATASET_BASE_PATH = "/gdrnpp_bop2022/datasets/Datensatzgenerator"
DATASET_NAME = "DFT_Labor" # Name of the Dataset (folder name)
DATASET_NAME = "DFT_Labor_Marker_Pferd" # Name of the Dataset (folder name)
DATASET_REF_KEY = "dft_labor"

#DATASET_TRAIN = "train"
DATASET_TRAIN = "train"
#DATASET_TEST = "test"
DATASET_TEST = "test"

DATASET_TRAIN_NUM_TRAIN_SCENES = 1#20 # !!!!!!!!!!!!!!!! HIER SPÄTER NOCH DIE AUTOMATISIERUNG IN DET/YOLOX/DATA/DATASETS/DFT_LABOR.PY ZEILE 72 self.scenes = [f"{i:06d}" for i in range(config.DATASET_TRAIN_NUM_SCENES)]
DATASET_TEST_NUM_SCENES = 1#5

## 3D-MODELS PARAMETERS
MODELS_NAME = ["001_Marker","002_Pferd"] # [1.3360, -0.5000, 3.5105]
MODELS_DIAMETERS = [148.76,196.682] # object diameters? (in mm) !!!! später hier automatisieren aus der models_info.json die diameter ziehen

## CAMERA PARAMETERS
CAMERA_WIDTH = 1280 #640 # Width of camera image
CAMERA_HEIGHT = 720 #480 # Height of camera image

CAMERA_CLIP_START = 0.25 # Camera start clipping distance
CAMERA_CLIP_END = 6.0 # Camera end clipping distance

CAMERA_FOCAL_X = 441.14288330078127 #1066.778 # focal length X-axis
CAMERA_FOCAL_Y = 441.14288330078127 #1067.487 # focal length Y-axis

CAMERA_C_X = 640.0 #312.9869 # camera principal point X-axis
CAMERA_C_Y = 360.0 #241.3109 # camera principal point Y-axis

CAMER_DEPTH_SCALE = 5.0 # !!!!! MUSS NOCH IN DEN SKRIPTEN ÜBERNOMMEN WERDEN

CAMERA_MATRIX = [[ CAMERA_FOCAL_X , 0.0 , CAMERA_C_X ],[ 0.0 , CAMERA_FOCAL_Y , CAMERA_C_Y] , [ 0.0 , 0.0 , 1.0 ]]

## TRAINING PARAMETERS
TRAIN_YOLOX_OPTIMIZER_LR = 0.001 # learning rate
TRAIN_YOLOX_OPTIMIZER_WEIGHT_DECAY = 0.0 # Momentum
TRAIN_YOLOX_EPOCHS = 30 # epochs
TRAIN_YOLOX_NO_AUGH_EPOCHS = 15 # epochs without augmenttation
TRAIN_YOLOX_DATALOADER_TEST_BATCHSIZE = 16 # dataloader test total batch size
TRAIN_YOLOX_DATALOADER_TEST_WORKERS = 4 # number of workers for dataloader test

TRAIN_OUTPUT_DIR = "/gdrnpp_bop2022/output"

TRAIN_INIT_CHECKPOINT = "pretrained_models/yolox/yolox_x.pth"

TRAIN_GDRNPP_DATALOADER_WORKERS = 1 #32 # number of workers for dataloader
TRAIN_GDRNPP_BATCH_SIZE = 1 #16 # images per batch

# OTHER SETTINGS (no need to change this)
DATASET_TRAIN_PBR = DATASET_TRAIN
DATASET_TRAIN_RENDER = DATASET_TRAIN
DATASET_TRAIN_REAL = DATASET_TRAIN

DATASET_MODELS_DIR_NAME = "models"
DATASET_MODELS_FINE_DIR_NAME = "models_fine"
DATASET_MODELS_EVAL_DIR_NAME = "models_eval"
DATASET_MODELS_SCALED_DIR_NAME = "models_rescaled"


DATASET_MODULE_NAME = "dft_labor"
