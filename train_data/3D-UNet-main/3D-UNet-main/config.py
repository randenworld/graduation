
"""""
Dataset configurations:
    :param DATASET_PATH -> the directory path to dataset .tar files
    :param TASK_ID -> specifies the the segmentation task ID (see the dict below for hints)
    :param IN_CHANNELS -> number of input channels
    :param NUM_CLASSES -> specifies the number of output channels for dispirate classes
    :param BACKGROUND_AS_CLASS -> if True, the model treats background as a class

"""""
# DATASET_PATH = 'train_data/3D-UNet-main/dataset_csf_resample_400'  # Path to the dataset
# DATASET_PATH = 'train_data/3D-UNet-main/dataset_csf_resample_resize'  # Path to the dataset
# DATASET_PATH = 'train_data/3D-UNet-main/dataset_csf_去邊_resample_400'  # Path to the dataset
# DATASET_PATH = 'train_data/3D-UNet-main/dataset_resize'  # Path to the dataset
# DATASET_PATH = 'train_data/3D-UNet-main/dataset_slicer_resample'  # Path to the dataset
# DATASET_PATH = 'train_data/3D-UNet-main/dataset_ROI_CSF'  # Path to the dataset
DATASET_PATH = 'train_data/3D-UNet-main/dataset_ROI_falx'
TASK_ID = 1
IN_CHANNELS = 1
NUM_CLASSES = 1
BACKGROUND_AS_CLASS = True


"""""
Training configurations:
    :param TRAIN_VAL_TEST_SPLIT -> delineates the ratios in which the dataset shoud be splitted. The length of the array should be 3.
    :param SPLIT_SEED -> the random seed with which the dataset is splitted
    :param TRAINING_EPOCH -> number of training epochs
    :param VAL_BATCH_SIZE -> specifies the batch size of the training DataLoader
    :param TEST_BATCH_SIZE -> specifies the batch size of the test DataLoader
    :param TRAIN_CUDA -> if True, moves the model and inference onto GPU
    :param BCE_WEIGHTS -> the class weights for the Binary Cross Entropy loss
"""""
TRAIN_VAL_TEST_SPLIT = [0.9, 0.1, 0.0]
PATIENCE = 10#每次訓練多少個epoch後，若val_loss沒有下降，則停止訓練
MODEL_PATH = "" #訓練好的模型權重
MODEL_PATH_LABEL = "checkpoints/checkpoint_20250506_024716/0506_epoch81valLoss_0.520698.pth" #訓練好的模型權重
PREDICT_FILE = '' #預測模型權重<en
LR = 1e-3 #初始學習率
FACTOR = 0.1 #LR更新率
ALPHA = 0.5 #loss的權重 1 BCE 0 DCIE
CLR = 20 #學習率更新的步數
FIRST_CHANNEL = 32
NUMBER_WORKERS = 2
SPLIT_SEED = 42
TRAINING_EPOCH = 1000
TRAIN_BATCH_SIZE = 1
VAL_BATCH_SIZE = 1
TEST_BATCH_SIZE = 1
TRAIN_CUDA = True
# TRAIN_CUDA = False
# BCE_WEIGHTS = [0.0001, 0.1, 0.4, 0.12, 0.1, 0.08, 0.15, 0.18, 0.18, 0.1]
BCE_WEIGHTS = [0.002,0.998]
