
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
DATASET_PATH = 'train_data/3D-UNet-main/dataset_csf_去邊_resample_400'  # Path to the dataset
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
TRAIN_VAL_TEST_SPLIT = [0.8, 0.2, 0.0]
PATIENCE = 10 #每次訓練多少個epoch後，若val_loss沒有下降，則停止訓練
MODEL_PATH = "" #訓練好的模型權重
MODEL_PATH_LABEL = "checkpoints/checkpoint_20250409_233437/0410_epoch7valLoss_0.769835.pth" #訓練好的模型權重
LR = 8e-1 #初始學習率
FACTOR = 0.5 #LR更新率
FIRST_CHANNEL = 32
NUMBER_WORKERS = 4
SPLIT_SEED = 42
TRAINING_EPOCH = 1000
TRAIN_BATCH_SIZE = 1
VAL_BATCH_SIZE = 1
TEST_BATCH_SIZE = 1
# TRAIN_CUDA = True
TRAIN_CUDA = False
# BCE_WEIGHTS = [0.05, 0.1, 0.4, 0.12, 0.1, 0.08, 0.15, 0.18, 0.18, 0.1]
# BCE_WEIGHTS = [1,1,1,1,1,1,1,1,1,1]
BCE_WEIGHTS = [0.004,0.996]
