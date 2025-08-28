# DATASET = 'D:/graduation/train_data/3D-UNet-main/dataset_new_ROI'
DATASET = 'D:/graduation/train_data/3D-UNet-main/dataset'

VERBOSE = 0
CLASS_TYPE = 9 #類別
EPOCH = 1000
BATCH_SIZE = 4
DROPOUT_RATE = 0.0
INPUT = 1
OUTPUT = 6
CHANNEL = 48
SEED = 42
MODEL_PATH = '../train_2D/log_2d/dataset_20250826_182559_model.keras' #訓練好的模型權重
PREDICT_FILE = '../train_data/3D-UNet-main/dataset_new_ROI/data_16_not_ok/original.nii.gz' #預測模型權重
# PREDICT_FILE = '../train_data/3D-UNet-main/dataset_new_ROI/data_1/mask_Ventricles_1.nii.gz' #預測模型權重

TYPE_MAP = {
    0: 'Basal-cistern',
    1: 'CSF',#ok
    2: 'Falx',#ok
    3: 'Fourth-ventricle',#OK
    4: 'Tentorium',
    5: 'Third-ventricle',#OK
    6: 'Ventricle_L',#OK
    7: 'Ventricle_R',#OK
    8: 'Ventricles',#ok
    9: 'merge', #合併
}