DATASET = 'D:/graduation/train_data/3D-UNet-main/dataset_new_ROI'

CLASS_TYPE = 7 #類別
EPOCH = 200
BATCH_SIZE = 2
INPUT = 1
CHANNEL =32
SEED = 42
MODEL_PATH = '../train_2D/log_2d/dataset_new_ROI_20250515_084807_model.keras' #訓練好的模型權重
PREDICT_FILE = '../train_data/3D-UNet-main/dataset_new_ROI/data_17/original_17.nii.gz' #預測模型權重
# PREDICT_FILE = '../train_data/3D-UNet-main/dataset_new_ROI/data_1/mask_Ventricles_1.nii.gz' #預測模型權重

TYPE_MAP = {
    0: 'Basal-cistern',
    1: 'CSF',#ok
    2: 'Falx',#ok
    3: 'Fourth-ventricle',
    4: 'Tentorium',
    5: 'Third-ventricle',
    6: 'Ventricle_L',
    7: 'Ventricle_R',
    8: 'Ventricles',#ok
}