import os
import torch
import nibabel as nib
import numpy as np
from unet3d import UNet3D  # 載入 U-Net3D 模型
import torch.nn.functional as F
import mat
from config import (TRAIN_CUDA,MODEL_PATH_LABEL)
from scipy.ndimage import binary_erosion, binary_dilation

def opening_3d(image: np.ndarray, structure: np.ndarray = None) -> np.ndarray:
    """對 3D NumPy 陣列執行開運算 (Opening: 先腐蝕再膨脹)

    參數:
        image (np.ndarray): 3D 二值影像 (0 或 1)
        structure (np.ndarray): 3D 結構元素 (預設為 3x3x3 立方體)

    返回:
        np.ndarray: 開運算後的 3D 影像
    """
    if structure is None:
        structure = np.ones((3, 3, 3), dtype=np.uint8)  # 預設為 3x3x3 立方體結構元素
    image = binary_erosion(image, structure)
    image = binary_dilation(image, structure)
    # image = binary_erosion(image, structure)
    
    return image.astype(np.uint8)
def labeling_data(path=""):
    # === 參數設定 ===
    if(path != ""):
        MODEL_PATH = path
    else:
        MODEL_PATH = MODEL_PATH_LABEL  # 訓練好的模型權重
    INPUT_NII = "train_data/3D-UNet-main/dataset_去邊_覆蓋/orignal_1.nii.gz"  # 輸入 CT 影像
    path = MODEL_PATH.split("/")[-1]
    path = path.split(".")[-3]
    way = INPUT_NII.split("/")[-1]
    way = way.split(".")[-3]
    OUTPUT_NII = "train_data/3D-UNet-main/3D-UNet-main/predict_nii/output_segmentation_"+way+"_"+path+".nii.gz"  # 儲存標記結果
    NUM_CLASSES = 2  # 多分類標記的類別數，請根據訓練時的設定修改
    IN_CHANNELS = 1  # 輸入影像的通道數

    # === 1. 載入 U-Net3D 模型 ===
    if torch.cuda.is_available() and TRAIN_CUDA:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet3D(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    # === 2. 載入 CT 影像 ===
    ct_nifti = nib.load(INPUT_NII)  # 讀取 NIfTI 檔案
    ct_data = ct_nifti.get_fdata()  # 取得 numpy 陣列
    ct_data = np.expand_dims(ct_data, axis=0)  # 增加通道數 (1, H, W, D)
    ct_tensor = torch.tensor(ct_data, dtype=torch.float32).unsqueeze(0).to(device)  # 轉換為張量 (1, 1, H, W, D)

    # === 3. 進行模型推論 ===
    with torch.no_grad():
        output = model(ct_tensor)  # 取得模型輸出 (1, NUM_CLASSES, H, W, D)
        predicted = torch.argmax(output, dim=1)  # 取得最大類別索引 (1, H, W, D)
        # print(predicted.shape)
        # print(predicted)

    # === 4. 儲存標記結果 ===
    predicted_np = predicted.cpu().numpy().astype(np.uint8).squeeze(0)  # 轉回 NumPy 陣列
    # predicted_np = opening_3d(predicted_np,np.ones((3, 3, 3), dtype=np.uint8))  # 開運算處理


    seg_nifti = nib.Nifti1Image(predicted_np, affine=ct_nifti.affine, header=ct_nifti.header)  # 建立 NIfTI 影像
    if seg_nifti is None:
        print("錯誤: seg_nifti 為 None，無法儲存標記結果！")
    else:
        nib.save(seg_nifti, OUTPUT_NII)
    # 確保輸出目錄存在

    output_dir = os.path.dirname(OUTPUT_NII)  # 取得路徑的資料夾部分
    os.makedirs(output_dir, exist_ok=True)  # 建立資料夾（若已存在則不報錯）
    nib.save(seg_nifti, OUTPUT_NII)  # 儲存標記結果

    print(f"✅ 分割完成！標記已儲存為: {OUTPUT_NII}")
    mat.mat(OUTPUT_NII)

if __name__ == "__main__":
    labeling_data()