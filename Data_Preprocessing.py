import numpy as np
import nibabel as nib
from scipy.ndimage import binary_erosion, binary_dilation, label
import pydicom
import os

def threshold_mask(ct_array, lower=-50, upper=100):
    return np.logical_and(ct_array >= lower+1000, ct_array <= upper+1000)

def spherical_structuring_element(radius_voxel):
    L = np.arange(-radius_voxel, radius_voxel+1)
    Z, Y, X = np.meshgrid(L, L, L, indexing='ij')
    dist = np.sqrt(X**2 + Y**2 + Z**2)
    return dist <= radius_voxel

def apply_morphology(mask, voxel_spacing, operation='erode'):
    # 根據體素大小設計 kernel 半徑
    # kernel_size_mm = max(voxel_spacing[2],4)
    # x, y, z = [max(1, int(round(kernel_size_mm // voxel_spacing[i]))) for i in range(3)]
    # struct = np.ones((x, y, z), dtype=np.uint8)
    structure = spherical_structuring_element(max(voxel_spacing[2],4))
    if operation == 'erode':
        return binary_erosion(mask, structure=structure)
    elif operation == 'dilate':
        return binary_dilation(mask, structure=structure)
    else:
        raise ValueError("Operation must be 'erode' or 'dilate'")

def keep_largest_island(mask, min_size=1000):
    structure = np.zeros((3, 3, 3), dtype=int)
    structure[1, 1, :] = 1
    structure[1, :, 1] = 1
    structure[:, 1, 1] = 1
    labeled, num = label(mask, structure=structure)
    
    if num == 0:
        return np.zeros_like(mask)

    counts = np.bincount(labeled.ravel())
    counts[0] = 0  # 忽略背景

    largest_label = np.argmax(counts)

    if counts[largest_label] < min_size:
        return np.zeros_like(mask)
    
    return (labeled == largest_label).astype(np.uint8)

def run_pipeline(input_path, output_ct_path, volume2):
    # Step 1: 載入 CT
    img = input_path
    ct_array = img.get_fdata()
    spacing = img.header.get_zooms()  
    # Step 2: 閾值範圍 [-50, 100]
    mask = threshold_mask(ct_array)
    # Step 3: 侵蝕
    mask = apply_morphology(mask, spacing, operation='erode')
    # Step 4: 最大島嶼（至少 1000 voxel）
    mask = keep_largest_island(mask, min_size=1000)
    # Step 5: 膨脹（同樣 kernel）
    mask = apply_morphology(mask, spacing, operation='dilate')
    # Step 6: 原圖 AND Mask
    extracted = volume2 * mask
    # Step 7: 儲存結果
    extracted_nii = nib.Nifti1Image(extracted.astype(np.float32), affine=img.affine, header=img.header)

    nib.save(extracted_nii, output_ct_path)
    print(f"✅ 儲存完成\n Extracted CT: {output_ct_path}")
    return extracted_nii

def load_dicom_series(dicom_files):
    # 讀入 DICOM slices 並排序
    slices = [pydicom.dcmread(f) for f in dicom_files]
    slices.sort(key=lambda x: int(1*x.InstanceNumber))  # 根據 Z 軸排序

    # 基本資訊
    # 取得影像數據與 slice thickness
    slice_thickness = slices[0].SliceThickness  # 取得 slice thickness
    pixel_spacing = slices[0].PixelSpacing  # 取得 pixel spacing (dx, dy)
   
    # 轉換為 NumPy 陣列
    image_data = np.stack([s.pixel_array for s in slices], axis=-1)
    # image_data = image_data.astype(np.int16)  # 轉換為 float32
    image_data = np.rot90(image_data, k=-1, axes=(0,1))  # 旋轉至正確方向  第三章旋轉
     # 轉換 HU 值
    intercept = slices[0].RescaleIntercept
    slope = slices[0].RescaleSlope
    image_data2 = image_data * slope + intercept
# 取得 WL / WW 參數
    print(f"Window Center: {slices[0].WindowCenter}, Window Width: {slices[0].WindowWidth}")
    window_center = float(slices[0].WindowCenter[0])  
    window_width = float(slices[0].WindowWidth[0])    
    img_min = window_center - (window_width / 2)
    img_max = window_center + (window_width / 2)

    # 套用 windowing
    image_data2 = np.clip(image_data2, img_min, img_max)
    image_data2 = ((image_data2 - img_min) / (img_max - img_min) * 255).astype(np.uint8)
    # image_data2 = np.rot90(image_data2, k=-1, axes=(0,1))  # 旋轉至正確方向  第三章旋轉
    # 建立 NIfTI 影像
    affine = np.diag([pixel_spacing[0], pixel_spacing[1], slice_thickness, 1])  # 轉換座標
    # 組 affine 矩陣
    spacing = np.array([pixel_spacing[0], pixel_spacing[1], slice_thickness])
    return image_data, image_data2, affine ,spacing

def run_total(dir):
    dicom_dir = dir# 輸入資料夾路徑
    dicom_files = [os.path.join(dicom_dir, f) for f in os.listdir(dicom_dir) if f]
    
    volume, volume2, affine, spacing= load_dicom_series(dicom_files)
    nii = nib.Nifti1Image(volume, affine=affine)
    header = nii.header
    header.set_data_dtype(np.int16)
    header.set_zooms((spacing[0], spacing[1], spacing[2]))
    output_ct = dir+"/original.nii.gz"# 輸出檔案名稱
    
    original = run_pipeline(nii, output_ct, volume2)
    return original
if __name__ == "__main__":
    # 請修改為你的檔案路徑
    dicom_dir = "doctor_dcm/1-000325562H"# 輸入資料夾路徑
    dicom_files = [os.path.join(dicom_dir, f) for f in os.listdir(dicom_dir) if f.startswith("CT")]
    
    volume, volume2, affine, spacing= load_dicom_series(dicom_files)
    nii = nib.Nifti1Image(volume, affine=affine)
    header = nii.header
    header.set_data_dtype(np.int16)
    header.set_zooms((spacing[0], spacing[1], spacing[2]))
    output_ct = "original.nii.gz"# 輸出檔案名稱
    
    run_pipeline(nii, output_ct, volume2)
