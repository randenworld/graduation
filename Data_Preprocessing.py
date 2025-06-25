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
    slices = []
    for f in dicom_files:
        try:
            ds = pydicom.dcmread(f)
            slices.append(ds)
        except:
            try:
                ds = pydicom.dcmread(f, force=True)
                if not hasattr(ds.file_meta, 'TransferSyntaxUID'):
                    from pydicom.uid import ImplicitVRLittleEndian
                    ds.file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
                slices.append(ds)
            except Exception as e:
                print(f"警告：無法讀取 DICOM 檔案 {f}: {e}")
                continue
    
    try:
        slices.sort(key=lambda x: int(x.InstanceNumber))  # 根據 InstanceNumber 排序
    except AttributeError:
        try:
            slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))  # 根據 Z 座標排序
        except (AttributeError, IndexError):
            file_slice_pairs = list(zip(dicom_files, slices))
            file_slice_pairs.sort(key=lambda x: x[0])
            slices = [pair[1] for pair in file_slice_pairs]

    # 基本資訊
    # 取得影像數據與 slice thickness
    slice_thickness = slices[0].SliceThickness  # 取得 slice thickness
    # 安全處理 PixelSpacing
    pixel_spacing_raw = slices[0].PixelSpacing
    if hasattr(pixel_spacing_raw, '__len__') and len(pixel_spacing_raw) >= 2:
        pixel_spacing = [float(pixel_spacing_raw[0]), float(pixel_spacing_raw[1])]
    else:
        pixel_spacing = [float(pixel_spacing_raw), float(pixel_spacing_raw)]
   
    # 轉換為 NumPy 陣列，過濾沒有像素資料的檔案
    valid_slices = []
    for s in slices:
        try:
            # 測試是否能取得 pixel_array
            _ = s.pixel_array
            valid_slices.append(s)
        except Exception as e:
            print(f"警告：跳過沒有像素資料的切片: {e}")
            continue
    
    if not valid_slices:
        raise ValueError("沒有找到有效的 DICOM 切片資料")
        
    slices = valid_slices
    image_data = np.stack([s.pixel_array for s in slices], axis=-1)
    # image_data = image_data.astype(np.int16)  # 轉換為 float32
    image_data = np.rot90(image_data, k=-1, axes=(0,1))  # 旋轉至正確方向  第三章旋轉
     # 轉換 HU 值、
    intercept = slices[0].RescaleIntercept
    slope = slices[0].RescaleSlope
    image_data2 = image_data * slope + intercept
# 取得 WL / WW 參數
    print(f"Window Center: {slices[0].WindowCenter}, Window Width: {slices[0].WindowWidth}")
    
    # 安全處理 WindowCenter 和 WindowWidth
    window_center_raw = slices[0].WindowCenter
    window_width_raw = slices[0].WindowWidth
    
    if hasattr(window_center_raw, '__len__') and len(window_center_raw) > 0:
        window_center = float(window_center_raw[0])
    else:
        window_center = float(window_center_raw)
    
    if hasattr(window_width_raw, '__len__') and len(window_width_raw) > 0:
        window_width = float(window_width_raw[0])
    else:
        window_width = float(window_width_raw)    
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
