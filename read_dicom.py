import pydicom
import numpy as np
import matplotlib.pyplot as plt
import os
import ipywidgets as widgets
import nibabel as nib
from matplotlib.widgets import Slider
from rt_utils import RTStructBuilder
import SimpleITK as sitk
from nibabel.processing import resample_from_to

def read_dicom(nii):
    data =pydicom.dcmread(nii)
    print(data)
def transRS2nii(rs):
    rs_file =rs
    dicom_folder = rs.split("/")[0:2]
    dicom_folder = "/".join(dicom_folder)
    
    # rs.split("/")[-1]
    # 讀取 DICOM 影像
    dicom_files = ([os.path.join(dicom_folder, f) for f in os.listdir(dicom_folder) if f.endswith(".dcm") and "CT" in f])
    slices = [pydicom.dcmread(f) for f in dicom_files]
    # print(slices[0])
    slices.sort(key=lambda x: int(x.InstanceNumber))  # 依 Z 軸排序

    # 取得體素資訊
    slice_thickness = slices[0].SliceThickness  # 取得切片厚度
    pixel_spacing = slices[0].PixelSpacing  # 取得像素間距 (dx, dy)

    # 讀取 RS 檔案
    rtstruct = RTStructBuilder.create_from(dicom_series_path=dicom_folder, rt_struct_path=rs_file)

    # 取得所有 ROI 名稱
    roi_names = rtstruct.get_roi_names()

    # 創建 3D 掩碼影像
    # image_shape = (slices[0].Rows, slices[0].Columns, len(slices))  # (x, y, z)
    roi_masks = {}

    for roi_name in roi_names:
        mask_3d = rtstruct.get_roi_mask_by_name(roi_name)  # ROI 轉換為 3D mask
        # roi_masks[roi_name] = mask_3d.astype(np.uint8)  # 轉為 uint8 格式（0,1）
        mask_3d = mask_3d.astype(np.uint8)  # 轉為 uint8 格式（0,1）
        # 轉換為 NIfTI
        affine = np.diag([pixel_spacing[0], pixel_spacing[1], slice_thickness, 1])  # 創建 affine 矩陣
        nii_img = nib.Nifti1Image(mask_3d, affine)
        roi_name = roi_name.replace(" ", "_")  # 移除空格
        # 儲存 `.nii.gz`
        output_file = f"{dir}/output_{roi_name}.nii.gz"
        nib.save(nii_img, output_file)
        print(f"ROI {roi_name} 轉換完成，儲存為 {output_file}")

    print("所有 ROI 轉換完成！")
def trans2nii(dir):
    dicom_folder = dir
    dicom_files = ([os.path.join(dicom_folder, f) for f in os.listdir(dicom_folder)])

    # 讀取 DICOM 檔案
    slices = [pydicom.dcmread(f) for f in dicom_files if f.endswith(".dcm") and "CT" in f]
    slices.sort(key=lambda x: int(x.InstanceNumber))  # 根據 Z 軸排序

    # 取得影像數據與 slice thickness
    slice_thickness = slices[0].SliceThickness  # 取得 slice thickness
    pixel_spacing = slices[0].PixelSpacing  # 取得 pixel spacing (dx, dy)
   
    # 轉換為 NumPy 陣列
    image_data = np.stack([s.pixel_array for s in slices], axis=-1)
    image_data = image_data.astype(np.float32)  # 轉換為 float32
     
     # 轉換 HU 值
    intercept = slices[0].RescaleIntercept
    slope = slices[0].RescaleSlope
    image_data = image_data * slope + intercept
    image_data = np.where((image_data < 20) & (image_data > -10), image_data, -1000)
# 取得 WL / WW 參數
    window_center = float(slices[0].WindowCenter)  
    window_width = float(slices[0].WindowWidth)    
    img_min = window_center - (window_width / 2)
    img_max = window_center + (window_width / 2)

    # 套用 windowing
    image_data = np.clip(image_data, img_min, img_max)
    image_data = ((image_data - img_min) / (img_max - img_min) * 255).astype(np.uint8)
    # image_data = np.flip(image_data, axis=1)  # 翻轉至正確方向 
    # image_data = np.flip(image_data, axis=0)  # 第二章對稱 
    # image_data = np.rot90(image_data, k=2, axes=(1,2))  # 旋轉至正確方向  240反轉
    # image_data = np.rot90(image_data, k=1, axes=(0,1))  # 旋轉至正確方向  第三章旋轉
    # image_data = np.rot90(image_data, k=1, axes=(0,1))  # 旋轉至正確方向  第三章旋轉
    # 建立 NIfTI 影像
    affine = np.diag([pixel_spacing[0], pixel_spacing[1], slice_thickness, 1])  # 轉換座標
    nii_img = nib.Nifti1Image(image_data, affine)

    # 儲存為 .nii.gz
    nib.save(nii_img, dir+"/output.nii.gz")

    print(f"轉換完成，儲存為 output.nii.gz")
def resample_image(nii, new_spacing=[1.0, 1.0, 1.0]):
    # 讀取 NIfTI
    output_file = [os.path.join(nii,f) for f in os.listdir(nii) if f.endswith(".nii.gz") and f.startswith("original_")]
    output_file.sort()  # 確保順序一致
    merge_file = [os.path.join(nii,f) for f in os.listdir(nii) if f.endswith(".nii.gz") and f.startswith("merged_labels_")]
    merge_file.sort()  # 確保順序一致
    csf_images = [sitk.ReadImage(csf_name) for csf_name in merge_file]
    for idx in range(len(output_file)):
        
        image = sitk.ReadImage(output_file[idx])
        # dir = nii.split("/")[0:2]
        # dir = "/".join(dir)
        # dir+="/"
        # os.path.join(dicom_folder, f)
        # ([os.path.join(dicom_folder, f) for f in os.listdir(dicom_folder) if f.endswith(".dcm") and "CT" in f])
        # roi_names = [f for f in os.listdir(dir)if f.endswith(".nii.gz") and f.startswith("output_")]
        # roi_names = [f for f in os.listdir(dir)if f.endswith(".nii.gz") and f.startswith("original_")]
        
        # print(roi_names)
        # roi_images = [sitk.ReadImage(os.path.join(dir, roi_name)) for roi_name in roi_names]
        # print(roi_names)
        # 儲存原始資訊
        original_spacing = image.GetSpacing()
        original_size = image.GetSize()
        original_direction = image.GetDirection()
        original_origin = image.GetOrigin()
        # 計算新的尺寸
        new_size = [
            int(round(original_size[i] * (original_spacing[i] / new_spacing[i])))
            for i in range(3)
        ]

        # 設定重採樣
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(new_spacing)
        resampler.SetSize(new_size)
        resampler.SetOutputDirection(original_direction)  # 保持座標方向
        resampler.SetOutputOrigin(original_origin)  # 保持原始原點
        resampler.SetInterpolator(sitk.sitkLinear)
        # 重採樣每個 ROI 影像
        # for roi_name, roi_image in zip(roi_names, roi_images):
        #     resampled_roi = resampler.Execute(roi_image)
        #     output_file = os.path.join(dir, f"resampled_{roi_name}")
        #     sitk.WriteImage(resampled_roi, output_file)
        #     print(f"ROI {roi_name} 重新採樣並儲存為 {output_file}")
        resampled_image = resampler.Execute(csf_images[idx])
        sitk.WriteImage(resampled_image, merge_file[idx])
        sitk.WriteImage(resampler.Execute(image), output_file[idx])

        print("影像已重新採樣並儲存！")

def merge_labels(dir):
    output_file = dir+ "/merged_labels.nii.gz"

    # 讀取所有 .nii.gz 檔案
    nii_files = [f for f in os.listdir(dir) if f.endswith(".nii.gz") and  f.startswith('resampled_output_')]

    # nii_files = [f for f in os.listdir(dir) if f.endswith(".nii.gz")]
    nii_files.sort()  # 確保順序一致

    # 載入第一個影像作為基準
    first_nii = nib.load(os.path.join(dir, nii_files[1]))
    merged_data = np.zeros((400,400,304), dtype=np.int16)  # 初始化合併陣列
    lab = 1
    # 逐個合併
    for i, nii_file in enumerate(nii_files, start=1):
        
        nii_path = os.path.join(dir, nii_file)
        nii_img = nib.load(nii_path)
        nii_data = nii_img.get_fdata()
        print(nii_path)
        print(nii_data.shape)
        # 調整大小 (若有不同大小的影像)
        if nii_data.shape != merged_data.shape:
            nii_data_resized = np.zeros_like(merged_data)
            nii_data_resized[:nii_data.shape[0], :nii_data.shape[1], -nii_data.shape[2]:] = nii_data
            nii_data = nii_data_resized
        # 每個 ROI 分配不同的標籤值 (避免覆蓋)
        # merged_data[nii_data > 0] = lab
        merged_data[(nii_data > 0)] = lab
        print(lab)
        lab += 1
    
    # merged_data = np.flip(merged_data, axis=1)  # 翻轉至正確方向 
    # merged_data = np.flip(merged_data, axis=0)  # 第二章對稱 
    # merged_data = np.rot90(merged_data, k=2, axes=(1,2))  # 旋轉至正確方向  240反轉
    # merged_data = np.rot90(merged_data, k=1, axes=(0,1))  # 旋轉至正確方向  第三章旋轉
    # merged_data = np.rot90(merged_data, k=1, axes=(0,1))  # 旋轉至正確方向  第三章旋轉

    merged_nii = nib.Nifti1Image(merged_data, affine=first_nii.affine, header=first_nii.header)

    # 儲存合併後的 NIfTI
    nib.save(merged_nii, output_file)

    print(f"合併完成，儲存為: {output_file}")
def change2real(dir):
    nii_file = [f for f in os.listdir(dir) if f.endswith(".nii.gz")]

    for nii in nii_file:
        first_nii = nib.load(os.path.join(dir, nii))
        original = first_nii.get_fdata()
        original_data = np.zeros((400,400,304), dtype=np.int16)  # 初始化合併陣列
        original_data[:original.shape[0], :original.shape[1], -original.shape[2]:] = original

    # 建立新的 NIfTI 影像
        original_nii = nib.Nifti1Image(original_data, affine=first_nii.affine, header=first_nii.header)

        # 儲存合併後的 NIfTI
        nib.save(original_nii,dir+nii)

def resize_nii(dir):
    # 讀取 NIfTI 影像
    # dir = "train_data/3D-UNet-main/dataset_resize/"
    nii_file = [f for f in os.listdir(dir) if f.endswith(".nii.gz")]
    for i in nii_file:
        
        nii_img = nib.load(dir+i)
        data = nii_img.get_fdata()

        # 獲取原始尺寸
        original_shape = data.shape

        # 設定新的尺寸
        new_shape = (256, 256, 304)
        # for x in 
        # resized_data = cv2.resize(data, new_shape, interpolation=cv2.INTER_LINEAR)
        # 計算縮放因子
        scale_factors = [new_shape[i] / original_shape[i] for i in range(3)]

        # 使用 scipy 的 zoom 函數進行縮放
        from scipy.ndimage import zoom
        resized_data = zoom(data, scale_factors, order=0)  # 使用線性插值

        # 儲存新的 NIfTI 影像
        resized_nii = nib.Nifti1Image(resized_data, affine=nii_img.affine, header=nii_img.header)
        nib.save(resized_nii, dir+i)
        print(f"已儲存縮放後的影像：{i.replace('.nii.gz', '_resized.nii.gz')}")
if __name__ == "__main__":
    # transRS2nii trans2nii  將轉出的output_CSF.nii.gz=>merged_labels_編號.nii.gz  output.nii.gz=>original.nii.gz 都拉到一個資料夾  resample_image 會將x,y,z統一 再用change2real將原本的影像改成一樣大小
    #rs_name 放RS開頭的dcm檔案
    # rs_name = "doctor_dcm/9-003040477E/RS.1.2.246.352.205.5130208064676815548.1885154630307975828.dcm"
    # # read_dicom(rs_name)
    # #dir[0:x]取決資料夾層數
    # dir = rs_name.split("/")[0:2]
    # dir = "/".join(dir)
    #這個dir是全部拉到同一個資料夾的路徑
    dir = "train_data/3D-UNet-main/dataset_csf_去邊_resample_400/"
    # transRS2nii(rs_name)
    # trans2nii(dir)
    resample_image(dir)
    # merge_labels(dir)
    change2real(dir)
    # resize_nii(dir)
