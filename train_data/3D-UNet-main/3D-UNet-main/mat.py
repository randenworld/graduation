import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from matplotlib.widgets import Slider
import os
def mat(dir):
    # 載入 3D CT 影像 (NIfTI 格式)
    # ct_image = nib.load(dir+"/merged_labels.nii.gz")
    # ct_image = nib.load("train_data/3D-UNet-main/3D-UNet-main/predict_nii/output_segmentation_orignal_1.nii.gz")
    ct_image = nib.load(dir)
    # print("體素尺寸 (Voxel size):", ct_image.header.get_zooms())  # (dx, dy, dz)
    # print(ct_image)
    ct_array = ct_image.get_fdata()  # 轉換為 NumPy 陣列，形狀 (Depth, Height, Width)
    print(ct_image.shape)
    print(np.unique(ct_array))

    depth, height, width = ct_array.shape

    # 初始化切片索引（初始為中間切片）
    axial_idx = depth // 2
    coronal_idx = height // 2
    sagittal_idx = width // 2

    # 創建圖形和子圖
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 顯示初始切片
    axial_img = axes[0].imshow(ct_array[axial_idx, :, :], cmap="gray")
    coronal_img = axes[1].imshow(ct_array[:, coronal_idx, :], cmap="gray")
    sagittal_img = axes[2].imshow(ct_array[:, :, sagittal_idx], cmap="gray")

    # 設定標題
    axes[0].set_title("Axial ")
    axes[1].set_title("Coronal ")
    axes[2].set_title("Sagittal ")

    # 移除座標軸
    for ax in axes:
        ax.axis("off")

    # 創建滑桿
    axcolor = 'lightgoldenrodyellow'
    axial_slider_ax = fig.add_axes([0.2, 0.02, 0.65, 0.02], facecolor=axcolor)
    coronal_slider_ax = fig.add_axes([0.2, 0.06, 0.65, 0.02], facecolor=axcolor)
    sagittal_slider_ax = fig.add_axes([0.2, 0.10, 0.65, 0.02], facecolor=axcolor)

    axial_slider = Slider(axial_slider_ax, "Axial", 0, depth-1, valinit=axial_idx, valstep=1)
    coronal_slider = Slider(coronal_slider_ax, "Coronal", 0, height-1, valinit=coronal_idx, valstep=1)
    sagittal_slider = Slider(sagittal_slider_ax, "Sagittal", 0, width-1, valinit=sagittal_idx, valstep=1)

    # 更新函數
    def update(val):
        axial_idx = int(axial_slider.val)
        coronal_idx = int(coronal_slider.val)
        sagittal_idx = int(sagittal_slider.val)
        
        axial_img.set_data(ct_array[axial_idx, :, :])
        coronal_img.set_data(ct_array[:, coronal_idx, :])
        sagittal_img.set_data(ct_array[:, :, sagittal_idx])

        fig.canvas.draw_idle()

    # 連接滑桿事件
    axial_slider.on_changed(update)
    coronal_slider.on_changed(update)
    sagittal_slider.on_changed(update)

    plt.show()

if __name__ == "__main__":
    # dir = "doctor_dcm" 
    # dcm_arrau=[os.path.join(dir,f) for f in os.listdir(dir) if not f.startswith("case")]
    # # print(dcm_arrau)
    # for i in dcm_arrau:
    #     print(i)
    #     if i.find("10-")!=-1 or i.find("5-")!=-1:
    #         continue
    #     mat(i+"/resampled_output.nii.gz")
    mat("doctor_dcm/1-000325562H/merged_labels.nii.gz")