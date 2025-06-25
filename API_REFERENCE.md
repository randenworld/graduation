# API 函數參考文件

本文件提供系統中所有主要函數和類別的詳細 API 說明，包括參數、返回值和使用範例。

## 目錄

1. [配置模組 (config.py)](#配置模組-configpy)
2. [DICOM 處理模組 (read_dicom.py)](#dicom-處理模組-read_dicompy)
3. [資料預處理模組 (Data_Preprocessing.py)](#資料預處理模組-data_preprocessingpy)
4. [訓練模組 (train.py)](#訓練模組-trainpy)
5. [體積分析模組 (transform.py)](#體積分析模組-transformpy)
6. [3D 模組 (3D-UNet-main/)](#3d-模組-3d-unet-main)

---

## 配置模組 (config.py)

### 全域常數

```python
DATASET: str = 'Dataset'
```
預設資料集路徑。

```python
CLASS_TYPE: int = 7
```
分類類別數量。

```python
TYPE_MAP: Dict[str, int]
```
腦部結構類別對應表。

**結構:**
```python
TYPE_MAP = {
    'Basal-cistern': 0,    # 基底池
    'CSF': 1,              # 腦脊髓液
    'Falx': 2,             # 大腦鐮
    'Fourth-ventricle': 3, # 第四腦室
    'Tentorium': 4,        # 小腦幕
    'Third-ventricle': 5,  # 第三腦室
    'Ventricle_L': 6,      # 左側腦室
    'Ventricle_R': 7,      # 右側腦室
    'Ventricles': 8        # 全腦室系統
}
```

### 訓練參數

```python
EPOCH: int = 200
BATCH_SIZE: int = 16
INPUT_CHANNELS: int = 3
LEARNING_RATE: float = 0.001
```

---

## DICOM 處理模組 (read_dicom.py)

### read_dicom()

```python
def read_dicom(dicom_dir: str) -> Tuple[np.ndarray, Tuple[float, float, float], float, np.ndarray]
```

讀取 DICOM 序列檔案並排序。

**參數:**
- `dicom_dir` (str): DICOM 檔案目錄路徑

**返回值:**
- `Tuple`: (影像陣列, 像素間距, 切片厚度, 方向矩陣)

**使用範例:**
```python
image_array, pixel_spacing, slice_thickness, orientation = read_dicom("/path/to/dicom/")
print(f"影像形狀: {image_array.shape}")
print(f"像素間距: {pixel_spacing}")
```

**異常:**
- `FileNotFoundError`: 找不到 DICOM 檔案
- `InvalidDicomError`: 無效的 DICOM 格式

### trans2nii()

```python
def trans2nii(dicom_path: str, output_path: str, window_width: int = 200, window_level: int = 50) -> str
```

將 DICOM 檔案轉換為 NIfTI 格式。

**參數:**
- `dicom_path` (str): DICOM 目錄路徑
- `output_path` (str): 輸出 NIfTI 檔案路徑
- `window_width` (int, optional): 窗寬設定，預設 200
- `window_level` (int, optional): 窗位設定，預設 50

**返回值:**
- `str`: 輸出檔案路徑

**使用範例:**
```python
output_file = trans2nii(
    dicom_path="/data/patient001/",
    output_path="/output/patient001.nii.gz",
    window_width=150,
    window_level=40
)
```

### resample_image()

```python
def resample_image(image_path: str, target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0), output_path: str = None) -> str
```

重採樣影像到指定的體素間距。

**參數:**
- `image_path` (str): 輸入影像路徑
- `target_spacing` (Tuple[float, float, float], optional): 目標間距 (x, y, z)，預設 (1.0, 1.0, 1.0)
- `output_path` (str, optional): 輸出路徑，若為 None 則覆蓋原檔案

**返回值:**
- `str`: 輸出檔案路徑

### transRS2nii()

```python
def transRS2nii(rt_file: str, image_file: str, output_dir: str) -> Dict[str, str]
```

將 RT Structure 轉換為 NIfTI 格式的標籤檔案。

**參數:**
- `rt_file` (str): RT Structure 檔案路徑
- `image_file` (str): 參考影像檔案路徑
- `output_dir` (str): 輸出目錄

**返回值:**
- `Dict[str, str]`: 結構名稱與輸出檔案路徑的對應字典

**使用範例:**
```python
structure_files = transRS2nii(
    rt_file="/data/RT_Structure.dcm",
    image_file="/data/CT_image.nii.gz",
    output_dir="/output/labels/"
)
print(f"產生的標籤檔案: {structure_files}")
```

---

## 資料預處理模組 (Data_Preprocessing.py)

### threshold_mask()

```python
def threshold_mask(image: np.ndarray, lower_bound: float = -50, upper_bound: float = 100) -> np.ndarray
```

基於 HU 值進行閾值處理。

**參數:**
- `image` (np.ndarray): 輸入影像陣列
- `lower_bound` (float, optional): HU 值下限，預設 -50
- `upper_bound` (float, optional): HU 值上限，預設 100

**返回值:**
- `np.ndarray`: 二值化遮罩

**使用範例:**
```python
import nibabel as nib

# 載入影像
image = nib.load("brain_ct.nii.gz")
image_data = image.get_fdata()

# 應用閾值
mask = threshold_mask(image_data, lower_bound=-30, upper_bound=80)
print(f"遮罩中的非零像素數: {np.sum(mask)}")
```

### spherical_structuring_element()

```python
def spherical_structuring_element(radius: int) -> np.ndarray
```

建立球形結構元素用於形態學操作。

**參數:**
- `radius` (int): 球形半徑（體素單位）

**返回值:**
- `np.ndarray`: 3D 球形結構元素

**使用範例:**
```python
# 建立半徑為 2 的球形結構元素
kernel = spherical_structuring_element(2)
print(f"結構元素形狀: {kernel.shape}")
print(f"結構元素中的非零元素: {np.sum(kernel)}")
```

### apply_morphology()

```python
def apply_morphology(binary_image: np.ndarray, operation: str = 'erosion', kernel_radius: int = 1, iterations: int = 1) -> np.ndarray
```

執行形態學操作。

**參數:**
- `binary_image` (np.ndarray): 二值化影像
- `operation` (str, optional): 操作類型 ('erosion', 'dilation', 'opening', 'closing')，預設 'erosion'
- `kernel_radius` (int, optional): 結構元素半徑，預設 1
- `iterations` (int, optional): 迭代次數，預設 1

**返回值:**
- `np.ndarray`: 處理後的二值化影像

**使用範例:**
```python
# 執行侵蝕操作
eroded = apply_morphology(mask, operation='erosion', kernel_radius=2, iterations=2)

# 執行膨脹操作
dilated = apply_morphology(mask, operation='dilation', kernel_radius=1, iterations=1)

# 執行開運算 (先侵蝕後膨脹)
opened = apply_morphology(mask, operation='opening', kernel_radius=1, iterations=1)
```

### keep_largest_island()

```python
def keep_largest_island(binary_image: np.ndarray, min_size: int = 100) -> np.ndarray
```

保留最大的連通區域。

**參數:**
- `binary_image` (np.ndarray): 二值化影像
- `min_size` (int, optional): 最小區域大小（體素數），預設 100

**返回值:**
- `np.ndarray`: 只包含最大連通區域的二值化影像

**使用範例:**
```python
# 保留最大連通區域
largest_component = keep_largest_island(binary_mask, min_size=200)
print(f"最大連通區域體素數: {np.sum(largest_component)}")
```

### run_pipeline()

```python
def run_pipeline(dicom_path: str, output_dir: str = None) -> str
```

執行完整的預處理流程。

**參數:**
- `dicom_path` (str): DICOM 資料夾路徑
- `output_dir` (str, optional): 輸出目錄，若為 None 則使用 dicom_path

**返回值:**
- `str`: 處理後的 NIfTI 檔案路徑

**使用範例:**
```python
processed_file = run_pipeline(
    dicom_path="/data/patient001/",
    output_dir="/processed/patient001/"
)
print(f"預處理完成: {processed_file}")
```

### run_total()

```python
def run_total(dicom_path: str) -> str
```

執行包含格式轉換的完整處理流程。

**參數:**
- `dicom_path` (str): DICOM 資料夾路徑

**返回值:**
- `str`: 最終處理檔案路徑

---

## 訓練模組 (train.py)

### build_2_5d_unet_model()

```python
def build_2_5d_unet_model(input_shape: Tuple[int, int, int] = (512, 512, 1), num_classes: int = 1) -> tf.keras.Model
```

建構 2.5D U-Net 模型。

**參數:**
- `input_shape` (Tuple[int, int, int], optional): 輸入形狀，預設 (512, 512, 1)
- `num_classes` (int, optional): 輸出類別數，預設 1（二元分割）

**返回值:**
- `tf.keras.Model`: 編譯好的 U-Net 模型

**使用範例:**
```python
# 建立標準 U-Net 模型
model = build_2_5d_unet_model((512, 512, 1))
model.summary()

# 建立多類別分割模型
multi_class_model = build_2_5d_unet_model((256, 256, 3), num_classes=5)
```

### load_data_2d()

```python
def load_data_2d(data_path: str, target_size: Tuple[int, int] = (512, 512)) -> Tuple[np.ndarray, np.ndarray]
```

載入 2D 訓練資料。

**參數:**
- `data_path` (str): 資料目錄路徑
- `target_size` (Tuple[int, int], optional): 目標影像大小，預設 (512, 512)

**返回值:**
- `Tuple[np.ndarray, np.ndarray]`: (影像陣列, 標籤陣列)

### load_data_2_5d()

```python
def load_data_2_5d(data_path: str, slice_range: int = 3) -> Tuple[np.ndarray, np.ndarray]
```

載入 2.5D 訓練資料。

**參數:**
- `data_path` (str): 資料目錄路徑
- `slice_range` (int, optional): 切片範圍，預設 3（使用前後各1片）

**返回值:**
- `Tuple[np.ndarray, np.ndarray]`: (2.5D 影像陣列, 標籤陣列)

**使用範例:**
```python
# 載入 2.5D 資料
images, labels = load_data_2_5d("/training/data/", slice_range=5)
print(f"載入影像形狀: {images.shape}")
print(f"載入標籤形狀: {labels.shape}")
```

### augment_image_mask()

```python
def augment_image_mask(image: np.ndarray, mask: np.ndarray, augmentation_params: Dict = None) -> Tuple[np.ndarray, np.ndarray]
```

對影像和遮罩進行資料增強。

**參數:**
- `image` (np.ndarray): 輸入影像
- `mask` (np.ndarray): 對應的遮罩
- `augmentation_params` (Dict, optional): 增強參數

**返回值:**
- `Tuple[np.ndarray, np.ndarray]`: (增強後的影像, 增強後的遮罩)

**增強參數範例:**
```python
augmentation_params = {
    'horizontal_flip': True,
    'vertical_flip': False,
    'rotation_range': 15,        # 度數
    'brightness_range': 0.1,     # 亮度變化範圍
    'zoom_range': 0.1           # 縮放範圍
}

aug_image, aug_mask = augment_image_mask(image, mask, augmentation_params)
```

### predict_2_5d_single_patient()

```python
def predict_2_5d_single_patient(model: tf.keras.Model, file_path: str, use_tta: bool = False, name: str = 'prediction') -> str
```

對單一病患進行 2.5D 預測。

**參數:**
- `model` (tf.keras.Model): 訓練好的模型
- `file_path` (str): 輸入 NIfTI 檔案路徑
- `use_tta` (bool, optional): 是否使用測試時間增強 (TTA)，預設 False
- `name` (str, optional): 輸出檔案名稱，預設 'prediction'

**返回值:**
- `str`: 預測結果檔案路徑

**使用範例:**
```python
# 載入模型
model = build_2_5d_unet_model((512, 512, 1))
model.load_weights("models/csf_model.keras")

# 進行預測
result_path = predict_2_5d_single_patient(
    model=model,
    file_path="/data/patient.nii.gz",
    use_tta=True,
    name="CSF"
)
print(f"預測結果保存至: {result_path}")
```

### calculate_metrics()

```python
def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> Dict[str, float]
```

計算分割評估指標。

**參數:**
- `y_true` (np.ndarray): 真實標籤
- `y_pred` (np.ndarray): 預測結果
- `threshold` (float, optional): 二值化閾值，預設 0.5

**返回值:**
- `Dict[str, float]`: 包含各種評估指標的字典

**返回的指標:**
- `dice`: Dice 係數
- `iou`: IoU (Intersection over Union)
- `precision`: 精確率
- `recall`: 召回率
- `f1_score`: F1 分數
- `accuracy`: 準確率

**使用範例:**
```python
# 計算評估指標
metrics = calculate_metrics(ground_truth, prediction, threshold=0.5)
print(f"Dice 係數: {metrics['dice']:.3f}")
print(f"IoU: {metrics['iou']:.3f}")
print(f"F1 分數: {metrics['f1_score']:.3f}")
```

---

## 體積分析模組 (transform.py)

### calculate_volume_from_nii()

```python
def calculate_volume_from_nii(nii_path: str, pixel_spacing: Tuple[float, float, float] = None) -> float
```

從 NIfTI 檔案計算 3D 體積。

**參數:**
- `nii_path` (str): NIfTI 檔案路徑
- `pixel_spacing` (Tuple[float, float, float], optional): 體素間距，若為 None 則從檔案讀取

**返回值:**
- `float`: 體積值（立方公分）

**使用範例:**
```python
# 計算腦室體積
ventricle_volume = calculate_volume_from_nii("/results/Ventricles.nii.gz")
print(f"腦室體積: {ventricle_volume:.2f} cm³")

# 指定體素間距
volume_with_spacing = calculate_volume_from_nii(
    nii_path="/results/CSF.nii.gz",
    pixel_spacing=(0.5, 0.5, 1.0)
)
```

### calculate_ventricle_symmetry()

```python
def calculate_ventricle_symmetry(left_path: str, right_path: str) -> Dict[str, Union[float, str]]
```

計算左右腦室的對稱性。

**參數:**
- `left_path` (str): 左側腦室 NIfTI 檔案路徑
- `right_path` (str): 右側腦室 NIfTI 檔案路徑

**返回值:**
- `Dict[str, Union[float, str]]`: 對稱性分析結果

**返回字典包含:**
- `left_volume`: 左側腦室體積
- `right_volume`: 右側腦室體積  
- `asymmetry_index`: 不對稱指數 (0-1)
- `asymmetry_percentage`: 不對稱百分比
- `dominant_side`: 優勢側 ('Left', 'Right', 'Symmetric')
- `symmetry_grade`: 對稱性等級 ('Highly Symmetric', 'Moderately Symmetric', 'Mildly Asymmetric', 'Markedly Asymmetric')

**使用範例:**
```python
symmetry_result = calculate_ventricle_symmetry(
    left_path="/results/Ventricle_L.nii.gz",
    right_path="/results/Ventricle_R.nii.gz"
)

print(f"左側腦室: {symmetry_result['left_volume']:.2f} cm³")
print(f"右側腦室: {symmetry_result['right_volume']:.2f} cm³")
print(f"不對稱指數: {symmetry_result['asymmetry_percentage']:.1f}%")
print(f"對稱性等級: {symmetry_result['symmetry_grade']}")
```

### batch_analyze_brain_volumes()

```python
def batch_analyze_brain_volumes(dataset_path: str, output_csv: str = None) -> pd.DataFrame
```

批次分析多個病患的腦部體積。

**參數:**
- `dataset_path` (str): 包含多個病患資料的目錄路徑
- `output_csv` (str, optional): 輸出 CSV 檔案路徑

**返回值:**
- `pd.DataFrame`: 包含所有病患分析結果的 DataFrame

**使用範例:**
```python
# 批次分析
results_df = batch_analyze_brain_volumes(
    dataset_path="/processed_patients/",
    output_csv="/results/volume_analysis.csv"
)

# 查看統計摘要
print(results_df.describe())

# 查看異常案例
abnormal_cases = results_df[results_df['asymmetry_percentage'] > 20]
print(f"異常不對稱案例數: {len(abnormal_cases)}")
```

### generate_volume_report()

```python
def generate_volume_report(patient_dir: str, output_format: str = 'dict') -> Union[Dict, str]
```

生成個別病患的詳細體積報告。

**參數:**
- `patient_dir` (str): 病患資料目錄路徑
- `output_format` (str, optional): 輸出格式 ('dict', 'json', 'html')，預設 'dict'

**返回值:**
- `Union[Dict, str]`: 根據 output_format 返回字典或格式化字串

**使用範例:**
```python
# 生成字典格式報告
report = generate_volume_report("/processed/patient001/", output_format='dict')
print(f"病患ID: {report['patient_id']}")
print(f"總腦室體積: {report['volumes']['Ventricles']} cm³")

# 生成 JSON 格式報告
json_report = generate_volume_report("/processed/patient001/", output_format='json')
with open("patient001_report.json", "w") as f:
    f.write(json_report)
```

---

## 3D 模組 (3D-UNet-main/)

### UNet3D 類別

```python
class UNet3D(tf.keras.Model):
    def __init__(self, n_classes: int = 1, dropout_rate: float = 0.3)
```

3D U-Net 模型類別。

**參數:**
- `n_classes` (int, optional): 輸出類別數，預設 1
- `dropout_rate` (float, optional): Dropout 比率，預設 0.3

**主要方法:**

#### call()
```python
def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor
```

模型前向傳播。

**參數:**
- `inputs` (tf.Tensor): 輸入張量，形狀 (batch, depth, height, width, channels)
- `training` (bool, optional): 是否為訓練模式，預設 False

**返回值:**
- `tf.Tensor`: 輸出預測結果

**使用範例:**
```python
# 建立 3D U-Net 模型
model = UNet3D(n_classes=1, dropout_rate=0.2)

# 編譯模型
model.compile(
    optimizer='adam',
    loss=CombinedLoss,
    metrics=['accuracy']
)

# 訓練模型
model.fit(train_dataset, epochs=100, validation_data=val_dataset)
```

### MedicalDataset 類別

```python
class MedicalDataset(tf.keras.utils.Sequence):
    def __init__(self, data_dir: str, batch_size: int = 2, subset: str = 'train', shuffle: bool = True)
```

醫學影像資料載入器。

**參數:**
- `data_dir` (str): 資料目錄路徑
- `batch_size` (int, optional): 批次大小，預設 2
- `subset` (str, optional): 資料子集 ('train', 'val', 'test')，預設 'train'
- `shuffle` (bool, optional): 是否打亂資料，預設 True

**主要方法:**

#### __getitem__()
```python
def __getitem__(self, index: int) -> Tuple[tf.Tensor, tf.Tensor]
```

取得指定索引的批次資料。

#### __len__()
```python
def __len__(self) -> int
```

返回批次數量。

**使用範例:**
```python
# 建立資料載入器
train_dataset = MedicalDataset(
    data_dir="/training_data/",
    batch_size=2,
    subset='train',
    shuffle=True
)

val_dataset = MedicalDataset(
    data_dir="/training_data/",
    batch_size=2,
    subset='val',
    shuffle=False
)

# 檢查資料
print(f"訓練批次數: {len(train_dataset)}")
print(f"驗證批次數: {len(val_dataset)}")

# 取得一個批次
images, masks = train_dataset[0]
print(f"影像批次形狀: {images.shape}")
print(f"遮罩批次形狀: {masks.shape}")
```

### CombinedLoss()

```python
def CombinedLoss(y_true: tf.Tensor, y_pred: tf.Tensor, alpha: float = 0.5) -> tf.Tensor
```

組合損失函數（BCE + Dice Loss）。

**參數:**
- `y_true` (tf.Tensor): 真實標籤
- `y_pred` (tf.Tensor): 預測結果
- `alpha` (float, optional): BCE 和 Dice Loss 的權重，預設 0.5

**返回值:**
- `tf.Tensor`: 組合損失值

**使用範例:**
```python
# 在模型編譯時使用
model.compile(
    optimizer='adam',
    loss=lambda y_true, y_pred: CombinedLoss(y_true, y_pred, alpha=0.3),
    metrics=['accuracy']
)

# 或作為獨立函數使用
loss_value = CombinedLoss(ground_truth, prediction, alpha=0.6)
```

## 使用注意事項

### 類型提示

本 API 文件中的所有函數都建議使用類型提示：

```python
from typing import Tuple, Dict, Union, List, Optional
import numpy as np
import tensorflow as tf
```

### 錯誤處理

大部分函數都會拋出相應的異常，使用時建議加上錯誤處理：

```python
try:
    result = calculate_volume_from_nii("/path/to/file.nii.gz")
    print(f"體積: {result:.2f} cm³")
except FileNotFoundError:
    print("檔案未找到")
except ValueError as e:
    print(f"數值錯誤: {e}")
except Exception as e:
    print(f"未知錯誤: {e}")
```

### 記憶體管理

處理大型醫學影像時，注意記憶體使用：

```python
import gc
import tensorflow as tf

# 清理記憶體
gc.collect()
tf.keras.backend.clear_session()

# 設定 GPU 記憶體增長
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
```

### 效能優化建議

1. **批次處理**: 使用批次處理提升效率
2. **GPU 加速**: 確保 TensorFlow GPU 正確設定
3. **資料預載入**: 使用 `tf.data` API 優化資料載入
4. **記憶體映射**: 對於大型檔案使用記憶體映射

這個 API 參考文件提供了系統中所有主要函數的詳細說明，可以幫助開發者和使用者更好地理解和使用系統功能。