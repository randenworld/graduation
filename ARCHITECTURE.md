# 系統架構與技術說明

## 整體架構概述

本系統採用模組化設計，將醫學影像處理流程分解為多個獨立但相互協作的模組，確保系統的可維護性和擴展性。

```
┌─────────────────────────────────────────────────────────────┐
│                     腦部影像自動分割系統                      │
├─────────────────────────────────────────────────────────────┤
│  輸入層: DICOM 檔案 + RT Structure                          │
├─────────────────────────────────────────────────────────────┤
│  預處理層: 格式轉換 + 影像增強 + 資料清理                    │
├─────────────────────────────────────────────────────────────┤
│  深度學習層: 2D/2.5D U-Net + 3D U-Net                      │
├─────────────────────────────────────────────────────────────┤
│  後處理層: 形態學操作 + 連通性分析                           │
├─────────────────────────────────────────────────────────────┤
│  分析層: 體積計算 + 對稱性分析 + 臨床指標                    │
├─────────────────────────────────────────────────────────────┤
│  輸出層: NIfTI 分割結果 + CSV 報告 + 視覺化                 │
└─────────────────────────────────────────────────────────────┘
```

## 核心模組詳細說明

### 1. 配置管理模組 (`config.py`)

**功能**: 集中管理系統的所有配置參數和常數定義

```python
# 主要配置項目
DATASET = 'Dataset'                    # 預設資料集路徑
CLASS_TYPE = 7                         # 分類類別數量
EPOCH = 200                           # 訓練輪數
BATCH_SIZE = 16                       # 批次大小
INPUT_CHANNELS = 3                    # 輸入通道數

# 腦部結構類別對應表
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

**設計特點**:
- 統一管理避免硬編碼
- 支援不同資料集的參數調整
- 醫學專業的解剖結構分類

### 2. DICOM 處理模組 (`read_dicom.py`)

**功能**: 處理醫學影像的標準格式 DICOM，轉換為可處理的 NIfTI 格式

#### 核心函數架構

```python
def read_dicom(dicom_dir):
    """
    讀取 DICOM 序列檔案
    
    處理流程:
    1. 遍歷目錄中的所有 DICOM 檔案
    2. 讀取並驗證 DICOM 檔案有效性
    3. 依據 InstanceNumber 或 SliceLocation 排序
    4. 提取重要的影像參數
    
    Returns:
        - 影像陣列 (numpy array)
        - 像素間距 (pixel spacing)
        - 切片厚度 (slice thickness)
        - 影像方向 (orientation)
    """
    
def trans2nii(dicom_path, output_path):
    """
    DICOM 轉 NIfTI 格式
    
    特殊處理:
    - 窗寬窗位調整 (Window Width/Level)
    - 像素值正規化
    - 座標系統轉換 (LPS to RAS)
    - 仿射矩陣計算
    """
    
def resample_image(image, target_spacing=(1.0, 1.0, 1.0)):
    """
    影像重採樣統一解析度
    
    目的:
    - 確保所有影像具有一致的體素大小
    - 改善模型訓練的穩定性
    - 標準化臨床測量單位
    """
```

#### 技術特點

1. **多格式支援**: 
   - Standard DICOM 檔案
   - Compressed DICOM (JPEG, JPEG 2000)
   - Multi-frame DICOM

2. **資料完整性檢查**:
   - 檔案完整性驗證
   - 序列連續性檢查
   - 像素資料有效性驗證

3. **標準化處理**:
   - HU 值校正 (Hounsfield Units)
   - 空間座標統一
   - 影像方向標準化

### 3. 資料預處理模組 (`Data_Preprocessing.py`)

**功能**: 針對腦部 CT 影像進行專業的醫學影像預處理

#### 核心演算法

```python
def threshold_mask(image, lower_bound=-50, upper_bound=100):
    """
    HU 值閾值過濾
    
    醫學依據:
    - 腦部軟組織 HU 值範圍: 0-100
    - 腦脊髓液 HU 值範圍: 0-15
    - 排除骨骼 (>100) 和空氣 (<-50)
    
    Parameters:
        lower_bound: 下限閾值 (預設 -50)
        upper_bound: 上限閾值 (預設 100)
    """
    
def spherical_structuring_element(radius):
    """
    建立球形結構元素
    
    優勢:
    - 各向同性的形態學操作
    - 保持腦部結構的圓形特徵
    - 避免方向性偏差
    """
    
def apply_morphology(binary_image, operation='erosion', iterations=1):
    """
    形態學操作
    
    操作類型:
    - erosion: 侵蝕 (去除小雜訊)
    - dilation: 膨脹 (填補小洞)
    - opening: 開運算 (先侵蝕後膨脹)
    - closing: 閉運算 (先膨脹後侵蝕)
    """
    
def keep_largest_island(binary_image, min_size=100):
    """
    保留最大連通區域
    
    醫學意義:
    - 去除小的假陽性區域
    - 保持主要解剖結構完整性
    - 提升分割結果的臨床可信度
    """
```

#### 完整處理流程

```python
def run_pipeline(dicom_path):
    """
    完整的預處理管線
    
    步驟:
    1. DICOM 讀取和驗證
    2. 影像品質評估
    3. HU 值正規化和閾值處理
    4. 形態學雜訊去除
    5. 影像增強和對比度調整
    6. 標準化輸出格式
    
    品質控制:
    - 每步驟都有品質檢查點
    - 異常情況的錯誤處理
    - 處理日誌記錄
    """
```

### 4. 深度學習模組 (`train.py`)

**功能**: 實現 2D/2.5D U-Net 模型的訓練和推論

#### U-Net 架構設計

```python
def build_2_5d_unet_model(input_shape=(512, 512, 1)):
    """
    建構專為醫學影像設計的 U-Net 模型
    
    架構特點:
    - 對稱的編碼-解碼結構
    - 跳躍連接保留細節資訊
    - 批次正規化加速收斂
    - Dropout 防止過擬合
    
    編碼路徑 (Encoder):
    - 卷積層: 32 → 64 → 128 → 256 filters
    - 池化: 每層後進行 2x2 最大池化
    - 激活函數: ReLU
    
    瓶頸層 (Bottleneck):
    - 512 filters 的卷積層
    - 最高層次的特徵提取
    
    解碼路徑 (Decoder):
    - 反卷積: 256 → 128 → 64 → 32 filters
    - 跳躍連接: 與對應編碼層串接
    - 最終輸出: Sigmoid 激活的機率圖
    """
```

#### 資料載入和增強

```python
def load_data_2_5d(data_path, slice_range=3):
    """
    2.5D 資料載入策略
    
    概念:
    - 使用連續的 3 個切片作為 RGB 通道
    - 保留部分 3D 空間資訊
    - 計算效率優於完整 3D 處理
    
    優勢:
    - 上下文資訊: 相鄰切片提供空間連續性
    - 計算效率: 比 3D 卷積更快
    - 記憶體友善: 適合大型醫學影像
    """
    
def augment_image_mask(image, mask):
    """
    醫學影像專用的資料增強
    
    增強策略:
    - 水平翻轉: 模擬左右對稱性變化
    - 旋轉: ±15 度內的小角度旋轉
    - 彈性變形: 模擬解剖變異
    - 亮度調整: 模擬不同掃描參數
    
    限制:
    - 保持醫學解剖學的合理性
    - 避免破壞組織間關係
    - 確保標籤的一致性轉換
    """
```

#### 評估指標系統

```python
def calculate_metrics(y_true, y_pred):
    """
    醫學影像分割專用評估指標
    
    Dice Coefficient:
    - 公式: 2 * |A ∩ B| / (|A| + |B|)
    - 範圍: 0-1，1 為完美分割
    - 醫學意義: 分割重疊程度
    
    IoU (Jaccard Index):
    - 公式: |A ∩ B| / |A ∪ B|
    - 嚴格的重疊評估
    - 對小目標更敏感
    
    Hausdorff Distance:
    - 邊界精確度評估
    - 最大邊界誤差測量
    - 臨床上的輪廓品質指標
    
    Volume Similarity:
    - 體積差異評估
    - 臨床測量的可靠性
    """
```

### 5. 3D 深度學習模組 (`3D-UNet-main/`)

**功能**: 完整的 3D 體積分割，保持最佳的空間一致性

#### 3D U-Net 架構 (`unet3d.py`)

```python
class UNet3D(tf.keras.Model):
    """
    3D U-Net 模型實現
    
    關鍵特點:
    - 3D 卷積操作保留完整空間資訊
    - 多尺度特徵提取
    - 跳躍連接維持細節
    - 批次正規化和 Dropout
    
    優勢:
    - 空間一致性: 同時處理整個 3D 體積
    - 上下文豐富: 完整的 3D 鄰域資訊
    - 邊界精確: 3D 連通性約束
    
    挑戰:
    - 計算複雜度: O(n³) 的複雜度
    - 記憶體需求: 大量的 3D 特徵圖
    - 訓練時間: 比 2D 方法顯著更長
    """
    
def CombinedLoss(y_true, y_pred):
    """
    組合損失函數
    
    BCE Loss:
    - 像素級別的二元分類損失
    - 對邊界敏感
    
    Dice Loss:
    - 1 - Dice Coefficient
    - 整體重疊度優化
    - 處理類別不平衡
    
    組合權重:
    - Total Loss = 0.5 * BCE + 0.5 * Dice
    - 平衡像素精度和整體重疊
    """
```

#### 資料載入系統 (`dataset.py`)

```python
class MedicalDataset(tf.keras.utils.Sequence):
    """
    醫學影像專用資料載入器
    
    特色功能:
    - 動態載入: 節省記憶體使用
    - 自動配對: 影像和標籤自動匹配
    - 多執行緒: 加速資料載入
    - 預處理整合: 即時影像處理
    
    資料分割策略:
    - 訓練集: 70%
    - 驗證集: 15%
    - 測試集: 15%
    - 確保病患級別的分割 (避免資料洩漏)
    """
```

### 6. 整合推論系統 (`befor_processes.py`)

**功能**: 完整的自動分割工作流程

#### 工作流程設計

```python
def complete_segmentation_pipeline(dicom_path):
    """
    完整分割流程
    
    階段 1: 資料準備
    - DICOM 轉換和預處理
    - 影像品質檢查
    - 標準化處理
    
    階段 2: 多模型推論
    - CSF 分割 (腦脊髓液)
    - Ventricles 分割 (全腦室)
    - Ventricle_L 分割 (左側腦室)
    - Ventricle_R 分割 (右側腦室)
    
    階段 3: 後處理計算
    - 第三腦室 = 全腦室 - 左腦室 - 右腦室
    - 第四腦室 = 全腦室 - 左腦室 - 右腦室 - 第三腦室
    - 連通性分析和雜訊去除
    
    階段 4: 品質控制
    - 解剖學合理性檢查
    - 體積範圍驗證
    - 空間一致性確認
    """
```

#### 模型管理策略

```python
def load_and_predict_with_model(model_path, input_data, structure_name):
    """
    模型載入和預測管理
    
    效能優化:
    - 模型快取: 避免重複載入
    - 批次預測: 提升 GPU 使用率
    - 記憶體管理: 及時釋放不需要的資源
    
    品質保證:
    - 模型版本檢查
    - 輸入資料驗證
    - 輸出結果檢驗
    """
```

### 7. 體積分析模組 (`transform.py`)

**功能**: 提供臨床相關的定量分析功能

#### 體積計算系統

```python
def calculate_volume_from_nii(nii_path, pixel_spacing=None):
    """
    精確的 3D 體積計算
    
    計算公式:
    Volume = Σ(binary_voxels) × voxel_volume
    voxel_volume = spacing_x × spacing_y × spacing_z
    
    單位轉換:
    - 輸入: mm³ (立方毫米)
    - 輸出: cm³ (立方公分)
    - 轉換係數: 1 cm³ = 1000 mm³
    
    誤差控制:
    - 亞體素精度的邊界處理
    - 部分體積效應修正
    - 量化誤差最小化
    """
    
def calculate_ventricle_symmetry(left_path, right_path):
    """
    腦室對稱性分析
    
    不對稱指數計算:
    AI = |V_L - V_R| / (V_L + V_R) × 100%
    
    臨床分級:
    - 高度對稱: AI < 5%
    - 中度對稱: 5% ≤ AI < 10%
    - 輕度不對稱: 10% ≤ AI < 20%
    - 明顯不對稱: AI ≥ 20%
    
    臨床意義:
    - 正常變異: AI < 10%
    - 可能異常: AI ≥ 10%
    - 需要進一步評估: AI ≥ 20%
    """
```

#### 批次分析系統

```python
def batch_analyze_brain_volumes(dataset_path):
    """
    批次體積分析
    
    功能:
    - 多病患並行處理
    - 標準化報告生成
    - 統計分析和比較
    - 異常值檢測
    
    輸出格式:
    - CSV 報告: 結構化數據
    - 統計摘要: 描述性統計
    - 視覺化圖表: 分佈和趨勢
    - 異常報告: 超出正常範圍的案例
    """
```

## 系統整合與資料流

### 資料流程圖

```
DICOM Files ─┬─→ read_dicom.py ─→ NIfTI Format
             │
             ├─→ Data_Preprocessing.py ─→ Cleaned Images
             │
             └─→ befor_processes.py ─┬─→ 2D/2.5D Models ─→ Segmentation Results
                                     │
                                     └─→ 3D Models ─→ Refined Results
                                                     │
                                                     ├─→ transform.py ─→ Volume Analysis
                                                     │
                                                     └─→ Clinical Reports
```

### 模型選擇策略

```python
def select_optimal_model(image_characteristics):
    """
    根據影像特性選擇最適合的模型
    
    決策因素:
    - 影像解析度: 高解析度偏好 3D
    - 計算資源: 有限資源使用 2D
    - 精度需求: 高精度需求使用 3D
    - 處理時間: 快速處理使用 2.5D
    
    混合策略:
    - 2D 預分割 + 3D 精修
    - 2.5D 主要分割 + 後處理優化
    - 多模型投票決策
    """
```

## 效能優化設計

### 計算效能

1. **GPU 加速**: 
   - CUDA 優化的 TensorFlow 操作
   - 混合精度訓練 (FP16)
   - 動態記憶體分配

2. **平行處理**:
   - 多執行緒資料載入
   - 批次處理優化
   - 分散式訓練支援

3. **記憶體管理**:
   - 漸進式載入大型影像
   - 智慧快取策略
   - 垃圾回收優化

### 臨床效能

1. **精度保證**:
   - 多重驗證機制
   - 交叉驗證評估
   - 臨床專家驗證

2. **穩定性**:
   - 錯誤處理和復原
   - 邊界條件處理
   - 輸入資料驗證

3. **可擴展性**:
   - 模組化設計
   - 標準化介面
   - 易於添加新功能

## 品質保證機制

### 程式品質

1. **程式碼標準**:
   - PEP 8 編碼風格
   - 類型提示 (Type Hints)
   - 文件字串 (Docstrings)

2. **測試覆蓋**:
   - 單元測試
   - 整合測試
   - 效能測試

3. **版本控制**:
   - Git 版本管理
   - 分支策略
   - 程式碼審查

### 醫學品質

1. **臨床驗證**:
   - 醫學專家審查
   - 臨床試驗驗證
   - 跨機構驗證

2. **安全性**:
   - 資料隱私保護
   - HIPAA 合規性
   - 安全傳輸協定

3. **可追溯性**:
   - 處理日誌記錄
   - 結果可重現
   - 審計軌跡

這個架構設計確保了系統在技術先進性、醫學專業性和實用性之間的平衡，為腦部影像自動分割提供了可靠的解決方案。