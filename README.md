# 腦部影像自動分割系統

這是一個基於深度學習的腦部 CT 影像自動分割系統，能夠識別和分割多種腦部解剖結構，包括腦脊髓液、腦室系統等重要結構。

## 功能特色

- **多結構分割**: 支援 9 種腦部解剖結構的自動分割
- **多模式訓練**: 提供 2D/2.5D 和 3D 深度學習模型
- **完整流程**: 從 DICOM 資料到最終分析報告的完整處理流程
- **批次處理**: 支援單一或多個病患資料的批次處理
- **平行運算**: 多核心平行處理大幅提升處理效率
- **臨床應用**: 腦室體積測量和左右對稱性分析
- **高精度**: 使用 U-Net 架構，結合多種評估指標

## 支援的腦部結構

1. **Basal-cistern** - 基底池
2. **CSF** - 腦脊髓液
3. **Falx** - 大腦鐮
4. **Fourth-ventricle** - 第四腦室
5. **Tentorium** - 小腦幕
6. **Third-ventricle** - 第三腦室
7. **Ventricle_L** - 左側腦室
8. **Ventricle_R** - 右側腦室
9. **Ventricles** - 全腦室系統

## 系統需求

- Python 3.8+
- CUDA 支援的 GPU (建議)
- 至少 8GB RAM
- 充足的儲存空間用於醫學影像資料

## 安裝

1. 複製專案
```bash
git clone <repository-url>
cd graduation
```

2. 安裝相依套件
```bash
pip install -r requirements.txt
```

## 專案架構

```
graduation/
├── README.md                 # 專案說明文件
├── requirements.txt          # Python 套件需求
├── config.py                # 全域配置檔案
├── read_dicom.py            # DICOM 檔案處理
├── Data_Preprocessing.py     # 資料預處理模組
├── train.py                 # 2D/2.5D 模型訓練
├── training1.py             # 額外訓練功能
├── befor_processes.py       # 完整推論流程
├── transform.py             # 體積分析模組
├── model/                   # 預訓練模型檔案
└── 3D-UNet-main/           # 3D 深度學習模組
```

## 快速開始

### 1. DICOM 轉換和預處理

```bash
python Data_Preprocessing.py <DICOM_資料夾路徑>
```

### 2. 完整推論流程

#### 單一病患處理（向後相容）
```bash
python befor_processes.py "DICOM資料夾路徑"
```

#### 多個病患批次處理
```bash
# 處理多個指定的資料夾
python befor_processes.py /path/to/patient1 /path/to/patient2 /path/to/patient3

# 自動發現並處理父目錄下的所有 DICOM 資料夾
python befor_processes.py --batch /path/to/patients/

# 從檔案讀取病患路徑清單
python befor_processes.py --input-list patients.txt
```

#### 進階批次處理選項
```bash
# 啟用平行處理（自動使用 CPU 核心數的一半）
python befor_processes.py --parallel /path/to/patient1 /path/to/patient2

# 指定平行處理的工作執行緒數量
python befor_processes.py --parallel --max-workers 4 /path/to/patient1 /path/to/patient2

# 統一輸出到指定目錄
python befor_processes.py --output-dir /results /path/to/patient1 /path/to/patient2

# 跳過已存在的結果檔案
python befor_processes.py --skip-existing /path/to/patient1 /path/to/patient2

# 組合使用多個選項
python befor_processes.py \
    --parallel \
    --max-workers 2 \
    --output-dir /results \
    --skip-existing \
    /path/to/patient1 /path/to/patient2 /path/to/patient3
```

這將會：
- 自動處理 DICOM 檔案
- 載入預訓練模型進行分割
- 生成各腦部結構的分割結果
- 計算第三腦室和第四腦室
- 生成批次處理報告

### 3. 體積分析

```python
from transform import calculate_volume_from_nii, calculate_ventricle_symmetry

# 計算體積
volume = calculate_volume_from_nii('path/to/segmentation.nii.gz')

# 分析左右腦室對稱性
symmetry_result = calculate_ventricle_symmetry('left_ventricle.nii.gz', 'right_ventricle.nii.gz')
```

## 模型訓練

### 2D/2.5D 模型訓練

```python
from train import build_2_5d_unet_model, load_data_2_5d

# 建立模型
model = build_2_5d_unet_model((512, 512, 1))

# 載入資料並訓練
train_data, train_labels = load_data_2_5d('訓練資料路徑')
model.fit(train_data, train_labels, epochs=100, batch_size=8)
```

### 3D 模型訓練

```bash
cd 3D-UNet-main
python train.py --dataset_path "資料路徑" --epochs 100 --batch_size 2
```

## 輸入輸出格式

### 輸入
- **DICOM 檔案**: 原始腦部 CT 影像
- **RT Structure**: 標註檔案 (訓練時需要)

### 輸出
- **NIfTI 檔案**: 各結構的分割遮罩 (.nii.gz)
- **體積報告**: CSV 格式的定量分析
- **對稱性分析**: 左右腦室對稱性評估

## 主要模組說明

### 1. 配置模組 (`config.py`)
包含全域參數設定：
- 資料集路徑和類別定義
- 訓練超參數
- 腦部結構類別對應表

### 2. DICOM 處理 (`read_dicom.py`)
- DICOM 檔案讀取和解析
- RT Structure 轉換
- 影像重採樣和格式轉換

### 3. 資料預處理 (`Data_Preprocessing.py`)
- HU 值閾值處理 (-50 到 100)
- 形態學操作 (侵蝕、膨脹)
- 最大連通區域提取

### 4. 模型訓練 (`train.py`)
- 2D/2.5D U-Net 模型架構
- 資料載入和增強
- 訓練流程和評估指標

### 5. 推論系統 (`befor_processes.py`)
- 完整的自動分割流程
- 多模型載入和預測
- 腦室系統分離計算
- **批次處理支援**: 支援單一或多個 DICOM 資料夾處理
- **平行處理**: 多核心平行處理提升效率
- **智慧輸出管理**: 統一輸出目錄和跳過已存在結果

### 6. 體積分析 (`transform.py`)
- 3D 體積計算
- 左右腦室對稱性分析
- 批次處理功能

## 技術特點

### 深度學習架構
- **U-Net 變體**: 專為醫學影像分割設計
- **多尺度訓練**: 2D 切片和 2.5D 體積訓練
- **3D 完整訓練**: 保持空間一致性

### 醫學影像特化
- **HU 值處理**: 針對腦部結構的專用閾值範圍
- **形態學操作**: 自動去除雜訊和小的不連通區域
- **連通性分析**: 保留最大連通區域以提升分割品質

### 評估指標
- **Dice Coefficient**: 分割重疊度評估
- **IoU (Intersection over Union)**: 交集比聯集
- **F1-Score**: 準確率和召回率的調和平均
- **對稱性指數**: 左右腦室不對稱程度

## 臨床應用

### 體積測量
- 精確測量各腦部結構體積
- 支援批次處理多個病患資料
- 自動生成標準化報告
- 批次處理報告含成功率統計和異常檢測

### 對稱性分析
- 左右腦室體積比較
- 不對稱指數計算：|L - R| / (L + R) × 100%
- 臨床意義判讀：
  - 高度對稱 (< 5%)
  - 中度對稱 (5-10%)
  - 輕度不對稱 (10-20%)
  - 明顯不對稱 (> 20%)

### 疾病監測
- 腦室擴大檢測
- 腦脊髓液異常識別
- 結構性病變評估

## 使用範例

### 1. 單一病患完整處理流程
```bash
# 執行完整分割（包含預處理）
python befor_processes.py "/path/to/dicom/folder"
```

### 2. 多病患批次處理流程
```bash
# 批次處理多個病患（平行處理）
python befor_processes.py \
    --parallel \
    --max-workers 3 \
    --output-dir "/results" \
    /data/patient001 \
    /data/patient002 \
    /data/patient003
```

### 3. 自動發現並批次處理
```bash
# 自動發現父目錄下的所有 DICOM 資料夾並處理
python befor_processes.py \
    --batch "/data/all_patients" \
    --parallel \
    --output-dir "/results" \
    --skip-existing
```

### 4. 從清單檔案批次處理
```bash
# 建立病患清單檔案
echo "/data/patient001" > patients.txt
echo "/data/patient002" >> patients.txt
echo "/data/patient003" >> patients.txt

# 從清單檔案批次處理
python befor_processes.py \
    --input-list patients.txt \
    --parallel \
    --output-dir "/results"
```

### 5. 研究用大規模批次處理
```bash
# 處理大量病患資料（適合研究用途）
python befor_processes.py \
    --batch "/research_data/all_patients" \
    --parallel \
    --max-workers 6 \
    --output-dir "/research_results" \
    --skip-existing \
    --log-level INFO
```

### 6. 進階批次處理功能

#### 命令列選項說明
```bash
# 檢視所有可用選項
python befor_processes.py --help

# 主要選項：
# --batch DIR           自動發現父目錄中的 DICOM 資料夾
# --input-list FILE     從檔案讀取 DICOM 資料夾路徑清單
# --output-dir DIR      統一輸出目錄
# --parallel            啟用平行處理
# --max-workers N       指定工作執行緒數量
# --skip-existing       跳過已存在的結果檔案
# --log-level LEVEL     設定日誌等級 (DEBUG/INFO/WARNING/ERROR)
```

#### 批次處理報告範例
```
腦部分割批次處理報告
生成時間: 2024-06-24 14:30:15

=== 處理摘要 ===
總病患數: 10
成功處理: 9
處理失敗: 1
成功率: 90.0%
總處理時間: 1245.67 秒
平均處理時間: 124.57 秒/病患

=== 詳細結果 ===
✓ patient001: 120.34秒
✓ patient002: 118.92秒
✗ patient003: 0.00秒 (錯誤: 找不到模型檔案)
...
```

### 7. 效能優化建議

#### 平行處理設定
```bash
# 根據系統資源調整工作執行緒數量
# 建議設定：CPU 核心數的 50-75%
python befor_processes.py --parallel --max-workers 4 [paths...]

# 對於大型資料集，建議使用批次模式
python befor_processes.py --batch /large/dataset --parallel --max-workers 6
```

#### 儲存空間管理
```bash
# 使用統一輸出目錄避免分散檔案
python befor_processes.py --output-dir /central/results [paths...]

# 跳過已處理的病患節省時間
python befor_processes.py --skip-existing [paths...]
```
