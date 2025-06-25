# 腦部影像自動分割系統

這是一個基於深度學習的腦部 CT 影像自動分割系統，能夠識別和分割多種腦部解剖結構，包括腦脊髓液、腦室系統等重要結構。

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

```bash
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

## 使用範例

### 我在用的方式

```bash
python3 befor_processes.py --batch "input_folder_name" --parallel --output-dir "output_folder_name" --skip-existing --max-workers
```

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