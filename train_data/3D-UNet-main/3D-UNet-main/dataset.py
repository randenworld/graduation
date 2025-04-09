import nibabel as nib
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.utils import shuffle
from config import (
    DATASET_PATH, TRAIN_VAL_TEST_SPLIT,
    TRAIN_BATCH_SIZE, VAL_BATCH_SIZE, TEST_BATCH_SIZE,NUMBER_WORKERS
)

class MedicalDataset(Dataset):
    """
    Dataset class for medical imaging data
    """
    def __init__(self, image_dir, transforms=None, split_ratios=TRAIN_VAL_TEST_SPLIT, mode=None):
        super(MedicalDataset, self).__init__()
        
        self.image_dir = image_dir
        self.transforms = transforms
        self.mode = mode
        
        # === 修改1: 初始化空列表以避免屬性錯誤 ===
        self.samples = []
        self.train_samples = []
        self.val_samples = []
        self.test_samples = []
        
        # 獲取所有標籤文件
        self.label_files = [f for f in os.listdir(image_dir) if f.startswith("merged_labels_") and f.endswith(".nii.gz")]
        
        # 印出找到的標籤文件
        # print(f"找到 {len(self.label_files)} 個標籤文件:")
        # for lf in self.label_files:
        #     print(f"  - {lf}")
        
        # 確保匹配的影像文件
        self.image_files = []
        self.valid_pairs = []

        # === 修改2: 改進文件名匹配邏輯 ===
        for label_file in self.label_files:
            # 從標籤文件名獲取對應的影像文件名
            label_id = label_file.replace("merged_labels_", "").replace(".nii.gz", "")
            
            # 嘗試不同可能的影像文件命名模式
            possible_patterns = [
                f"original_{label_id}.nii.gz",  # 原始預期的命名
                f"orignal_{label_id}.nii.gz",   # 常見拼寫錯誤 (缺少 'i')
                f"image_{label_id}.nii.gz",     # 另一種可能的命名
                f"{label_id}.nii.gz"            # 簡單命名
            ]
            
            found_match = False
            for pattern in possible_patterns:
                if os.path.exists(os.path.join(image_dir, pattern)):
                    self.image_files.append(pattern)
                    self.valid_pairs.append((pattern, label_file))
                    # print(f"匹配: {label_file} -> {pattern}")
                    found_match = True
                    break
            
            if not found_match:
                print(f"警告: 找不到 {label_file} 的對應影像文件。")
                # === 修改3: 列出目錄中的所有文件，幫助調試 ===
                # print(f"目錄中的文件: {os.listdir(image_dir)[:10]}...")

        # 創建樣本列表
        self.samples = [{'image': os.path.join(image_dir, img), 'label': os.path.join(image_dir, lbl)} for img, lbl in self.valid_pairs]

        print(f"找到 {len(self.samples)} 個有效樣本對")

        if not self.samples:
            print("警告: 沒有找到有效的影像-標籤對！請檢查數據集路徑和文件命名。")
            return  # === 修改4: 即使沒有樣本也繼續初始化其他屬性 ===
        
        # 隨機打亂並分割數據集
        shuffled_samples = shuffle(self.samples, random_state=42)
        num_samples = len(shuffled_samples)

        # 計算訓練、驗證、測試集的分割索引
        train_end = int(num_samples * split_ratios[0])
        val_end = train_end + int(num_samples * split_ratios[1])

        self.train_samples = shuffled_samples[:train_end]
        self.val_samples = shuffled_samples[train_end:val_end]
        self.test_samples = shuffled_samples[val_end:]

        print(f"數據集分割: {len(self.train_samples)} 訓練, {len(self.val_samples)} 驗證, {len(self.test_samples)} 測試")
    
    def set_mode(self, mode):
        """設置數據集模式（train, val, test）"""
        self.mode = mode
    
    def __len__(self):
        """根據模式返回數據集大小"""
        # === 修改5: 增加安全檢查以避免屬性錯誤 ===
        if self.mode == "train" and hasattr(self, 'train_samples'):
            return len(self.train_samples)
        elif self.mode == "val" and hasattr(self, 'val_samples'):
            return len(self.val_samples)
        elif self.mode == "test" and hasattr(self, 'test_samples'):
            return len(self.test_samples)
        else:
            return len(self.samples)
    
    def __getitem__(self, idx):
        """獲取數據集樣本"""
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # === 修改6: 增加安全檢查以避免屬性錯誤 ===
        try:
            # 根據模式選擇樣本
            if self.mode == "train" and hasattr(self, 'train_samples') and len(self.train_samples) > 0:
                sample = self.train_samples[idx]
            elif self.mode == "val" and hasattr(self, 'val_samples') and len(self.val_samples) > 0:
                sample = self.val_samples[idx]
            elif self.mode == "test" and hasattr(self, 'test_samples') and len(self.test_samples) > 0:
                sample = self.test_samples[idx]
            else:
                sample = self.samples[idx]
        except IndexError:
            print(f"索引錯誤: 索引 {idx} 超出範圍")
            if len(self.samples) > 0:
                sample = self.samples[0]  # 使用第一個樣本作為後備
            else:
                raise RuntimeError("數據集為空，無法獲取樣本")
        
        img_path = sample['image']
        label_path = sample['label']
        
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"影像文件未找到: {img_path}")
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"標籤文件未找到: {label_path}")

        # 載入影像與標籤
        try:
            img_object = nib.load(img_path)
            img_array = img_object.get_fdata()

            label_object = nib.load(label_path)
            label_array = label_object.get_fdata()
            # print(f"Original image shape: {img_array.shape}")
            # print(f"Original label shape: {label_array.shape}")
            # 若為 4D 數據，則將通道維度移至第一維
            if len(img_array.shape) == 4:
                img_array = np.moveaxis(img_array, -1, 0)
            else:
                img_array = img_array[np.newaxis, ...]  # 3D 數據加一個通道維度
            
            # 構建輸出
            output = {
                'name': os.path.basename(img_path),
                'image': img_array,
                'label': label_array
            }
            
            # if self.transforms:
            #     output = self.transforms(output)

            return output
            
        except Exception as e:
            print(f"載入文件時出錯: {e}")
            print(f"影像: {img_path}, 標籤: {label_path}")
            raise e

# === 修改7: 改進數據加載器創建函數以處理空數據集 ===
def get_train_val_test_Dataloaders(train_transforms=None, val_transforms=None, test_transforms=None):
    """
    創建訓練、驗證和測試數據加載器
    """
    # 創建主數據集以檢查是否有有效樣本
    # dataset = MedicalDataset(image_dir=DATASET_PATH, transforms=None)
    
    # # 檢查是否有有效樣本
    # if len(dataset.samples) == 0:
    #     print("錯誤: 數據集中沒有找到有效樣本。請檢查文件路徑和命名。")
    #     print(f"當前數據集路徑: {DATASET_PATH}")
    #     print(f"目錄中的文件: {os.listdir(DATASET_PATH)[:10]}...")
    #     return None, None, None
    
    train_dataset = MedicalDataset(
        image_dir=DATASET_PATH,
        transforms=train_transforms,
        mode="train"
    )
    
    val_dataset = MedicalDataset(
        image_dir=DATASET_PATH,
        transforms=val_transforms,
        mode="val"
    )
    
    test_dataset = MedicalDataset(
        image_dir=DATASET_PATH,
        transforms=test_transforms,
        mode="test"
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=NUMBER_WORKERS,
        pin_memory=True,
        persistent_workers=True#讓 worker 保持存活，減少 epoch 間的重啟時間
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=NUMBER_WORKERS,
        pin_memory=True,
        persistent_workers=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=NUMBER_WORKERS,
        pin_memory=True,
        persistent_workers=True
    )
    
    return train_loader, val_loader, test_loader