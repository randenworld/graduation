import os
import torch
import torch.nn as nn
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import datetime
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
now_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = "log_cnn/"+now_time    # 創建日誌目錄
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir)
start_time = time.time()
# 自動編碼器模型

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

class Conv3DAutoencoder(nn.Module):
    def __init__(self):
        super(Conv3DAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Dropout3d(p=0.2),
            nn.MaxPool3d(2),  # (304→152, 384→192, 128→64)

            nn.Conv3d(32, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Dropout3d(p=0.2),
            nn.MaxPool3d(2),  # 76x96x32

            nn.Conv3d(64, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Dropout3d(p=0.2),
            nn.MaxPool3d(2),  # 38x48x16
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv3d(128, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Dropout3d(p=0.2),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),  # 76x96x32

            nn.Conv3d(128, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Dropout3d(p=0.2),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),  # 152x192x64

            nn.Conv3d(64, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Dropout3d(p=0.2),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),  # 304x384x128

            nn.Conv3d(32, 1, 3, padding=1)  # 不加 sigmoid，因為是 regression
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# Dice Loss 實作
def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = y_true.view(-1)
    y_pred_f = y_pred.view(-1)
    intersection = (y_true_f * y_pred_f).sum()
    return 1 - (2. * intersection + smooth) / (y_true_f.sum() + y_pred_f.sum() + smooth)


# 自訂 Dataset
class NiiDataset(Dataset):
    def __init__(self, data_dir):
        self.file_paths = [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith(".nii.gz") and f.startswith("original")
        ]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img = nib.load(self.file_paths[idx]).get_fdata()
        img = np.expand_dims(img, axis=0)  # shape → (1, 304, 384, 128)
        img = img.astype(np.float32)
        return torch.from_numpy(img), torch.from_numpy(img)


# 資料讀取
data_dir = "train_data/3D-UNet-main/dataset_ROI_CSF/"
dataset = NiiDataset(data_dir)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# 模型訓練
model = Conv3DAutoencoder().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
early_stopper = EarlyStopping(min_delta=1e-4)
for epoch in range(100):
    model.train()
    epoch_loss = 0
    all_preds, all_labels = [], []
    for batch_idx, (x, y) in tqdm(enumerate(dataloader)):
        x, y = x.to(device), y.to(device)
        output = model(x)
        loss = nn.MSELoss()(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step(loss)
        epoch_loss += loss.item()

        all_preds.append(output.cpu().detach().numpy() > 0.5)
        all_labels.append(y.cpu().numpy() > 0.5)

    # 計算訓練指標
    epoch_loss /= len(dataloader)
    scheduler.step(epoch_loss)
    early_stopper(epoch_loss)

    if early_stopper.early_stop:
        print("Early stopping triggered.")
        break
    train_preds = np.concatenate(all_preds).astype(int).ravel()
    train_labels = np.concatenate(all_labels).astype(int).ravel()
    train_acc = accuracy_score(train_labels, train_preds)
    train_prec = precision_score(train_labels, train_preds, zero_division=0)
    train_rec = recall_score(train_labels, train_preds, zero_division=0)
    writer.add_scalar('train/Loss', epoch_loss, epoch)
    writer.add_scalar('train/Accuracy', train_acc, epoch)
    writer.add_scalar('train/Precision', train_prec, epoch)
    writer.add_scalar('train/Recall', train_rec, epoch)

    print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")
    print(f"Train Accuracy: {train_acc:.4f}\nPrecision: {train_prec:.4f}\nRecall: {train_rec:.4f}")
# 模型儲存
writer.flush()
writer.close()
os.makedirs("CNN", exist_ok=True)
print("Saving model...")
torch.save(model.state_dict(), "CNN/3D_autoencoder.pth")
print("Model saved!")
print(f"Total time: {time.time() - start_time:.2f}s")