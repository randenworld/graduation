import math
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from tqdm import tqdm
import torch
from config import (
    TRAINING_EPOCH, NUM_CLASSES, IN_CHANNELS, BCE_WEIGHTS, 
    BACKGROUND_AS_CLASS, TRAIN_CUDA, DATASET_PATH,PATIENCE,
    LR,FACTOR,MODEL_PATH,ALPHA,CLR
)
from dataset import get_train_val_test_Dataloaders
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from unet3d import UNet3D,CombinedLoss
from transforms import (
    train_transform, train_transform_cuda,
    val_transform, val_transform_cuda
)
import torch.nn.functional as F
import time
import datetime
from torch.amp import GradScaler, autocast
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassF1Score, BinaryPrecision, BinaryRecall, BinaryF1Score
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.optim.lr_scheduler import ReduceLROnPlateau
import shutil

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
torch.cuda.empty_cache()
learn_rate_counter = 0
# import torch.backends.cudnn as cudnn
# cudnn.benchmark = True
if __name__ == "__main__":
    MODEL_PATH = MODEL_PATH  # 訓練好的模型權重
    # === 修改1: 檢查數據集路徑是否存在 ===
    if not os.path.exists(DATASET_PATH):
        print(f"錯誤: 數據集路徑不存在: {DATASET_PATH}")
        print("請檢查配置文件中的DATASET_PATH設置")
        exit(1)
    now_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # 創建checkpoints目錄
    dir_checkpoints = f'checkpoints/checkpoint_{now_time}'
    
    os.makedirs(dir_checkpoints, exist_ok=True)
    shutil.copy('train_data/3D-UNet-main/3D-UNet-main/config.py', dir_checkpoints)
    # 指定固定的日誌目錄
    log_dir = "logs/"+now_time
    # 創建日誌目錄
    os.makedirs(log_dir, exist_ok=True)

    latest_step = 0
    # 讀取原來的 Log
    event_files = [f for f in os.listdir(log_dir) if "events.out.tfevents" in f]
    if not event_files:
        print("❌ 沒有找到 TensorBoard 記錄！")
    else:
        event_file_path = os.path.join(log_dir, event_files[-1])  # 讀取最新的事件檔
        event_acc = EventAccumulator(event_file_path)
        event_acc.Reload()
        print(f"✅ 讀取 TensorBoard 記錄：{event_file_path}")
        # 取得 scalar 數據
        # scalar_tags = event_acc.Tags()["scalars"]
        events = event_acc.Scalars("Loss/Train")
        latest_step = events[-1].step+1
        print(f"✅ 最新的 step: {latest_step}")
    # 根據配置調整類別數
    if BACKGROUND_AS_CLASS:
        NUM_CLASSES += 1

    # 初始化TensorBoard
    writer = SummaryWriter(log_dir)
    
    # 初始化模型
    model = UNet3D(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES,use_dropout=True)
    if MODEL_PATH != "":
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cuda"))
    # 設置轉換函數
    train_transforms = train_transform 
    val_transforms = val_transform

    # 設定CUDA
    if torch.cuda.is_available() and TRAIN_CUDA:
        model = model.cuda()
        train_transforms = train_transform_cuda
        val_transforms = val_transform_cuda
    elif not torch.cuda.is_available() and TRAIN_CUDA:
        print('CUDA不可用！訓練將在CPU上進行...')

    # 取得數據加載器
    # === 修改2: 處理可能返回None的情況 ===
    dataloaders = get_train_val_test_Dataloaders(
        train_transforms=train_transforms, 
        val_transforms=val_transforms, 
        test_transforms=val_transforms
    )

    if dataloaders is None or dataloaders[0] is None:
        print("無法創建數據加載器，訓練中止")
        exit(1)

    train_dataloader, val_dataloader, test_dataloader = dataloaders

    # === 修改3: 檢查數據加載器大小 ===
    if len(train_dataloader) == 0 or len(val_dataloader) == 0:
        print("訓練或驗證數據加載器為空，訓練中止")
        exit(1)
    else:
        print(f"訓練數據加載器: {len(train_dataloader)} 批次")
        print(f"驗證數據加載器: {len(val_dataloader)} 批次")

    if torch.cuda.is_available() and TRAIN_CUDA:
        criterion = CombinedLoss(alpha=ALPHA).cuda()
    else:
        criterion = CombinedLoss(alpha=ALPHA)

    optimizer = Adam(params=model.parameters(), lr=LR,weight_decay = 1e-5, amsgrad=True)
    # 定義學習率調度器
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=FACTOR, patience=PATIENCE)
    # 初始化最佳驗證損失
    min_valid_loss = math.inf
    if MODEL_PATH != "":
        min_valid_loss = float(MODEL_PATH.split("_")[-1].replace(".pth", ""))
    scaler = GradScaler()

    # 訓練循環
    print(f"開始訓練，共 {TRAINING_EPOCH} 個 Epoch")
    for epoch in range(TRAINING_EPOCH):
        start_time = time.time()
        
        # 訓練階段
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        print(f"\nEpoch {epoch+1}/{TRAINING_EPOCH} - 訓練階段")
        for data in tqdm(train_dataloader):
            
            image, ground_truth = data['image'], data['label']
            image=image.float()
                     
            # 移至CUDA（如果可用）
            if torch.cuda.is_available() and TRAIN_CUDA:
                # with 
                image = image.cuda(non_blocking=True)
                ground_truth = ground_truth.cuda(non_blocking=True)
            
            # 前向傳播與反向傳播
            optimizer.zero_grad()
            try:
                with autocast(device_type= 'cuda'):  # 使用 FP16 訓練，減少記憶體佔用
                    # output = model(data)
                    # loss = criterion(output, target)
                    target = model(image)
                # === 修改6: 確保ground_truth是整數類型 ===
                    ground_truth = ground_truth.long()
                    loss = criterion(target, ground_truth)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update() 
                # loss.backward()
                # optimizer.step()
                
                
                total_loss += loss.detach()
                del loss  # 釋放 loss 變數
                # torch.cuda.empty_cache()
                num_batches += 1
            except Exception as e:
                print(f"訓練錯誤: {e}")
                print(f"Image shape: {image.shape}, Label shape: {ground_truth.shape}")
                continue
        
        # 計算平均訓練損失
        train_loss = (total_loss / num_batches)
        scheduler.step(train_loss)
        # 驗證階段
        model.eval()
        total_loss = 0.0
        num_batches = 0
        if torch.cuda.is_available() and TRAIN_CUDA:
            if len(BCE_WEIGHTS)>2:
                # 使用多類別精度計算
                precision_metric = MulticlassPrecision(num_classes=NUM_CLASSES, average="macro").cuda()
                recall_metric = MulticlassRecall(num_classes=NUM_CLASSES, average="macro").cuda()
                f1_metric = MulticlassF1Score(num_classes=NUM_CLASSES, average="macro").cuda()
            else:
                precision_metric = BinaryPrecision().cuda()
                recall_metric = BinaryRecall().cuda()
                f1_metric = BinaryF1Score().cuda()
            
        else:
            if len(BCE_WEIGHTS)>2:
                # 使用多類別精度計算
                precision_metric = MulticlassPrecision(num_classes=NUM_CLASSES, average="macro")
                recall_metric = MulticlassRecall(num_classes=NUM_CLASSES, average="macro")
                f1_metric = MulticlassF1Score(num_classes=NUM_CLASSES, average="macro")
            else:
                precision_metric = BinaryPrecision()
                recall_metric = BinaryRecall()
                f1_metric = BinaryF1Score()
    
        print(f"Epoch {epoch+1}/{TRAINING_EPOCH} - 驗證階段")
        with torch.no_grad():
            for data in tqdm(val_dataloader):
                image, ground_truth = data['image'], data['label']
                image=image.float()
                
                # 移至CUDA（如果可用）
                if torch.cuda.is_available() and TRAIN_CUDA:
                    image = image.cuda(non_blocking=True)
                    ground_truth = ground_truth.cuda(non_blocking=True)
                
                # 前向傳播
                try:
                    
                    target = model(image)
                    # === 修改8: 確保ground_truth是整數類型 ===
                    ground_truth = ground_truth.long()
                    loss = criterion(target, ground_truth)
                    
                    total_loss += loss.detach()
                    num_batches += 1
                    # 計算預測結果
                    predicted = torch.argmax(target, dim=1)  # (B, H, W, D) 取最大類別索引

                    # 記錄 Precision、Recall、F1-Score
                    precision_metric.update(predicted, ground_truth)
                    recall_metric.update(predicted, ground_truth)
                    f1_metric.update(predicted, ground_truth)
                except Exception as e:
                    print(f"驗證錯誤: {e}")
                    continue
        
        # 計算平均驗證損失
        valid_loss = (total_loss / num_batches)
        # 計算最終結果
        precision = precision_metric.compute()
        recall = recall_metric.compute()
        f1_score = f1_metric.compute()

        print(f"train Loss: {train_loss:.4f}")
        print(f"Validation Loss: {total_loss / num_batches:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1_score:.4f}")
        print(f"Learning Rate: {scheduler._last_lr[0]:.6f}")

        # 清空緩存
        precision_metric.reset()
        recall_metric.reset()
        f1_metric.reset()
        # 記錄到TensorBoard
        writer.add_scalar("Loss/Train", train_loss, epoch+latest_step)
        writer.add_scalar("Loss/Validation", valid_loss, epoch+latest_step)
        writer.add_scalar("Metrics/Precision", precision, epoch+latest_step)   
        writer.add_scalar("Metrics/Recall", recall, epoch+latest_step)
        writer.add_scalar("Metrics/F1-Score", f1_score, epoch+latest_step)
        writer.add_scalar("Learning Rate", scheduler._last_lr[0], epoch+latest_step)
        # 輸出訓練與驗證損失
        epoch_time = time.time() - start_time
        print(f'Epoch {epoch+1}/{TRAINING_EPOCH} 完成 (用時 {epoch_time:.2f}s)')
        print(f'訓練損失: {train_loss:.6f} \t 驗證損失: {valid_loss:.6f}')
        
        # 保存最佳模型
        # if min_valid_loss > valid_loss or epoch %10 == 0 or epoch == TRAINING_EPOCH-1:
        print(f'驗證損失減少 ({min_valid_loss:.6f} -> {valid_loss:.6f}) \t 正在保存模型...')
        if(min_valid_loss>valid_loss):
            min_valid_loss = valid_loss
        #     learn_rate_counter = 0
        # else:
        #     learn_rate_counter  += 1
        #     if learn_rate_counter > CLR or scheduler._last_lr[0] < 1e-6:
        #         optimizer = Adam(params=model.parameters(), lr=LR,weight_decay = 1e-5, amsgrad=True)
        #         # 定義學習率調度器
        #         scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=FACTOR, patience=PATIENCE)
        #         # 初始化最佳驗證損失
        #         learn_rate_counter = 0
        # 保存模型
        date = datetime.datetime.now().strftime("%m%d")
        torch.save(model.state_dict(), f'{dir_checkpoints}/'+date+f'_epoch{epoch+1+latest_step}valLoss_{min_valid_loss:.6f}.pth')
        
    # 關閉TensorBoard
    writer.flush()
    writer.close()

    print("訓練完成！")