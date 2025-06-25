import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, concatenate, Conv2DTranspose
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,TensorBoard
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import nibabel as nib
from config import DATASET, EPOCH, BATCH_SIZE, SEED, INPUT,TYPE_MAP,CLASS_TYPE,PREDICT_FILE,MODEL_PATH
import datetime
import random
from tqdm import tqdm
# 設定資料路徑
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 使用 CPU 訓練
IMG_HEIGHT = 512
IMG_WIDTH = 512
IMG_CHANNELS = INPUT  # 2.5D方法使用3個通道（三張連續切片）
EPOCHS = EPOCH
now_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

class BinaryIoU(tf.keras.metrics.Metric):
    def __init__(self, name='binary_iou', **kwargs):
        super().__init__(name=name, **kwargs)
        self.iou = tf.keras.metrics.IoU(num_classes=2, target_class_ids=[1])

    def update_state(self, y_true, y_pred, sample_weight=None):
        # 將機率轉成 class label (0 or 1)
        y_pred = tf.cast(y_pred > 0.5, tf.int32)
        y_true = tf.cast(y_true, tf.int32)
        self.iou.update_state(y_true, y_pred, sample_weight)

    def result(self):
        return self.iou.result()

    def reset_states(self):
        self.iou.reset_states()
def f1_score(y_true, y_pred):
    y_pred = tf.round(y_pred)  # 將機率轉為 0 或 1
    tp = tf.reduce_sum(tf.cast(y_true * y_pred, 'float'), axis=0)
    fp = tf.reduce_sum(tf.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = tf.reduce_sum(tf.cast(y_true * (1 - y_pred), 'float'), axis=0)

    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * precision * recall / (precision + recall + 1e-7)
    return tf.reduce_mean(f1)  # 對多類別情況取平均
# 載入和預處理資料 - 2.5D方式：每個樣本由連續的三張切片組成
def load_data_2_5d(image_files, mask_files):
    X = []
    y = []
    
    # 將所有圖像分組，以便可以獲取連續的切片
    for i in range(1, len(image_files) - 1):  # 跳過第一張和最後一張，確保每個樣本都有前中後三張
        # 載入三張連續切片作為2.5D輸入

        prev_img = image_files[i-1]/ 255.0
        curr_img = image_files[i]/ 255.0
        next_img = image_files[i+1]/ 255.0

        # 組合成三通道輸入
        combined_img = np.stack([prev_img, curr_img, next_img], axis=-1)
        
        # 載入當前切片的標記
        mask = mask_files[i]

        X.append(combined_img)
        y.append(mask)
    
    return np.array(X), np.array(y)
def load_data_2d(original_data, label_data):
    X = []
    y = []
    for i in range(len(original_data)):
    # 載入原始影像
        stack_img = np.stack([original_data[i]], axis=-1)
    # 載入標記
        mask = label_data[i]
        if mask.max() == 0:
            continue
        # 將每個切片的影像和標記分別加入列表
        X.append(stack_img)
        y.append(mask)

    return np.array(X), np.array(y)
# 載入2.5D資料
# 建立適合2.5D輸入的U-Net模型
def build_2_5d_unet_model(input_shape):
    inputs = Input(input_shape, name="input_layer")
    
    # 編碼部分
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
    
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
    
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D((2, 2))(c4)
    
    # 橋接部分
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    
    # 解碼部分
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
    
    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
    
    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
    
    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    return model
def augment_image_mask(image, mask):
    # 水平翻轉
    if random.random() > 0.5:
        image = np.fliplr(image)
        mask = np.fliplr(mask)

    # 垂直翻轉
    if random.random() > 0.5:
        image = np.flipud(image)
        mask = np.flipud(mask)

    # 隨機旋轉 90 度的倍數
    if random.random() > 0.5:
        k = 2
        image = np.rot90(image, k)
        mask = np.rot90(mask, k)
    # print(image.shape, mask.shape)
    return image, mask
# 創建數據增強生成器
def create_data_generator(X, y, batch_size):
    while True:
        idx = np.random.choice(X.shape[0], size=batch_size, replace=False)
        batch_X = []
        batch_y = []

        for i in idx:
            image, mask = X[i], y[i]
            image, mask = augment_image_mask(image, mask)
            batch_X.append(image)
            batch_y.append(mask)

        yield np.array(batch_X), np.array(batch_y)
# 測試模型
def test_2_5d_model(model, X_test, y_test, num_samples=5):
    # 確保有足夠的樣本
    num_samples = min(num_samples, len(X_test))
    
    # 隨機選擇樣本
    idx = np.random.choice(len(X_test), size=num_samples, replace=False)
    
    plt.figure(figsize=(15, 5 * num_samples))
    for i, idx_sample in enumerate(idx):
        x_sample = X_test[idx_sample]
        y_true = y_test[idx_sample].squeeze()
        
        # 預測
        y_pred = model.predict(x_sample[np.newaxis, ...]).squeeze()
        y_pred_binary = (y_pred > 0.5).astype(np.float32)
        
        # 繪製原始圖像 (取中間切片)
        plt.subplot(num_samples, 3, i*3 + 1)
        if INPUT == 1:
            plt.imshow(x_sample[:, :, 0], cmap='gray')  # 顯示中間切片
        else:
            plt.imshow(x_sample[:, :, 1], cmap='gray')  # 顯示中間切片
        plt.title('original')
        plt.axis('off')
        
        # 繪製真實標記
        plt.subplot(num_samples, 3, i*3 + 2)
        plt.imshow(y_true, cmap='gray')
        plt.title('read')
        plt.axis('off')
        
        # 繪製預測標記
        plt.subplot(num_samples, 3, i*3 + 3)
        plt.imshow(y_pred_binary, cmap='gray')
        plt.title('predict')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'log_2d/{DATASET.split("/")[-1]}_{now_time}_predict.png')
    plt.close()
def evaluate_model(model, X_test, y_test):
    # 預測結果
    y_pred = model.predict(X_test)
    y_pred_binary = (y_pred > 0.5).astype(np.float32)
    
    
    # 輸出一般評估指標
    evaluation = model.evaluate(X_test, y_test)
    dice = 2 * evaluation[3] * evaluation[2] / (evaluation[3] + evaluation[2] + 1e-7)
    # print(f"平均Dice係數: {np.mean(dice_scores):.4f}")
    print(f"平均Dice: {dice:.4f}")
    # print(f"標準差: {np.std(dice_scores):.4f}")
    print(f"Loss: {evaluation[0]:.4f}")
    print(f"Accuracy: {evaluation[1]:.4f}")
    print(f"Recall: {evaluation[2]:.4f}")
    print(f"Precision: {evaluation[3]:.4f}")
# 預測單一樣本的函數 - 需要連續三張切片作為輸入
def predict_2_5d_single_patient(model, image_paths,train=True,name=None):

    # image_list = [f for f in nib.load(image_paths).get_fdata()]
    image_data = nib.load(image_paths)
    affine = image_data.affine
    header = image_data.header
    image_list = image_data.get_fdata()
    out_img = []

    # 預測
    for i in tqdm(range(0, image_list.shape[2])):
        image = image_list[:, :, i]
        if image.shape[0] != 512 or image.shape[1] != 512:
            img = np.zeros((512, 512))
            img[:image.shape[0], :image.shape[1]] = image
            image = img
        # image = image / 255.0
        img_input = image.reshape(1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
        prediction = model.predict(img_input,verbose=0).squeeze()
        prediction_binary = (prediction > 0.5).astype(np.uint16)
        out_img.append(prediction_binary)
    if train:
        nib.save(nib.Nifti1Image(np.array(out_img).transpose(1, 2, 0),affine=affine,header=header), f'log_2d/{DATASET.split("/")[-1]}_{now_time}_predict.nii.gz')
    else:
        # 修復路徑問題：使用輸入檔案的目錄而不是根目錄
        output_dir = os.path.dirname(image_paths)
        output_path = os.path.join(output_dir, f'{name}.nii.gz')
        nib.save(nib.Nifti1Image(np.array(out_img).transpose(1, 2, 0),affine=affine,header=header), output_path)
def train():
     # 載入資料
    
    dir_list = [os.path.join(DATASET,x) for x in os.listdir(DATASET) if "not_ok" not in x]

    original_data_files = sorted([os.path.join(x, y) for x in dir_list for y in os.listdir(x) if y.startswith("original_")])
    label_data_files = sorted([os.path.join(x, y) for x in dir_list for y in os.listdir(x) if y.startswith("mask_"+TYPE_MAP[CLASS_TYPE])])
    
    original_data = [nib.load(f) for f in original_data_files]
    label_data = [nib.load(f) for f in label_data_files]
    
    label =[]
    original = []
    for i in range(len(label_data)):
        for j in range(label_data[i].shape[2]):
            if label_data[i].get_fdata()[:,:,j].max() == 0:
                continue
            if original_data[i].shape[0]!=512 or original_data[i].shape[1]!=512:
                img = np.zeros((512, 512))
                mask = np.zeros((512, 512))
                img[:original_data[i].shape[0],:original_data[i].shape[1]] = original_data[i].get_fdata()[:,:,j]
                mask[:label_data[i].shape[0],:label_data[i].shape[1]] = label_data[i].get_fdata()[:,:,j]
            else:
                img = original_data[i].get_fdata()[:,:,j]
                mask = label_data[i].get_fdata()[:,:,j]
            mask[mask > 0] = 1
            label.append(mask)
            original.append(img)

    if INPUT == 1:
        X, y = load_data_2d(original, label)
    else:
        X, y = load_data_2_5d(original, label)
    # 將資料集形狀調整為適合CNN的格式
    X = X.reshape(-1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
    y = y.reshape(-1, IMG_HEIGHT, IMG_WIDTH, 1)

    if len(X) == 0 or len(y) == 0:
        print("❗ 資料集為空，請檢查資料讀取流程！")
        exit()
    else:
        print(f"2.5D資料集形狀: X={X.shape}, y={y.shape}")
        print(f"2.5D輸入通道維度: {X.shape[-1]}")

    # 分割資料集為訓練集和驗證集
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED)

    print("訓練集形狀:", X_train.shape, y_train.shape)
    print("驗證集形狀:", X_val.shape, y_val.shape)

    # 構建2.5D模型
    model = build_2_5d_unet_model((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    # model = UNet2_5D(input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), num_classes=1)
    model.compile(optimizer=Adam(learning_rate=1e-4),
                   loss='binary_crossentropy',
                     metrics=['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.Precision(), BinaryIoU()])
    model.summary()
    log_dir = "D:/logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir)
    # 設置回調函數
    callbacks = [
        ModelCheckpoint((f'log_2d/{DATASET.split("/")[-1]}_{now_time}_model.keras'), verbose=1, save_best_only=True),
        EarlyStopping(patience=20, verbose=1),
        ReduceLROnPlateau(factor=0.5, patience=10, verbose=1)
    ]

    # 訓練模型
    train_gen = create_data_generator(X_train, y_train, BATCH_SIZE)
    val_gen = create_data_generator(X_val, y_val, BATCH_SIZE)

    steps_per_epoch = len(X_train) // BATCH_SIZE
    validation_steps = len(X_val) // BATCH_SIZE

    # 如果批次大小無法整除資料集大小，確保至少有一步
    steps_per_epoch = max(1, steps_per_epoch)
    validation_steps = max(1, validation_steps)

    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        validation_data=val_gen,
        validation_steps=validation_steps,
        callbacks=[callbacks, tensorboard_callback]
    )

    # 繪製訓練過程
    f1_train = [2 * p * r / (p + r + 1e-7) for p, r in zip(history.history['precision'], history.history['recall'])]
    f1_val = [2 * p * r / (p + r + 1e-7) for p, r in zip(history.history['val_precision'], history.history['val_recall'])]


    # 測試模型表現
    test_2_5d_model(model, X_val, y_val, num_samples=5)

    # 保存模型
    model.save(f'log_2d/{DATASET.split("/")[-1]}_{now_time}_model.keras')
    print("2.5D模型訓練完成並保存!")

    # 評估模型
    evaluate_model(model, X_val, y_val)
    return model

if __name__ == "__main__":
    model = train()
    # model = build_2_5d_unet_model((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    # model.load_weights(MODEL_PATH)
    
    # 可以在測試階段使用這個函數進行預測
    predict_2_5d_single_patient(model, PREDICT_FILE)