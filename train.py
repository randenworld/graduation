import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, concatenate, Conv2DTranspose, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,TensorBoard
from sklearn.model_selection import train_test_split
import nibabel as nib
from config import DATASET, EPOCH, BATCH_SIZE, SEED, INPUT,TYPE_MAP,CLASS_TYPE,PREDICT_FILE,MODEL_PATH,CHANNEL,VERBOSE,DROPOUT_RATE,OUTPUT
import datetime
import random
from tqdm import tqdm
import gc
# 設定資料路徑

IMG_HEIGHT = 512
IMG_WIDTH = 512
now_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
def multiclass_dice_loss(y_true, y_pred, smooth=1e-6):
    """
    y_true: (N, H, W) -> 整數標籤
    y_pred: (N, H, W, C) -> softmax 輸出
    """
    # num_classes = tf.shape(y_pred)[-1]

    # One-hot encoding
    y_true_onehot = tf.one_hot(tf.cast(y_true, tf.int32), OUTPUT)

    dice_list = []
    for c in range(OUTPUT):
        pred_flat = tf.reshape(y_pred[..., c], [-1])
        true_flat = tf.reshape(y_true_onehot[..., c], [-1])

        intersection = tf.reduce_sum(pred_flat * true_flat)
        dice_c = (2. * intersection + smooth) / (
                    tf.reduce_sum(pred_flat) + tf.reduce_sum(true_flat) + smooth)

        dice_list.append(dice_c)

    return 1 - tf.reduce_mean(tf.stack(dice_list))

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

# 建立適合2.5D輸入的U-Net模型
def build_2_5d_unet_model(input_shape,CHANNEL = CHANNEL, dropout_rate=DROPOUT_RATE):
    inputs = Input(input_shape, name="input_layer")
    # 編碼部分
    c1 = Conv2D(CHANNEL, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = BatchNormalization()(c1)
    c1 = Conv2D(CHANNEL, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    c1 = BatchNormalization()(c1)
    c1 = Dropout(dropout_rate)(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(CHANNEL*2, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = BatchNormalization()(c2)
    c2 = Conv2D(CHANNEL*2, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    c2 = BatchNormalization()(c2)
    c2 = Dropout(dropout_rate)(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(CHANNEL*4, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = BatchNormalization()(c3)
    c3 = Conv2D(CHANNEL*4, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    c3 = BatchNormalization()(c3)
    c3 = Dropout(dropout_rate)(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(CHANNEL*8, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = BatchNormalization()(c4)
    c4 = Conv2D(CHANNEL*8, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    c4 = BatchNormalization()(c4)
    c4 = Dropout(dropout_rate)(c4)
    p4 = MaxPooling2D((2, 2))(c4)

    # 橋接部分
    c5 = Conv2D(CHANNEL*16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = BatchNormalization()(c5)
    c5 = Conv2D(CHANNEL*16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    c5 = BatchNormalization()(c5)
    c5 = Dropout(dropout_rate)(c5)

    # 解碼部分
    u6 = Conv2DTranspose(CHANNEL*8, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(CHANNEL*8, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = BatchNormalization()(c6)
    c6 = Conv2D(CHANNEL*8, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
    c6 = BatchNormalization()(c6)
    c6 = Dropout(dropout_rate)(c6)

    u7 = Conv2DTranspose(CHANNEL*4, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(CHANNEL*4, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = BatchNormalization()(c7)
    c7 = Conv2D(CHANNEL*4, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
    c7 = BatchNormalization()(c7)
    c7 = Dropout(dropout_rate)(c7)

    u8 = Conv2DTranspose(CHANNEL*2, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(CHANNEL*2, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = BatchNormalization()(c8)
    c8 = Conv2D(CHANNEL*2, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
    c8 = BatchNormalization()(c8)
    c8 = Dropout(dropout_rate)(c8)

    u9 = Conv2DTranspose(CHANNEL, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv2D(CHANNEL, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = BatchNormalization()(c9)
    c9 = Conv2D(CHANNEL, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
    c9 = BatchNormalization()(c9)
    c9 = Dropout(dropout_rate)(c9)
   
    if OUTPUT == 1:
        outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    else:
        outputs = Conv2D(OUTPUT, (1, 1), activation='softmax')(c9)

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
            if CLASS_TYPE != 6 and CLASS_TYPE != 7  and CLASS_TYPE != 0 and CLASS_TYPE != 9:
                image, mask = augment_image_mask(image, mask)
            batch_X.append(image)
            batch_y.append(mask)

        yield np.array(batch_X), np.array(batch_y)
# 測試模型
def evaluate_model(model, X_test, y_test):
    dice = 0
    # 輸出一般評估指標
    evaluation = model.evaluate(X_test, y_test)
    if OUTPUT==1:
        dice = 2 * evaluation[3] * evaluation[2] / (evaluation[3] + evaluation[2] + 1e-7)
    else:
        pass
    print(f"平均Dice: {dice:.4f}")
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
        img_input = image.reshape(1, IMG_HEIGHT, IMG_WIDTH, INPUT)
        prediction = model.predict(img_input,verbose=0).squeeze()
        if OUTPUT == 1:
            # prediction = (prediction > 0.5).astype(np.uint16)
            prediction_binary = (prediction > 0.5).astype(np.uint16)
            out_img.append(prediction_binary)
        else:
            prediction_class = np.argmax(prediction, axis=-1).astype(np.uint16)  
            out_img.append(prediction_class)
    # print(out_img[0].shape)
    if train:
        nib.save(nib.Nifti1Image(np.array(out_img).transpose(1, 2, 0),affine=affine,header=header), f'log_2d/{DATASET.split("/")[-1]}_{now_time}_predict.nii.gz')
    else:
        nib.save(nib.Nifti1Image(np.array(out_img).transpose(1, 2, 0),affine=affine,header=header), f'{image_paths.split("/")[0]}/{name}.nii.gz')
        
def train():
     # 載入資料
    
    dir_list = [os.path.join(DATASET,x) for x in os.listdir(DATASET) if "not_ok" not in x ]

    original_data_files = sorted([os.path.join(x, y) for x in dir_list for y in os.listdir(x) if y.startswith("original")])
    # label_data_files = sorted([os.path.join(x, y) for x in dir_list for y in os.listdir(x) if y.startswith("mask_"+TYPE_MAP[CLASS_TYPE])])
    label_data_files = sorted([os.path.join(x, y) for x in dir_list for y in os.listdir(x) if TYPE_MAP[CLASS_TYPE] in y])
    if OUTPUT == 1:
        loss_function = 'binary_crossentropy'   
        metrics = [
            'accuracy',
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.Precision(name="precision")
        ]
    else:
        loss_function = 'sparse_categorical_crossentropy'
        metrics = [
            'sparse_categorical_accuracy',
            multiclass_dice_loss,
            # tf.keras.metrics.Recall(name="recall", average="macro"),
            # tf.keras.metrics.Precision(name="precision", average="macro")
        ]



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
            # mask[mask > 0] = 1
            label.append(mask)
            original.append(img)

    X, y = load_data_2d(original, label)
    
    # 將資料集形狀調整為適合CNN的格式
    X = X.reshape(-1, IMG_HEIGHT, IMG_WIDTH, INPUT)
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
    model = build_2_5d_unet_model((IMG_HEIGHT, IMG_WIDTH, INPUT))
    # model = UNet2_5D(input_shape=(IMG_HEIGHT, IMG_WIDTH, INPUT), num_classes=1)
   
    model.compile(optimizer=Adam(learning_rate=1e-4),
                   loss= loss_function,
                    #  metrics=['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.Precision(), BinaryIoU()])
                    metrics = metrics)
    log_dir = "D:/logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir)
    # 設置回調函數
    callbacks = [
        ModelCheckpoint((f'log_2d/{DATASET.split("/")[-1]}_{now_time}_model.keras'), verbose=VERBOSE, save_best_only=True),
        EarlyStopping(patience=20, verbose=VERBOSE),
        ReduceLROnPlateau(factor=0.1, patience=10, verbose=VERBOSE)
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
        epochs=EPOCH,
        validation_data=val_gen,
        validation_steps=validation_steps,
        callbacks=[callbacks, tensorboard_callback]
    )
    # 保存模型
    model.save(f'log_2d/{DATASET.split("/")[-1]}_{now_time}_final_model.keras')
    print("模型訓練完成並保存!")
    tf.keras.backend.clear_session()
    gc.collect()
    model.load_weights(f'log_2d/{DATASET.split("/")[-1]}_{now_time}_model.keras')
    # 評估模型
    evaluate_model(model, X_val, y_val)
    
    return model

if __name__ == "__main__":
    args = sys.argv[1]
    if args == 'train':
        model = train()
    elif args == 'predict':
        model = build_2_5d_unet_model((IMG_HEIGHT, IMG_WIDTH, INPUT))
        model.load_weights(MODEL_PATH)
    
    # 可以在測試階段使用這個函數進行預測
    predict_2_5d_single_patient(model, PREDICT_FILE)