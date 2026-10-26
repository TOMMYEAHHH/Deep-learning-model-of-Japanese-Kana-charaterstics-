import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ReduceLROnPlateau, Callback
import matplotlib.pyplot as plt
import time
import math
import matplotlib
import os

matplotlib.rcParams['font.family'] = 'MS Gothic'

# ====== 跨平台硬件检测 ======
def detect_accelerator():
    physical_devices = tf.config.list_physical_devices()
    logical_devices = tf.config.list_logical_devices()
    gpu_devices = [d for d in physical_devices if d.device_type == 'GPU']
    print("可用逻辑设备：", logical_devices)
    print("可用物理设备：", physical_devices)
    if gpu_devices:
        print("检测到 GPU 可用 (Apple/NVIDIA)")
        return "GPU"
    else:
        print("未检测到 GPU/MPS，将使用 CPU")
        return "CPU"

accelerator = detect_accelerator()

# ====== 加载训练数据 ======
BATCH_SIZE = 32
IMG_SIZE = (28, 28)

train_dir = "data/all_kana/train"
test_dir  = "data/all_kana/test"

# ====== 加载 Dataset ======
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir, image_size=IMG_SIZE, color_mode="grayscale", batch_size=BATCH_SIZE
)
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir, image_size=IMG_SIZE, color_mode="grayscale", batch_size=BATCH_SIZE
)

# ====== class_names ======
class_names = train_ds.class_names
NUM_CLASSES = len(class_names)
print("类别总数:", NUM_CLASSES)

# ====== 预处理函数 ======
def preprocess(image, label):
    image = tf.cast(image, tf.float32)/255.0
    image = tf.image.grayscale_to_rgb(image)
    # 数据增强
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, 0.9, 1.1)
    image = tf.image.resize_with_crop_or_pad(image, 29, 29)
    image = tf.image.random_crop(image, [28,28,3])
    return image, label

ds_train = train_ds.unbatch().map(preprocess).shuffle(10000).batch(BATCH_SIZE).repeat().prefetch(tf.data.AUTOTUNE)
ds_test  = test_ds.unbatch().map(preprocess).batch(BATCH_SIZE).repeat().prefetch(tf.data.AUTOTUNE)

# ====== 可视化 Callback ======
vis_test_ds = test_ds.unbatch().map(preprocess).batch(32).take(1)

class TestVisualizationCallback(Callback):
    def __init__(self, test_dataset, class_names):
        super().__init__()
        self.test_dataset = test_dataset
        self.class_names = class_names

    def on_epoch_end(self, epoch, logs=None):
        import numpy as np
        images, labels = next(iter(self.test_dataset))
        preds = self.model.predict(images, verbose=0)
        pred_labels = np.argmax(preds, axis=1)
        pred_probs = np.max(preds, axis=1)

        train_loss = logs.get('loss', 0)
        val_loss = logs.get('val_loss', 0)
        train_acc = logs.get('accuracy', 0)
        val_acc = logs.get('val_accuracy', 0)

        plt.figure(figsize=(12,6))
        plt.suptitle(
            f"Epoch {epoch+1} - Train Loss: {train_loss:.3f}, Val Loss: {val_loss:.3f}, "
            f"Train Acc: {train_acc:.3f}, Val Acc: {val_acc:.3f}", fontsize=14
        )

        for i in range(min(16, images.shape[0])):
            plt.subplot(4,4,i+1)
            img = images[i].numpy()
            plt.imshow(img[:,:,0], cmap='gray')

            true_index = int(labels[i].numpy())
            pred_index = int(pred_labels[i])

            true_label = self.class_names[true_index] if true_index < len(self.class_names) else f"UNK({true_index})"
            pred_label = self.class_names[pred_index] if pred_index < len(self.class_names) else f"UNK({pred_index})"

            plt.title(f"T:{true_label}\nP:{pred_label} ({pred_probs[i]:.2f})")
            plt.axis('off')

        plt.tight_layout(rect=[0,0,1,0.95])
        plt.show()

# ====== 构建 CNN 模型 ======
model = models.Sequential([
    layers.Input(shape=(28,28,3)),
    layers.Conv2D(32,(3,3),activation='relu',padding='same'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,(3,3),activation='relu',padding='same'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ====== 动态统计训练/验证样本 ======
def count_images(root_dirs):
    total = 0
    for root_dir in root_dirs:
        for subdir, _, files in os.walk(root_dir):
            total += len([f for f in files if f.lower().endswith((".png",".jpg",".jpeg"))])
    return total

num_train_samples = count_images([train_dir])
num_val_samples = count_images([test_dir])
steps_per_epoch = math.ceil(num_train_samples / BATCH_SIZE)
validation_steps = math.ceil(num_val_samples / BATCH_SIZE)

# ====== 回调 ======
lr_callback = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)
vis_callback = TestVisualizationCallback(vis_test_ds, class_names)

# ====== 开始训练 ======
print("开始训练 CNN 模型...")
start = time.time()
history = model.fit(
    ds_train,
    epochs=25,
    validation_data=ds_test,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    callbacks=[lr_callback, vis_callback]
)
end = time.time()
print(f"\n训练完成！耗时：{end - start:.2f} 秒")

# ====== 可视化训练曲线 ======
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history["loss"], label="train loss")
plt.plot(history.history["val_loss"], label="val loss")
plt.title("Loss")
plt.legend()
plt.subplot(1,2,2)
plt.plot(history.history["accuracy"], label="train acc")
plt.plot(history.history["val_accuracy"], label="val acc")
plt.title("Accuracy")
plt.legend()
plt.show()
