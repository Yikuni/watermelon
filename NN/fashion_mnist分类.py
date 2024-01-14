import tensorflow as tf
from keras import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Dropout, Flatten
from keras.optimizers import Adam
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

model = Sequential([
    Flatten(),
    tf.keras.layers.Dense(
        256,  # 隐藏层中有3个神经元
        activation='relu',  # 激活函数使用softmax
        kernel_regularizer=tf.keras.regularizers.l2(0.001),  # 使用l2正则化
    ),
    Dropout(0.2),
    tf.keras.layers.Dense(
        64,  # 隐藏层中有3个神经元
        activation='relu',  # 激活函数使用softmax
        kernel_regularizer=tf.keras.regularizers.l2(0.001),  # 使用l2正则化
    ),
    Dropout(0.2),
    tf.keras.layers.Dense(
        10,  # 隐藏层中有3个神经元
        activation='softmax',  # 激活函数使用softmax
        kernel_regularizer=tf.keras.regularizers.l2(0.001),  # 使用l2正则化
    ),
])

model.compile(
    optimizer=Adam(learning_rate=0.04, beta_1=0.9, beta_2=0.99, decay=0.02),  # 设置优化器, 学习率为0.1
    # optimizer=MomentumOptimizer(learning_rate=0.01, momentum=0.9),  # Momentum优化器
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),  # 损失函数设置为交叉熵, 原始数据设置为false, 因为激活函数使用了softmax
    metrics=['sparse_categorical_accuracy']  # 评测指标
)
model_path = "./model/fmnist_0_0"
cp_callback = ModelCheckpoint(
    filepath=model_path,
    save_weights_only=True,
    save_best_only=True,
)
history = model.fit(
    train_images,  # 训练集的输入特征
    train_labels,  # 训练集的标签
    batch_size=512,  # 一次循环32个数据
    epochs=100,  # 指定循环500次
    validation_split=0.2,  # 划分训练集和测试集, 0.2比例
    validation_freq=20,  # 每迭代20次验证一次准确率
    callbacks=[cp_callback],
    shuffle=True,
)

model.summary()
result = model.evaluate(test_images, test_labels, batch_size=20)  # 用测试集进行评价
print("test loss, test acc:", result)

acc_history = history.history['sparse_categorical_accuracy']
loss_history = history.history['loss']
plt.subplot(2, 1, 1)
plt.title('loss')
plt.plot(np.arange(len(loss_history)), loss_history)

plt.subplot(2, 1, 2)
plt.title('acc')
plt.plot(np.arange(len(acc_history)), acc_history)

plt.show()