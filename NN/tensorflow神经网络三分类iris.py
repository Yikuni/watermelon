from keras import Sequential
from keras.callbacks import ModelCheckpoint
from keras.optimizers import legacy
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import Dropout, BatchNormalization
from sklearn import datasets
import numpy as np
from sklearn.metrics import f1_score


model_path = "./model/iris_model_12_3_01"
cp_callback = ModelCheckpoint(
    filepath=model_path,
    # save_weights_only=True,
    # save_best_only=True,
    save_freq=50,
)

iris = datasets.load_iris()
x_train = iris.data
y_train = iris.target

np.random.seed(116)
np.random.shuffle(x_train)
np.random.seed(116)
np.random.shuffle(y_train)
tf.random.set_seed(116)

index = 100
x_test = x_train[index:, :]
x_train = x_train[:index, :]
y_test = y_train[index:]
y_train = y_train[:index]

model = Sequential([
    BatchNormalization(),  # 归一化
    tf.keras.layers.Dense(
        12,  # 隐藏层中有3个神经元
        activation='relu',  # 激活函数使用softmax
        kernel_regularizer=tf.keras.regularizers.l2(0.001),  # 使用l2正则化
    ),
    Dropout(0.2),
    tf.keras.layers.Dense(
        3,  # 隐藏层中有3个神经元
        activation='softmax',  # 激活函数使用softmax
        kernel_regularizer=tf.keras.regularizers.l2(0.001),  # 使用l2正则化
    ),
])

model.compile(
    optimizer=legacy.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.99, decay=0.03),  # 设置优化器, 学习率为0.1
    # optimizer=MomentumOptimizer(learning_rate=0.01, momentum=0.9),  # Momentum优化器
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),  # 损失函数设置为交叉熵, 原始数据设置为false, 因为激活函数使用了softmax
    metrics=['sparse_categorical_accuracy']  # 评测指标
)

history = model.fit(
    x_train,  # 训练集的输入特征
    y_train,  # 训练集的标签
    batch_size=32,  # 一次循环32个数据
    epochs=100,  # 指定循环500次
    validation_split=0.2,  # 划分训练集和测试集, 0.2比例
    validation_freq=20,  # 每迭代20次验证一次准确率
    callbacks=[cp_callback],
    shuffle=True,
)

model.summary()
result = model.evaluate(x_test, y_test, batch_size=20)  # 用测试集进行评价
print("test loss, test acc:", result)

acc_history = history.history['sparse_categorical_accuracy']
loss_history = history.history['loss']
plt.subplot(2, 2, 1)
plt.title('loss')
plt.plot(np.arange(len(loss_history)), loss_history)

plt.subplot(2, 2, 2)
plt.title('acc')
plt.plot(np.arange(len(acc_history)), acc_history)

plt.subplot(2, 2, 3)
plt.title('predict')
y_predict = tf.argmax(model.predict(iris.data), axis=1)
# print("F-score: {0:.2f}".format(f1_score(y_predict, iris.data, average='micro')))
color = np.where(y_predict == 1, 'r', y_predict)
color = np.where(y_predict == 2, 'g', color)
color = np.where(y_predict == 0, 'b', color)
plt.scatter(iris.data[:, 0], iris.data[:, 1], color=color)
plt.show()
