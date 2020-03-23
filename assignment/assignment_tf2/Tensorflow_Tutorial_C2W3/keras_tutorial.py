#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: keras_tutorial.py
@time: 2020/3/23 18:25
@desc:
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, losses

from assignment.assignment_tf2.Tensorflow_Tutorial_C2W3.tf_utils import load_dataset, convert_to_one_hot

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T
X_train = X_train_flatten / 255.
X_test = X_test_flatten / 255.
Y_train = convert_to_one_hot(Y_train_orig, 6)
Y_test = convert_to_one_hot(Y_test_orig, 6)

# 构建数据集对象
train_dataset = tf.data.Dataset.from_tensor_slices((X_train.T, Y_train.T))
# 批量训练
train_dataset = train_dataset.batch(32)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test.T, Y_test.T))
# 批量训练
test_dataset = test_dataset.batch(32)

model = keras.Sequential([
    layers.Dense(25, activation='relu', input_shape=(12288,)),
    layers.Dense(12, activation='relu'),
    layers.Dense(6)])
model.summary()
model.compile(optimizer=optimizers.Adam(learning_rate=0.0001),
              loss=losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy']  # 设置测量指标为准确率
              )

history = model.fit(train_dataset, epochs=1500)
model.evaluate(test_dataset)
