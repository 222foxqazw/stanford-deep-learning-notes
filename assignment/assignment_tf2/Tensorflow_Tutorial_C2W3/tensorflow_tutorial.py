#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: tensorflow_tutorial.py
@time: 2020/3/19 13:58
@desc:
"""
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from assignment.assignment_tf2.Tensorflow_Tutorial_C2W3.tf_utils import load_dataset, random_mini_batches, \
    convert_to_one_hot, predict

np.random.seed(1)


def initialize_parameters():
    tf.random.set_seed(1)

    W1 = tf.Variable(tf.initializers.GlorotUniform(seed=1)(shape=(25, 12288)))
    b1 = tf.Variable(tf.initializers.zeros()(shape=(25, 1)))
    W2 = tf.Variable(tf.initializers.GlorotUniform(seed=1)(shape=(12, 25)))
    b2 = tf.Variable(tf.initializers.zeros()(shape=(12, 1)))
    W3 = tf.Variable(tf.initializers.GlorotUniform(seed=1)(shape=(6, 12)))
    b3 = tf.Variable(tf.initializers.zeros()(shape=(6, 1)))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}

    return parameters


def forward_propagation(X, parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    Z1 = W1 @ X + b1
    A1 = tf.nn.relu(Z1)
    Z2 = W2 @ A1 + b2
    A2 = tf.nn.relu(Z2)
    Z3 = W3 @ A2 + b3

    return Z3


def compute_cost(Z3, Y):
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    return cost


def model(X_train, Y_train, X_test, Y_test, learning_rate=0.0001,
          num_epochs=1500, minibatch_size=32, print_cost=True):
    tf.random.set_seed(1)  # to keep consistent results
    m = X_train.shape[1]  # (m : number of examples in the train set)

    def train_and_test(X, Y, parameters):
        seed = 3
        epoch_cost = 0.  # Defines a cost related to an epoch
        num_minibatches = int(m / minibatch_size)  # number of minibatches of size minibatch_size in the train set
        seed = seed + 1
        minibatches = random_mini_batches(X, Y, minibatch_size, seed)
        out = None
        for minibatch in minibatches:
            # Select a minibatch
            (minibatch_X, minibatch_Y) = minibatch
            with tf.GradientTape() as tape:
                minibatch_X = tf.convert_to_tensor(minibatch_X, dtype=tf.float32)
                minibatch_Y = tf.convert_to_tensor(minibatch_Y, dtype=tf.float32)
                # Forward propagation: Build the forward propagation in the tensorflow graph
                Z3 = forward_propagation(minibatch_X, parameters)

                # Cost function: Add cost function to tensorflow graph
                minibatch_cost = compute_cost(Z3, minibatch_Y)

            # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.

            W1 = parameters['W1']
            b1 = parameters['b1']
            W2 = parameters['W2']
            b2 = parameters['b2']
            W3 = parameters['W3']
            b3 = parameters['b3']
            grads = tape.gradient(minibatch_cost, [W1, b1, W2, b2, W3, b3])
            optimizer.apply_gradients(zip(grads, [W1, b1, W2, b2, W3, b3]))
            if out is None:
                out = Z3
            else:
                out = tf.concat([out, Z3], 1)
            epoch_cost += minibatch_cost / num_minibatches

        parameters = {"W1": W1,
                      "b1": b1,
                      "W2": W2,
                      "b2": b2,
                      "W3": W3,
                      "b3": b3}
        # Print the cost every epoch
        if print_cost == True and epoch % 100 == 0:
            print("Cost after epoch %i: %f" % (epoch, epoch_cost))
        if print_cost == True and epoch % 5 == 0:
            costs.append(epoch_cost)

        # Calculate the correct predictions
        correct_prediction = tf.equal(tf.argmax(out), tf.argmax(Y))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        return costs, accuracy, parameters

    parameters = initialize_parameters()
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
    costs = []

    for epoch in range(num_epochs):
        costs, train_accuracy, parameters = train_and_test(X_train, Y_train, parameters)

    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
    Y_test = tf.convert_to_tensor(Y_test, dtype=tf.float32)
    out = predict(X_test, parameters)
    # Calculate the correct predictions
    correct_prediction = tf.equal(tf.argmax(out), tf.argmax(Y_test))
    # Calculate accuracy on the test set
    test_accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    print("Parameters have been trained!")
    print("Train Accuracy:", train_accuracy)
    print("Test Accuracy:", test_accuracy)

    return parameters


if __name__ == '__main__':
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

    X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
    X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T
    X_train = X_train_flatten / 255.
    X_test = X_test_flatten / 255.
    Y_train = convert_to_one_hot(Y_train_orig, 6)
    Y_test = convert_to_one_hot(Y_test_orig, 6)
    parameters = model(X_train, Y_train, X_test, Y_test)
