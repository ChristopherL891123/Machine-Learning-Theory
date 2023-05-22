import tensorflow as tf
import matplotlib.pyplot as plt


# define linear regression
def linear_regression(m, x, b):
    return m * x + b


# define Mean Square Error; Loss function
def MSE(y, y_hat):
    return tf.reduce_sum((y_hat - y) ** 2) / len(y)


# set up y data
y = tf.constant([1.86, 1.31, 0.62, 0.33, 0.09, -0.67, -1.23, -1.37])

# set up the m parameter; slope
m = tf.Variable([0.9], dtype=tf.float32)

# set up x input for the model(graph)
x_input = tf.constant([i for i in range(8)], dtype=tf.float32)

# set up the b parameter; intercept
b = tf.Variable([0.1], dtype=tf.float32)

# Forward Propagation
y_hat = linear_regression(m, x_input, b)

# Gradient Descent
optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.01)

# iterative version
epochs = 100

for epoch in range(epochs): # iteratively calculate gradients for each epoch
    with tf.GradientTape() as gt: # record operations on parameters to compute gradients
        y_hat = linear_regression(m, x_input, b) 
        C = MSE(y, y_hat)

    gradients = gt.gradient(C, [m, b])
    optimizer.apply_gradients(zip(gradients, [m, b]))

    print("Epoch =", epoch, "Loss =", C.numpy(), "m Gradient =", gradients[0].numpy(), "b Gradient =", gradients[1].numpy())
