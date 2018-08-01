import tensorflow as tf
import matplotlib.pyplot as plt

tf.reset_default_graph()
#Variables

input_data = tf.placeholder(dtype=tf.float32, shape = None)
output_data = tf.placeholder(dtype=tf.float32, shape = None)

slope = tf.Variable(0.5, dtype=tf.float32)
intercept = tf.Variable(0.1, dtype=tf.float32)

model_operation = slope * input_data + intercept

error = model_operation - output_data
squared_error = tf.square(error)
loss = tf.reduce_mean(squared_error)
#optimizers
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.005)
train = optimizer.minimize(loss)
init = tf.global_variables_initializer()

#Training Data

x_values = [0, 1, 2, 3, 4]
y_values = [1, 3, 5, 7, 9]

with tf.Session() as sess:
    sess.run(init)
    for i in range(2000):
        sess.run(train, feed_dict={input_data: x_values, output_data: y_values})
        if i % 100 == 0:
            print(sess.run([slope, intercept]))
            plt.plot(x_values, sess.run(model_operation, feed_dict={input_data: x_values}))
    print(sess.run(loss, feed_dict={input_data: x_values, output_data: y_values}))
    plt.plot(x_values, y_values, 'ro', 'Training Data')
    plt.plot(x_values, sess.run(model_operation, feed_dict={input_data: x_values}))

    plt.show()
