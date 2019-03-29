from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np

mnist = input_data.read_data_sets('../dataset', one_hot=True)


x = tf.placeholder(name='img', dtype='float', shape=[None, 784])

w1 = tf.Variable(tf.zeros([784, 10]))
b1 = tf.Variable(tf.zeros([10]))

y1 = tf.nn.softmax(tf.matmul(x, w1) + b1)

out = tf.identity(y1, name='out')

y_label = tf.placeholder('float', shape=[None, 10])

print('shape y1: {}'.format(str(y1.shape)))
print('shape log(y1): {}'.format(str(tf.log(y1).shape)))
print('shape y_label: {}'.format(str(y_label.shape)))

cross_entropy = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=y1, labels=y_label))

train_step = tf.train.GradientDescentOptimizer(0.002).minimize(cross_entropy)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

# print('w.shape:{}'.format(w.eval(sess)[:, 0].reshape([28, 28, -1]).shape))

for i in range(20001):
    if i % 100 == 0:
        print('saving step : {}'.format(i))
        # for j in range(10):
        #     w_i = w1.eval(sess)[:, j].reshape([28, 28])
        #     np.savetxt('w_' + str(i) + '_' + str(j), w_i)
        print('{} written'.format(i))
    batch_xs, batch_ys = mnist.train.next_batch(50)
    sess.run(train_step, feed_dict={x: batch_xs, y_label: batch_ys})

# saver = tf.train.Saver()
# saver.save(sess.graph, 'model.pb')

correct_pred = tf.equal(tf.argmax(y1, 1), tf.argmax(y_label, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, 'float'))

print(sess.run(accuracy, feed_dict={x:mnist.test.images, y_label:mnist.test.labels}))

# saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)
# saver.save(sess, 'fc_softmax.ckpt')
#
# converter = tf.contrib.lite.TFLiteConverter.from_session(sess, [x], [out])
# tflite_model = converter.convert()
# with open('fc_softmax.tflite', 'wb') as output:
#     output.write(tflite_model)