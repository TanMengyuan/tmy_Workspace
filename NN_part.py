import numpy as np
import tensorflow as tf
import os, re, random

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# random.seed(1)
# Set parameters
BATCH_SIZE = 5

# Loading data
source_path = './room_data/'
target_path = './room_result_SNR/revised_data/'
# target_path = './room_result_SNR/'
train_data, test_data = [], []
sta_test = ['008', '010', '013', '014', '016', '018']
for each in os.listdir(target_path):
    match = re.search(pattern=r'\d{3}_out.npy', string=each)
    if match:
        data_id = match.group(0)[:3]
        train_data.append([source_path + data_id + '.npy', target_path + data_id + '_out.npy'])

for data_id in sta_test:
    test_data.append([source_path + data_id + '.npy', target_path + data_id + '_out.npy'])

# Set neural network
x = tf.placeholder("float", [None, 100])
room_img = tf.reshape(x, [-1, 10, 10, 1])
y_ = tf.placeholder("float", [None, 100])

# CNN part
conv1 = tf.layers.conv2d(  # shape (10, 10, 1)
    inputs=room_img,
    filters=16,
    kernel_size=5,
    strides=1,
    padding='same',
    activation=tf.nn.relu
)  # -> (10, 10, 16)
pool1 = tf.layers.max_pooling2d(
    conv1,
    pool_size=2,
    strides=2,
)  # -> (5, 5, 16)
conv2 = tf.layers.conv2d(pool1, 32, 5, 1, 'same', activation=tf.nn.relu)  # -> (5, 5, 32)
pool2 = tf.layers.max_pooling2d(conv2, 1, 1)  # -> (5, 5, 32)
flat = tf.reshape(pool2, [-1, 5 * 5 * 32])  # -> (5*5*32, )
y = tf.layers.dense(flat, 100)  # output layer

# W = tf.Variable(tf.zeros((100, 100)))
# b = tf.Variable(tf.zeros(100))
# # y = tf.nn.relu(tf.matmul(x, W) + b)
# y = tf.nn.softmax(tf.matmul(x, W) + b)


# Optimizer
with tf.name_scope('loss'):
    loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=y_, logits=y)  # compute cost
    train_op = tf.train.AdamOptimizer(0.01).minimize(loss)
    # loss= -tf.reduce_sum(y_ * tf.log(y))
    tf.summary.scalar('loss', loss)
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

# Accuracy
led_num = tf.cast((tf.reduce_sum(x, axis=1) / 25 - 1e-3) + 1, tf.int32)
top_4 = tf.contrib.framework.sort(tf.nn.top_k(y, 4).indices)
one_hot_y, one_hot_y_, tmp = [], [], tf.constant([-1 for _ in range(4)])
for i in range(BATCH_SIZE):
    one_hot_y.append(
        tf.reduce_sum(tf.one_hot(tf.concat([top_4[i][:led_num[i]], tmp[led_num[i]:]], axis=0), 101), axis=0)
    )
    one_hot_y_.append(
        tf.reduce_sum(
            tf.one_hot(tf.concat([tf.nn.top_k(y_[i], led_num[i], sorted=True).indices, tmp[led_num[i]:]], axis=0), 101),
            axis=0)
    )
# accuracy = tf.reduce_sum(tf.cast(tf.equal(one_hot_y, one_hot_y_), 'float')) / (4 * BATCH_SIZE)
with tf.name_scope('accuracy'):
    accuracy = tf.reduce_sum(tf.multiply(one_hot_y, one_hot_y_)) / (4 * BATCH_SIZE)
    tf.summary.scalar('accuracy', accuracy)

# Init
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
merged = tf.summary.merge_all()

writer = tf.summary.FileWriter("./logs/", sess.graph)


def next_batch(batch_size, list_):
    data_files = random.sample(list_, batch_size)
    xs, ys = np.array([]), np.array([])
    for data in data_files:
        xs = np.append(xs, (np.load(data[0]).reshape(1, 100)[0]))
        ys = np.append(ys, (np.load(data[1])[0]))
    return xs.reshape(-1, 100), ys.reshape(-1, 100)


def build_test_batch(list_):
    xs, ys = np.array([]), np.array([])
    for data in list_:
        xs = np.append(xs, (np.load(data[0]).reshape(1, 100)[0]))
        ys = np.append(ys, (np.load(data[1])[0]))
    return xs.reshape(-1, 100), ys.reshape(-1, 100)


test_batch_xs, test_batch_ys = build_test_batch(test_data)

for i in range(10000):
    batch_xs, batch_ys = next_batch(BATCH_SIZE, train_data)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    if i % 100 == 0:
        _, res = sess.run([accuracy, merged], feed_dict={x: batch_xs, y_: batch_ys})
        writer.add_summary(summary=res, global_step=i)
        print('step: %d' % i, sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys}))
        # print('step: %d' % i, sess.run(accuracy, feed_dict={x: test_batch_xs, y_: test_batch_ys}))
