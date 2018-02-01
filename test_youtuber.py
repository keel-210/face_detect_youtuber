import os
import numpy as np
import cv2
import tensorflow as tf

CLASSES_NUM = 24
IMG_SIZE = 28
IMG_PIXELS = IMG_SIZE*IMG_SIZE*3

#----------------------------------------------------------------------
#画像をNumpy配列に変換する
# 画像のあるディレクトリ
train_img_dirs = ['KizunaAI','MiraiAkari','Nekomasu','Shiro','KaguyaLuna','Cafeno-Zombiko','DD','Fuji-Aoi',
                  'Fujisaki-Yua','hoonie','Kurumi-chan','MIDI','Miial','Mochi-Hiyoko','Moscowmule','MyuMyu',
                  'Neets','Nemu','Nora-cat','Raiden-Kasuka', 'Suzuki-Secil', 'Todoki-Uka', 'Tokinosora', 'Umakoshi-Kentaro']

# 学習画像データ
train_image = []
# 学習データのラベル
train_label = []

for a, d in enumerate(train_img_dirs):
    # ./data/以下の各ディレクトリ内のファイル名取得
    files = os.listdir('./test_data/' + d)
    for f in files:
        # 画像読み込み
        img = cv2.imread('./test_data/' + d + '/' + f)
        # 1辺がIMG_SIZEの正方形にリサイズ
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        # 1列にして
        img = img.flatten().astype(np.float32)/255.0
        train_image.append(img)

        # one_hot_vectorを作りラベルとして追加
        tmp = np.zeros(CLASSES_NUM)
        tmp[a] = 1
        train_label.append(tmp)
    print(str(train_img_dirs[a])+" : Complete")
        
    
# numpy配列に変換
train_image = np.asarray(train_image)
train_label = np.asarray(train_label)
#----------------------------------------------------------------------
#tf.reset_default_graph()
#----------------------------------------------------------------------
#ここから学習ネットワーク定義
ses = tf.InteractiveSession()

# placeholder用意 xは学習用画像
x = tf.placeholder(tf.float32, [None, IMG_PIXELS])
# ｙ_は学習用ラベル
y_ = tf.placeholder(tf.float32, [None, CLASSES_NUM])

# weightとbias
# さっきの例ではw * xだったけど、今回はw * x + b
W = tf.Variable(tf.zeros([IMG_PIXELS, CLASSES_NUM]),name='W')
b = tf.Variable(tf.zeros([CLASSES_NUM]),name='b')
ses.run(tf.global_variables_initializer())

# 第一層のweightsとbiasのvariable
W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 3, 32], stddev=0.1),name='W_conv1')
b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]),name='b_conv1')

# 画像を28x28にreshape
x_image = tf.reshape(x, [-1,IMG_SIZE,IMG_SIZE,3])

# 第一層のconvolutionalとpool
h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)
h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 画像を784の一次元から28x28の二次元に変換する
# 4つめの引数はチャンネル数
x_image = tf.reshape(x, [-1,IMG_SIZE,IMG_SIZE,3])

# 第二層
W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1),name='W_conv2')
b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]),name='b_conv2')
h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)
h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 第一層と第二層でreduceされてできた特徴に対してrelu
W_fc1 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1024], stddev=0.1),name='W_fc1')#変更元は7*7*64
b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]),name='b_fc1')
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])#変更元は7*7*64
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 出来上がったものに対してSoftmax
W_fc2 = tf.Variable(tf.truncated_normal([1024, CLASSES_NUM], stddev=0.1),name='W_fc2')
b_fc2 = tf.Variable(tf.constant(0.1, shape=[CLASSES_NUM]),name='b_fc2')
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# 交差エントロピー
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))

# 今回はGradientDescentOptimizerではなく、AdamOptimizer
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# accuracyを途中確認するための入れ物
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# variable初期化
ses.run(tf.global_variables_initializer())

#----------------------------------------------------------------------

#----------------------------------------------------------------------
init = tf.global_variables_initializer()
saver = tf.train.Saver()
sess = tf.InteractiveSession()
sess.run(init)
#saver = tf.train.import_meta_graph('./Model/model.ckpt.meta')#注意
saver = tf.train.Saver()
 #学習モデルの読み込み、利用できるようにする
saver.restore(sess, "./Model/model.ckpt")
print(sess.run(W_fc2))
print(sess.run(b))
print('Restored a model')
acc = accuracy.eval(session=ses,feed_dict={x:train_image, y_:train_label, keep_prob: 1.0})
print('Test Accuracy: %s' %acc)