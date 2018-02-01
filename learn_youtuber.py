# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 00:20:16 2017

@author: KEEL
"""

import tensorflow as tf
import random
import os
import cv2
import numpy as np

CLASSES_NUM = 5
IMG_SIZE = 28
IMG_PIXELS = IMG_SIZE*IMG_SIZE*3
train_img_dirs = ['KizunaAI','MiraiAkari','Nekomasu','Shiro','KaguyaLuna']
# 学習画像データ 
train_image = []
# 学習データのラベル
train_label = []

    
def image2data(directory):
    # 学習画像データ
    train_images = []
    # 学習データのラベル
    train_labels = []
    for a, d in enumerate(train_img_dirs):
        # ./data/以下の各ディレクトリ内のファイル名取得
        files = os.listdir(directory + d)
        for f in files:
            # 画像読み込み
            img = cv2.imread(directory + d + '/' + f)
            # 1辺がIMG_SIZEの正方形にリサイズ
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            # 1列にして
            img = img.flatten().astype(np.float32)/255.0
            train_images.append(img)

            # one_hot_vectorを作りラベルとして追加
            tmp = np.zeros(CLASSES_NUM)
            tmp[a] = 1
            train_labels.append(tmp)
        print(str(train_img_dirs[a])+" : Complete")
        
    
    # numpy配列に変換
    print(len(np.asarray(train_images)))
    return np.asarray(train_images)
    
def labering (directory):
    # 学習データのラベル
    train_labels = []
    for a, d in enumerate(train_img_dirs):
        # ./data/以下の各ディレクトリ内のファイル名取得
        files = os.listdir(directory + d)
        for f in files:
            # one_hot_vectorを作りラベルとして追加
            tmp = np.zeros(CLASSES_NUM)
            tmp[a] = 1
            train_labels.append(tmp)
        print(str(train_img_dirs[a])+" : Complete")
        
    # numpy配列に変換
    train_label = np.asarray(train_labels)
    return train_label

def Two_CNN_model():
    # placeholder用意 xは学習用画像
    x = tf.placeholder(tf.float32, [None, IMG_PIXELS])
    # ｙ_は学習用ラベル
    y_ = tf.placeholder(tf.float32, [None, CLASSES_NUM])

    # weightとbias
    # さっきの例ではw * xだったけど、今回はw * x + b
    W = tf.Variable(tf.zeros([IMG_PIXELS, CLASSES_NUM]),name='W')
    b = tf.Variable(tf.zeros([CLASSES_NUM]),name='b')

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

    tf.add_to_collection('vars', W)
    tf.add_to_collection('vars', b)
    tf.add_to_collection('vars', W_conv1)
    tf.add_to_collection('vars', b_conv1)
    tf.add_to_collection('vars', W_conv2)
    tf.add_to_collection('vars', b_conv2)
    tf.add_to_collection('vars', W_fc1)
    tf.add_to_collection('vars', b_fc1)
    tf.add_to_collection('vars', W_fc2)
    tf.add_to_collection('vars', b_fc2)
    
    model = {'x':x,'y_conv':y_conv,'y_':y_}
    print("Model Constructed")
    return model

def TrainAndSave(model):

    """
    # Model
    設計したモデルのうち、訓練で使うのは、X, Y, Y_のみなので、それらを取り出す。
    """
    x, y_conv, y_ = model['x'], model['y_conv'], model['y_']

    """
    # Functions
    訓練で利用する関数をそれぞれ定義する。
    """
    # 交差エントロピー
    cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))

    # 今回はGradientDescentOptimizerではなく、AdamOptimizer
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    # accuracyを途中確認するための入れ物
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    """
    # Setting
    初期化。saverは、モデルを保存するためのインスタンス。
    """
    print("Constructing...")
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    print("Session : OK")
    
   
    """
    # Training
    100バッチずつ訓練するエポックを１回繰り返す。
    訓練エラー、訓練精度、テスト精度を算出する。
    """
    

    STEPS = 30 # 学習ステップ数
    BATCH_SIZE = 200 # バッチサイズ
    print(len(train_image))
    for i in range(STEPS):
        random_seq = list(range(len(train_image)))
        random.shuffle(random_seq)
  #print(int(len(train_image)/BATCH_SIZE))
        for j in range(int(len(train_image)/BATCH_SIZE)):
            batch = BATCH_SIZE * j
            train_image_batch = []
            train_label_batch = []
            for k in range(BATCH_SIZE):
                train_image_batch.append(train_image[random_seq[batch + k]])
                train_label_batch.append(train_label[random_seq[batch + k]])

            train_step.run(session=sess,feed_dict={x: train_image_batch, y_: train_label_batch, keep_prob: 0.5})
            train_accuracy = accuracy.eval(feed_dict={x:train_image_batch, y_: train_label_batch, keep_prob: 0.5})
            print("step %d, training accuracy %g"%(i, train_accuracy))

# Save a model
    
    if i % STEPS-1 == 0:
        saver = tf.train.Saver()
        for f in os.listdir('../model/'):
            os.remove('../model/'+f)
        saver.save(sess, '../model/test_model')
        
        print('Saved a model.')

        sess.close()
    
def restore_model(directory,model):
    
    image2data(directory)
    
    # Model
    x, y_conv, y_ = model['x'], model['y_conv'], model['y_']

    # Function
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1)), tf.float32))

    """
    # Setting
    SaverとSessionのインスタンスを生成し、モデルを復元する。
    """
    sess = tf.Session()
    saver = tf.train.Saver()
    
    saver.restore(sess, '../model/model.ckpt')
    print('Restored a model')

    test_accuracy = sess.run(accuracy, feed_dict={x: train_image, y_: train_label})
    print('Test Accuracy: %s' % test_accuracy)
    
if __name__ == '__main__':
    train_image = image2data('./data/')
    train_label = labering('./data/')
    model = Two_CNN_model()
    TrainAndSave(model)
    restore_model('./data/',model)