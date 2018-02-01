# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 23:41:03 2018

@author: KEEL
"""


import os
import numpy as np
import cv2
import tensorflow as tf
import glob

IMG_SIZE = 28
IMG_PIXELS = IMG_SIZE*IMG_SIZE*3

#----------------------------------------------------------------------
#画像をNumpy配列に変換する
# 画像のあるディレクトリ
train_img_dirs = ['KizunaAI','MiraiAkari','Nekomasu','Shiro','KaguyaLuna','Cafeno-Zombiko','DD','Fuji-Aoi',
                  'Fujisaki-Yua','hoonie','Kurumi-chan','MIDI','Miial','Mochi-Hiyoko','Moscowmule','MyuMyu',
                  'Neets','Nemu','Nora-cat','Raiden-Kasuka', 'Suzuki-Secil', 'Todoki-Uka', 'Tokinosora', 'Umakoshi-Kentaro']
CLASSES_NUM = len(train_img_dirs)

xml_path = "./lbpcascade_animeface.xml"
out_path = "./face/"
def faceDetect(img_path):
    classifier = cv2.CascadeClassifier(xml_path)
    
    img_count = 1
    face_imgs = []
    #for img_path in img_list:
    org_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    gray_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    face_points = classifier.detectMultiScale(gray_img, \
                                              scaleFactor=1.2, minNeighbors=2, minSize=(1,1))

    for points in face_points:
        
        x, y, width, height =  points

        dst_img = org_img[y:y+height, x:x+width]
        dst_img = cv2.resize(dst_img, (IMG_SIZE,IMG_SIZE))
        face_imgs.append(dst_img)

        face_img = cv2.rectangle(org_img, (x,y), (x+width,y+height), (0, 0, 0), 2)
        new_img_name = out_path + str(img_count) + 'face.jpg'
        cv2.imwrite(new_img_name, face_img)
        
        img_count += 1
    print(img_count)
    if img_count != 1:
        return face_points[0]


def inference(images_placeholder, keep_prob):
    """ モデルを作成する関数

    引数: 
      images_placeholder: inputs()で作成した画像のplaceholder
      keep_prob: dropout率のplace_holder

    返り値:
      cross_entropy: モデルの計算結果
    """
    W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 3, 32], stddev=0.1),name='W_conv1')
    b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]),name='b_conv1')

    x_image = tf.reshape(images_placeholder, [-1,IMG_SIZE,IMG_SIZE,3])

    h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)
    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    x_image = tf.reshape(images_placeholder, [-1,IMG_SIZE,IMG_SIZE,3])

    W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1),name='W_conv2')
    b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]),name='b_conv2')
    h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)
    h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    W_fc1 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1024], stddev=0.1),name='W_fc1')#変更元は7*7*64
    b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]),name='b_fc1')
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])#変更元は7*7*64
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = tf.Variable(tf.truncated_normal([1024, CLASSES_NUM], stddev=0.1),name='W_fc2')
    b_fc2 = tf.Variable(tf.constant(0.1, shape=[CLASSES_NUM]),name='b_fc2')
    y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    #cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
    return y_conv


    
images_placeholder = tf.placeholder("float", shape=(None, IMG_PIXELS))
keep_prob = tf.placeholder("float")

init = tf.global_variables_initializer()

logits = inference(images_placeholder, keep_prob)

sess = tf.InteractiveSession()
sess.run(init)

saver = tf.train.Saver()
saver = tf.train.import_meta_graph('./Model/model.ckpt.meta')#注意
saver.restore(sess, "./Model/model.ckpt")

input_path = './test_img/'
filename = glob.glob(input_path + '*.jpg')

for img_path in input_path:
    frame = cv2.imread(img_path)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#顔の検出
dets = faceDetect(filename)
                
if not isinstance(dets,type(None)):
    x, y, width, height =  dets
    image = frame[y:y+height, x:x+width]
    cv2.rectangle(frame, (x,y), (x+width, y+height), (0, 0, 0), 4)
    cv2.imwrite(filename, frame)
    cv2.imshow("1",image)
    cv2.waitKey(0)
    img = cv2.resize(img.copy(), (28, 28))
    ximage = []
    ximage.append(img.flatten().astype(np.float32)/255.0)
    ximage = np.asarray(ximage)

    print(ximage.shape)
    pred = np.argmax(logits.eval(session=sess,feed_dict={ images_placeholder : ximage, keep_prob: 1.0 }))
    print(pred)
