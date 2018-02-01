# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 10:22:57 2018

@author: KEEL
"""

from tweepy import *
import urllib
import urllib.request
import sys
import datetime
import re
import cv2
import numpy as np
import tensorflow as tf

NUM_CLASSES = 24
IMG_SIZE = 28
IMG_PIXELS = IMG_SIZE*IMG_SIZE*3

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

    W_fc2 = tf.Variable(tf.truncated_normal([1024, NUM_CLASSES], stddev=0.1),name='W_fc2')
    b_fc2 = tf.Variable(tf.constant(0.1, shape=[NUM_CLASSES]),name='b_fc2')
    y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    return y_conv

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
        
        if width > face_points[0,3]:
            face_points[0] = points
        img_count += 1
    print(img_count)
    if img_count != 1:
        return face_points[0]


def get_oauth():
	consumer_key = lines[0]
	consumer_secret = lines[1]
	access_key = lines[2]
	access_secret = lines[3]
	auth = OAuthHandler(consumer_key, consumer_secret)
	auth.set_access_token(access_key, access_secret)
	return auth

class StreamListener(StreamListener):
    # ツイートされるたびにここが実行される
    def on_status(self, status):
        if status.in_reply_to_screen_name=='jdatmtjp':
            print('replyed')
            if 'media' in status.entities:
                print('The reply has media')
                text = re.sub(r'@jdatmtjp ', '', status.text)
                text = re.sub(r'(https?|ftp)(://[\w:;/.?%#&=+-]+)', '', text)
                medias = status.entities['media']
                m =  medias[0]
                media_url = m['media_url']
                print(media_url)
                now = datetime.datetime.now()
                time = now.strftime("%H%M%S")
                filename = '{}.jpg'.format(time)
                try:
                    urllib.request.urlretrieve(media_url, filename)
                except IOError:
                    print("保存に失敗しました")

                frame = cv2.imread(filename)
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                #顔の検出
                dets = faceDetect(filename)
                
                if not isinstance(dets,type(None)):
                    x, y, width, height =  dets
                    image = frame[y:y+height, x:x+width]
                    cv2.rectangle(frame, (x,y), (x+width, y+height), (0, 0, 0), 4)
                    cv2.imwrite(filename, frame)
                    #cv2.imshow("1",image)
                    #cv2.waitKey(0)
                    image = cv2.resize(image.copy(), (28, 28))
                    ximage = []
                    ximage.append(image.flatten().astype(np.float32)/255.0)
                    ximage = np.asarray(ximage)

                    print(ximage.shape)
                    result = logits.eval(session=sess,feed_dict={ images_placeholder : ximage, keep_prob: 1.0 })
                    pred = np.argmax(result)
                    rate = np.amax(result)*100
                    print(pred)
                    VTuber_Name = ['キズナアイ','ミライアカリ','バーチャルのじゃロリ狐娘Youtuberおじさん','電脳少女シロ',
                                   '輝夜 月','カフェ野ゾンビ子','虚拟DD','富士 葵','藤崎由愛','Hoonie','ぜったい天使くるみちゃん',
                                   'ミディ','ミアル','もちひよこ','モスコミュール','アリシア・ソリッド(諸々の事情からこの表記で)','ニーツ',
                                   'バーチャル美少女YouTuberねむ','のらきゃっと','雷電カスカ','スズキセシル','届木ウカ','トキノソラ','馬越健太郎',]
                    message = '.@'+status.author.screen_name+' '+'%5.3f'%rate+'%%の確率で'+'%s'%VTuber_Name[pred]
                else:
                    image = frame
                    cv2.imwrite("original.jpg", image)
                    print("no face")
                    message = '.@'+status.author.screen_name+' This image has no face.'
                try:
                    #画像をつけてリプライ
                    api.update_with_media(filename, status=message, in_reply_to_status_id=status.id)
                except(TweepError, e):
                    print("error response code: " + str(e.response.status))
                    print("error message: " + str(e.response.reason))
                    

# TwitterAPIのログイン情報
f = open('config.txt')
data = f.read()
f.close()
lines = data.split('\n')

images_placeholder = tf.placeholder("float", shape=(None, IMG_PIXELS))
keep_prob = tf.placeholder("float")

logits = inference(images_placeholder, keep_prob)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
saver = tf.train.import_meta_graph('./Model/model.ckpt.meta')
saver.restore(sess, "./Model/model.ckpt")
# streamingを始めるための準備
auth = get_oauth()
api = API(auth)
stream = Stream(auth, StreamListener(), secure=True)
print("Start Streaming!")
stream.userstream()