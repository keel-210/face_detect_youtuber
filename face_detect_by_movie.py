# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 21:47:42 2017

@author: KEEL
"""

import cv2
import glob

VTuber_name = 'uka'
video_path = './movie/'+VTuber_name+'.mp4'
video_name = VTuber_name + '_'
output_path = './faces/'
out_face_path = './face/'
xml_path = "./lbpcascade_animeface.xml"

def movie_to_image(num_cut):

    capture = cv2.VideoCapture(video_path)
    print(video_name)
    img_count = 0
    frame_count = 0

    while(capture.isOpened()):

        ret, frame = capture.read()
        if ret == False:
            break

        if frame_count % num_cut == 0:
            img_file_name = output_path + str(img_count) + ".jpg"
            cv2.imwrite(img_file_name, frame)
            img_count += 1

        frame_count += 1

    capture.release()

def face_detect(img_list):

    classifier = cv2.CascadeClassifier(xml_path)

    img_count = 1
    for img_path in img_list:

        org_img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        gray_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        face_points = classifier.detectMultiScale(gray_img, \
                scaleFactor=1.2, minNeighbors=2, minSize=(1,1))

        for points in face_points:

            x, y, width, height =  points

            dst_img = org_img[y:y+height, x:x+width]

            face_img = cv2.resize(dst_img, (64,64))
            new_img_name = out_face_path + video_name + str(img_count) + 'face.jpg'
            
            cv2.imwrite(new_img_name, face_img)
            img_count += 1
    print(img_count)
if __name__ == '__main__':

    movie_to_image(int(10))
    print('cut finished')
    images = glob.glob(output_path + '*.jpg')
    face_detect(images)