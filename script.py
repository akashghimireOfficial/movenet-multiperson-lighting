import numpy as np
import tensorflow as tf
import cv2 as cv
import matplotlib.pyplot as plt
import pandas as pd
import os
from glob import glob

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

test_video_dire=os.path.join('test_video')
test_img_dire=os.path.join('test_img')

model_dire=os.path.join('saved_model')

model = tf.saved_model.load(model_dire)
movenet = model.signatures['serving_default']

def get_keypoint_dict(outputs):
    outputs=outputs['output_0'].numpy()
    keypoints_dict={'Person_1':{},'Person_2':{},'Person_3':{},'Person_4':{},'Person_5':{},'Person_6':{}}
    for num,key in enumerate(keypoints_dict.keys()):
        keypoints_dict[key]['keypoints']=outputs[:,num,:51].reshape(17,3)
        keypoints_dict[key]['bbox']=outputs[:,num,51:].flatten()
    return keypoints_dict
    


def draw_keypoints(img,keypoints_dict,confidence):
    Y,X,c=img.shape
    for num,key in enumerate(keypoints_dict.keys()):
        for y,x,conf in keypoints_dict[key]['keypoints']:
            if conf>confidence:
                cv.circle(img,(int(X*x),int(Y*y)),6, (0,255,0), -1)    


test_direlist=glob('test_video/*.mp4')

if __name__ == "__main__":
    cap=cv.VideoCapture(test_direlist[0])
    while cap.isOpened():
        success,img=cap.read()
        if success:
            input_img=img.copy()
            input_img=cv.resize(input_img,(608,320))
            input_img=np.array(input_img,dtype='int32')
            input_img=tf.convert_to_tensor(input_img)
            input_img=tf.expand_dims(input_img,axis=0)
            outputs=movenet(input_img)
            keypoints_dict=get_keypoint_dict(outputs)
            draw_keypoints(img,keypoints_dict,0.5)
            cv.imshow('demo',img)
            if cv.waitKey(1) & 0xFF==ord('q'):
                break
                
        else:
            break
    cap.release()
    cv.destroyAllWindows()
            