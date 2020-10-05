#================================================================
#
#   File name   : detect_mnist.py
#   Author      : Jangmin
#   Created date: 2020-10-05
#   GitHub      : https://github.com/JangminSon
#   Description : Convert Models To TF Lite
#
#================================================================
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import numpy as np
import random
import time
import tensorflow as tf
from yolov3.yolov4 import Create_Yolo
from yolov3.utils import detect_image
from yolov3.configs import *

yolo = Create_Yolo(input_size=YOLO_INPUT_SIZE, CLASSES=TRAIN_CLASSES)
yolo.load_weights(f"./checkpoints/yolov3_custom") # use keras weights

c = tf.lite.TFLiteConverter.from_keras_model(yolo)
c.target_spec.supported_ops =[tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]

tflite_model = c.convert()
open("./yolov3_custom.tflite", 'wb').write(tflite_model)
