import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import models

inverter = models.load_model('models/inverted.h5') 
model    = models.load_model('models/binary.h5')
alphabet = 'ABCDEFGHIKLMNOPQRSTUVWXY'

# predict a window
def pred_window(window):
    window = cv2.cvtColor(window, cv2.COLOR_BGR2GRAY)
    _, window = cv2.threshold(window,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    window = np.expand_dims(window, 0)
    window = np.expand_dims(window, 3)

    in_tensor = tf.convert_to_tensor(window)
    in_tensor = tf.image.resize_with_pad(in_tensor, 128, 128)
    if inverter(in_tensor)[0,0] > 0.5:
        in_tensor = tf.math.abs(in_tensor - 255)
    
    lbl = np.argmax(model(in_tensor).numpy())
    return alphabet[lbl]

def put_text(frame, text):
    H, W, _ = frame.shape
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, H - 20)
    fontScale              = 1
    fontColor              = (0,255,0)
    lineType               = 2

    frame = np.array(frame)
    cv2.putText(frame, text, 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        lineType)

    return frame

def put_letter(frame, letter):
    H, W, _ = frame.shape
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (W - 30, H - 20)
    fontScale              = 1
    fontColor              = (0,255,0)
    lineType               = 2

    frame = np.array(frame)
    cv2.putText(frame, letter, 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        lineType)

    return frame
