# Khatna Bold
# Adapted from: https://github.com/cansik/yolo-hand-detection/tree/master/ 
import argparse
import cv2
import numpy as np
from yolo import YOLO
from PIL import Image
import utils

yolo = YOLO("models/cross-hands-tiny-prn.cfg", "models/cross-hands-tiny-prn.weights", ["hand"])
yolo.size = int(512)
yolo.confidence = float(0.2)

# main Loop
print("starting webcam...")
vc = cv2.VideoCapture(0)
detecting = True

code = ''
text = '---'
correct_count = 0
while True:
    _, frame = vc.read()
    frame = np.flip(frame, axis=1)
    H, W, _ = frame.shape
    ref = int(min(H, W) / 3)
    
    window = None
    if detecting:
        width, height, inference_time, results = yolo.inference(frame)    
        for detection in results:
            id, name, confidence, x, y, w, h = detection
            h = min(h, H - y)
            w = min(w, W - x)
            # draw a bounding box rectangle
            frame[y:y+h-1, x, 1]   = 255
            frame[y, x:x+w-1, 1]   = 255
            frame[y:y+h-1, x+w-1, 1] = 255
            frame[y+h-1, x:x+w-1, 1] = 255
            window = frame[y:y+h-1, x:x+w-1, :]

    if not detecting or len(results) == 0:
        # Make detection window (not centered right now)
        frame[ref:2*ref, ref, 1]   = 255
        frame[ref, ref:2*ref, 1]   = 255
        frame[ref:2*ref, 2*ref, 1] = 255
        frame[2*ref, ref:2*ref, 1] = 255

        # get image from detection window
        window = frame[ref:2*ref, ref:2*ref,:]

    prediction = utils.pred_window(window)
    
    frame = utils.put_letter(frame, prediction)

    # Input Logic
    key=cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('d'):
        detecting = not detecting
    elif key == ord('c'):
        if len(code) < 3:
            code += prediction
            text = code + '-' * (3 - len(code))
            if len(code) == 3:
                text = '---'
        elif correct_count != 3 and prediction == code[correct_count]:
            correct_count += 1
            text = '*' * correct_count + '-' * (3 - correct_count)
        if correct_count == 3:
            text = 'AUTHORIZED'

    # show image
    frame = utils.put_text(frame, text)
    cv2.imshow("Capturing", frame)

vc.release()
cv2.destroyAllWindows()
