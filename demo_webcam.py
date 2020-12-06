# Khatna Bold
# Adapted from: https://github.com/cansik/yolo-hand-detection/tree/master/ 
import argparse
import cv2
import numpy as np
from tensorflow.keras import models
from yolo import YOLO
from PIL import Image
import utils

# Argument parser
ap = argparse.ArgumentParser()
ap.add_argument('-n', '--network', default="prn", help='Network Type: normal / tiny / prn / v4-tiny')
ap.add_argument('-d', '--device', default=0, help='Device to use')
ap.add_argument('-s', '--size', default=512, help='Size for yolo')
ap.add_argument('-c', '--confidence', default=0.2, help='Confidence for yolo')
ap.add_argument('-m', '--model', default="asl_binary", help="Model: asl / asl_binray / fingers")
args = ap.parse_args()

# Application setup
if args.network == "normal":
    print("loading yolo...")
    yolo = YOLO("models/cross-hands.cfg", "models/cross-hands.weights", ["hand"])
elif args.network == "prn":
    print("loading yolo-tiny-prn...")
    yolo = YOLO("models/cross-hands-tiny-prn.cfg", "models/cross-hands-tiny-prn.weights", ["hand"])
elif args.network == "v4-tiny":
    print("loading yolov4-tiny-prn...")
    yolo = YOLO("models/cross-hands-yolov4-tiny.cfg", "models/cross-hands-yolov4-tiny.weights", ["hand"])
else:
    print("loading yolo-tiny...")
    yolo = YOLO("models/cross-hands-tiny.cfg", "models/cross-hands-tiny.weights", ["hand"])

yolo.size = int(args.size)
yolo.confidence = float(args.confidence)

if args.model == "asl":
    model = models.load_model('models/contrast_dropout.h5')
    alphabet = 'ABCDEFGHIKLMNOPQRSTUVWXY'
    size = (28, 28)
    rescale = True
    gray = True
    binary = False
elif args.model == 'asl_binary':
    model = models.load_model('models/binarized.h5')
    alphabet = 'ABCDEFGHIKLMNOPQRSTUVWXY'
    size = (128, 128)
    rescale = False
    gray = True
    binary = True
elif args.model == 'fingers':
    model = models.load_model('models/finger_binarized.h5')
    alphabet = '012345'
    size = (128, 128)
    rescale = False
    gray = True
    binary = True

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

    prediction = utils.pred_window(
        window, model, alphabet, size, rescale, gray, binary
    )
    
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
