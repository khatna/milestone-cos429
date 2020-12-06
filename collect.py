import cv2
import numpy as np 
import os

vc = cv2.VideoCapture(0)

letter = input('Enter letter: ')
collecting = False

i = 0
while(True):
    _, frame = vc.read()
    frame = np.flip(frame, axis=1)
    H, W, _ = frame.shape
    ref = int(min(H, W) / 3)

    # Make detection window (not centered right now)
    frame[ref:2*ref, ref, 1]   = 255
    frame[ref, ref:2*ref, 1]   = 255
    frame[ref:2*ref, 2*ref, 1] = 255
    frame[2*ref, ref:2*ref, 1] = 255

    # capture every 10th frame
    if collecting:
        img = frame[ref:2*ref, ref:2*ref]
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        cv2.imwrite(os.path.join('finger_dataset',letter,str(i)+'.jpg'), thresh)
        i += 1
        print(i)
        
    cv2.imshow("Face", frame)
    key = cv2.waitKey(1)
    if key == ord('s'):
        collecting = not collecting
    
    if(i > 299):
        break

vc.release()
cv2.destroyAllWindows()