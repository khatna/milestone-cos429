Run `models/download_models.sh` and `python demo_webcam.py` to open the application. The webcam should start working. 

The application takes two 3 key-inputs:

1. "c" - record a letter. Tap "c" again after recording three letters, to verify your code.
2. "d" - Turn hand detection off. This will improve FPS
3. "q" - Quit the application.

Please use your left hand!

For best results, make sure the background is uniform, and contrasts with your hand

# Datasets and Models used

* Dataset: https://www.kaggle.com/datamunge/sign-language-mnist
* Hand Detection model: https://github.com/cansik/yolo-hand-detection

# Dependencies

* TensorFlow v2
* OpenCV
* PIL
* numpy