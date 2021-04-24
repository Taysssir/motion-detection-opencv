# motion-detection-opencv
This repository implement a Python application  using OpenCV and using K-Nearest Neighbor algorithm (KNN) to detect a motion from a video stream, video file or a webcam.

K-Nearest Neighbor Algorithm (KNN)
```
def createBackgroundSubtractorKNN(history=None, dist2Threshold=None, detectShadows=None): 
    """
    createBackgroundSubtractorKNN([, history[, dist2Threshold[, detectShadows]]]) -> retval
    .   @brief Creates KNN Background Subtractor
    .   
    .   @param history Length of the history.
    .   @param dist2Threshold Threshold on the squared distance between the pixel and the sample to decide
    .   whether a pixel is close to that sample. This parameter does not affect the background update.
    .   @param detectShadows If true, the algorithm will detect shadows and mark them. It decreases the
    .   speed a bit, so if you do not need this feature, set the parameter to false.
    """
```
#### Download and Usage

```
$ git clone https://github.com/Taysssir/motion-detection-opencv.git
$ cd motion-detection-opencv
$ python motion_detector.py -h
```

Output Console : You will find something similar to this:

```
usage: motion_detector.py [-h] -i INPUT -s {True,False}

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        path to the input video file or put 0 for launching Webcam
  -s {True,False}, --save {True,False}
                        Save Motion detection True/False
```

For Using Webcam and save motion detction :

```
python motion_detector.py -i 0 -s True
```
For Using Video File path and save motion detction :

python motion_detector.py -i videos/Fire.mp4 -s True
