import cv2
import argparse
import numpy as np
import time

# Construct the argument parser and parse the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', required=True, help='path to the input video file or put 0 for launching Webcam')
parser.add_argument('-s', '--save', required=True, choices=['True', 'False'],help='Save Motion detection True/False')
args   = vars(parser.parse_args())

# Parse Args Video File Path / Webcam
if isinstance(args['input'], str) and args['input'] != "0":
    video_source = args['input']
    args['input']= args['input'][:-4]
    video        = cv2.VideoCapture(video_source)
else:
    video_source = 0
    args['input']= "videos/webcam"
    video        = cv2.VideoCapture(video_source)
time.sleep(2)

if video.isOpened() == False:
    print("[INFO] Unable to read the camera feed, Wrong Input video")

# Create Background Subtractor KNN 
knnSubtractor     = cv2.createBackgroundSubtractorKNN(100, 400, True)

# Motion detection parameters
percentage        = 0.2
thresholdCount    = 1500
detectionText     = "Movement is Detected"
textColor         = (255, 255, 255)
titleTextSize     = 1.2
titleTextPosition = (50, 50)
frameID           = 0

# Prepare video Writer
# if args['save'] : 
#     fourcc = cv2.VideoWriter_fourcc(*'MP4V')
#     writer = None

print("[INFO] Processing...(Press q to stop)")

while(1):
    # Return Value and the current frame
    ret, frame = video.read()
    frameID += 1
    print('[INFO] Frame Number: %d' % (frameID))

    #  Check if a current frame actually exist
    if not ret:
        break

    if args['save'] :
        fourcc                      = cv2.VideoWriter_fourcc(*'MP4V')
        (frame_height, frame_width) = frame.shape[:2]
        output_file_name            = args['input'] + '_motion_detection_{}_{}'.format(frame_height, frame_width)+'.mp4'
        writer                      = cv2.VideoWriter( output_file_name, fourcc, 20.0, (frame_width, frame_height))
        
        output_motion_file_name     = args['input'] + '_motion_{}_{}'.format(frame_height, frame_width)+'.mp4'
        writer_motion               = cv2.VideoWriter(output_motion_file_name, fourcc, 20.0, (frame_width, frame_height),0)
        
        pixel_total                 = frame_height * frame_width
        thresholdCount              = (percentage * pixel_total) / 100

        print('[INFO] frame_height={}, frame_width={}'.format(frame_height, frame_width))
        print('[INFO] Number of pixels of the frame: {}'.format(pixel_total))
        print('[INFO] Number of pixels to trigger Detection ({}%) : {}'.format(percentage,thresholdCount))

    print("\n[INFO] Perform Movement Detection: KNN")

    start_time         = time.time()
    knnMask            = knnSubtractor.apply(frame)
    end_time           = time.time()
    knnPixelCount      = np.count_nonzero(knnMask)
    knnPixelPercentage = (knnPixelCount*100.0)/pixel_total

    print('[INFO] Processing time Movement Detection : {0:2.2f} ms'.format((end_time-start_time)*1000))
    print('[INFO] Percentage of Moving Pixel: {0:2.4f} % ({1:d})'.format(knnPixelPercentage, knnPixelCount))

    if (knnPixelCount > thresholdCount) and (frameID > 1):
        cv2.putText(frame, detectionText, titleTextPosition, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Show detections
    cv2.imshow('Original', frame)
    cv2.imshow('Movement: KNN', knnMask)

    # cv2.moveWindow('Original', 50, 50)
    # cv2.moveWindow('Movement: KNN',  frame_width, 50)

    # Record Video
    writer.write(frame) if args['save'] else 0
    writer_motion.write(knnMask) if args['save'] else 0

    # if the `q` key was pressed, break from the loop
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break


video.release()
cv2.destroyAllWindows()