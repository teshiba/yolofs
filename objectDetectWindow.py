
#####################################################################
import cv2
import Yolov8Lib as yolo
import WindowCapture as cap
import sys

#####################################################################
# settings
#####################################################################
poseMode = False

# set yolo model data file path.
modelFile = "yolov8x-pose.pt"
modelFile = "yolov8n.pt"
modelFile = "yolov8x.pt"

#####################################################################
# arguments
#####################################################################
args = sys.argv

if len(sys.argv) == 1:
    print("[WindowCapture] Input a part of the window title you want to search.")
    print("[WindowCapture] Argument error: Exit.")
    sys.exit(1)

#####################################################################
# init vars
#####################################################################
lookupWindowTitle = args[1]
print("[WindowCapture] lookup window: " + lookupWindowTitle)

Yolov8Lib = yolo.Yolov8Lib(modelFile, poseMode)
capture = cap.WindowCapture(lookupWindowTitle)

#####################################################################
# main
#####################################################################
while True:
    try:
        screenshot = capture.GetImage()
    except AttributeError:
        print("[captureApp]cannot lookup window...")
        Yolov8Lib = yolo.Yolov8Lib(modelFile, poseMode)
        capture = cap.WindowCapture(lookupWindowTitle)

    img = Yolov8Lib.GenerateAnnotatedImage(screenshot)

    cv2.imshow("test",img)

    cv2.waitKey(1)
#####################################################################
