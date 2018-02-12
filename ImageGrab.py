import cv2
import numpy
import time

def equalArray(array1, array2):
    if len(array1) != len(array2):
        return False
    else:
        for index in range(len(array1)):
            if array1[index] != array2[index]:
                return False
        return True

def grab():
    start = time.time()
    duration = 5

    #cap = cv2.VideoCapture(1)      # for non-built-in camera
    cap = cv2.VideoCapture(0)      # for built-in camera

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    output = cv2.VideoWriter('output.avi', fourcc, 10.0, (640,480), isColor=False)

    frame = None
    detectTop = 100
    detectBottom = 350
    while time.time() < start + duration:
        # Capture frame-by-frame
        ret, frame = cap.read()
        frame = frame[detectTop:detectBottom, detectTop:detectBottom]
        cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                
    # When everything done, release the capture
    cap.release()
    output.release()
    cv2.destroyAllWindows()
    return frame
