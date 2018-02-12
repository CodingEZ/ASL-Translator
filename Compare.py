import cv2
import os
import numpy as np
import time

def compare(inputImg, check):
    '''Compares gray-scale images for differences'''
    # pixel score
    pixelDif = 0
    (height, width) = inputImg.shape
    for y in range(0, height, 6):
        for x in range(0, width, 6):
            if inputImg[y, x] != check[y, x]:
                pixelDif += (inputImg[y, x]/255 - check[y, x]/255) ** 2

    # contour score, have not yet incorporated
    contours, _ = cv2.findContours(inputImg, 2, 1)[-2:]
    cnt1 = contours[0]
    contours, _ = cv2.findContours(check, 2, 1)[-2:]
    cnt2 = contours[0]
    ret = cv2.matchShapes(cnt1,cnt2,1,0.0)
    
    return pixelDif**0.5 * 255 / inputImg.size + ret * 10**(-308)

def bestMatch(inputImg):
    scores = []
    checks = []
    path = os.getcwd()
    for directory in os.listdir(path):
        if not os.path.isdir(directory):
            continue
        for innerDir in os.listdir(path + '\\' + directory):
            if not os.path.isdir(directory + '\\' + innerDir):
                continue
            for file in os.listdir(path + '\\' + directory + '\\' + innerDir):
                check = cv2.imread(path + '\\' + directory + '\\' + innerDir + '\\' + file, 0)
                score = compare(inputImg, check)

                scores.append(score)
                checks.append(path + '\\' + directory + '\\' + innerDir + '\\' + file)
                
    minScore = min(scores)
    minCheck = checks[scores.index(minScore)]
    print('Minimum Score: ', minScore, 'Minimum Check: ', minCheck)
    return minCheck

def capture(cap):
    detectTop = 100
    detectBottom = 350
    
    _, img = cap.read()
    cv2.rectangle(img, (detectBottom, detectBottom), (detectTop, detectTop), (0,255,0),0)
    crop_img = img[detectTop:detectBottom, detectTop:detectBottom]

    # convert to grayscale
    grey = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    # applying gaussian blur
    blurred = cv2.GaussianBlur(grey, (35, 35), 0)
    # thresholdin: Otsu's Binarization method
    thresh1 = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]

    # check OpenCV version to avoid unpacking error
    (version, _, _) = cv2.__version__.split('.')
    if version == '3':
        image, contours, hierarchy = cv2.findContours(thresh1.copy(), \
               cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    elif version == '2':
        contours, hierarchy = cv2.findContours(thresh1.copy(),cv2.RETR_TREE, \
               cv2.CHAIN_APPROX_NONE)

    # find contour with max area
    cnt = max(contours, key = lambda x: cv2.contourArea(x))    
    # drawing contours
    drawing = np.zeros(crop_img.shape,np.uint8)
    cv2.drawContours(drawing, [cnt], 0, (0, 255, 0), 0)

    # convert thresh1 from grey to BGR
    thresh1 = cv2.cvtColor(thresh1, cv2.COLOR_GRAY2BGR)
    threshReal = np.hstack((crop_img, thresh1))
    cv2.imshow('Compare', threshReal)

    # convert thresh1 from grey to BGR
    thresh1 = cv2.cvtColor(thresh1, cv2.COLOR_BGR2GRAY)
    return thresh1

def delay(cap):    
    while (cap.isOpened()):
        capture(cap)
        k = cv2.waitKey(10)
        if k == ord('s'):
            break  

def imgGrab():
    detectTop = 100
    detectBottom = 350
    
    cap = cv2.VideoCapture(0)
    delay(cap)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    output = cv2.VideoWriter('output.avi', fourcc, 10.0, (640,480), isColor=False)
            
    start = time.time()
    duration = 1

    frame = None
    while time.time() < start + duration:
        # Capture frame-by-frame
        frame = capture(cap)
                
    cap.release()
    output.release()
    cv2.destroyAllWindows()
    return frame

def getLetter():
    inputImg = imgGrab()
    print('Grabbed image! Checking for best match ...')
    bestCheck = bestMatch(inputImg)
    print('Finished best match! Getting letter.')
    locations = bestCheck.split('\\')
    print('Detected letter:', locations[-2])

    checkImg = cv2.imread(bestCheck, 0)
    threshReal = np.hstack((inputImg, checkImg))
    cv2.imshow('Compare', threshReal)
    cv2.putText(threshReal, ':)', (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    return locations[-2]

print(getLetter())
