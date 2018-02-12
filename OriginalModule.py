import cv2
import numpy as np
import math
import string
import os.path

cap = cv2.VideoCapture(0)
detectTop = 100
detectBottom = 350

while(cap.isOpened()):
    # read image
    ret, img = cap.read()

    # get hand data from the rectangle sub window on the screen
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
    ##cv2.imshow('Crop', thresh1)
    threshReal = np.hstack((crop_img, thresh1))
    cv2.imshow('Compare', threshReal)

    '''
    # show appropriate images in windows
    cv2.imshow('Gesture', img)
    all_img = np.hstack((drawing, crop_img))
    cv2.imshow('Contours', all_img)
    '''

    k = cv2.waitKey(10)
    orders = [ord(i) for i in string.ascii_lowercase]
    if k == 27:
        break
    elif k == ord(' '):
        i = 1
        while os.path.exists('Letter Images\\' + str(i) + '.jpg'):
            i += 1
        cv2.imwrite('Letter Images\\' + str(i) + '.jpg', thresh1)
    elif k in orders:
        i = 1
        while os.path.exists('Letter Images\\' + chr(k-32) + '\\' + str(i) + '.jpg'):
            i += 1
            print(i)
        cv2.imwrite('Letter Images\\' + chr(k-32) + '\\' + str(i) + '.jpg', thresh1)
        
cv2.destroyAllWindows()
cap.release()
