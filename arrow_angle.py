import cv2
import numpy as np
from math import atan2, cos, sin, sqrt, pi, trunc

def nothing():
    pass

def isArrow(pts):
    if(pts==7):  # arrow has 7 edge points
        isArrow = True
    else:
        isArrow = False
    return(isArrow)

def isRed(val):
    # print(val)
    if(val<22 or val>160):  # hsv value of red is in this range
        isRed = True
    else:
        isRed = False
    return(isRed)

def findAngle(pts, img):

    myatan = lambda x,y: np.pi*(1.0-0.5*(1+np.sign(x))*(1-np.sign(y**2))\
        -0.25*(2+np.sign(x))*np.sign(y))\
        -np.sign(x*y)*np.arctan((np.abs(x)-np.abs(y))/(np.abs(x)+np.abs(y)))

    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i,0] = pts[i,0,0]
        data_pts[i,1] = pts[i,0,1]
    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)

    # angle = np.arctan2(eigenvectors[0,1], eigenvectors[0,0]) #angle is returned in radians 
    # print("eigenvectors[0,1]    :   " + str(eigenvectors[0,1]))
    # print("eigenvectors[0,0]    :   " + str(eigenvectors[0,0]))

    angle = myatan(eigenvectors[0,1], eigenvectors[0,0]) #angle is returned in radians 
    angleDeg = angle * (180/np.pi)
    angleDeg = trunc((10 ** 3)*angleDeg)/(10 ** 3)

    return(angleDeg)
 

def getContours(img, imgContour):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for contour in contours:
        area=cv2.contourArea(contour)
        if(area>2000):
            points = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
            print(points)
            print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
            M = cv2.moments(contour)
            if M['m00'] != 0.0:
                x = int(M['m10']/M['m00'])
                y = int(M['m01']/M['m00'])

                hsv_frame = cv2.cvtColor(imgContour, cv2.COLOR_BGR2HSV)
                pixel_center = hsv_frame[y, x]
                hue_value = pixel_center[0]

            if(isArrow(len(points)) and isRed(hue_value)):
                # print(contour)
                cv2.drawContours(imgContour, contour, -1, (0,255,0), 3)
                angle = findAngle(contour, imgContour)
                
                cv2.putText(imgContour, (str(angle)), (x, y),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                # print(angle)


vidIn = cv2.VideoCapture(0)
# imgSample = cv2.imread('arrow_45.jpg')

cv2.namedWindow('image')
cv2.createTrackbar('valueMin','image',48,255,nothing)
cv2.createTrackbar('valueMax','image',58,255,nothing)

while(True):
    ret,img = vidIn.read()
    # img = imgSample
    imgBlur = cv2.GaussianBlur(img,(5,5),0)
    imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
    imgContour = img

    min=cv2.getTrackbarPos('valueMin','image')
    max=cv2.getTrackbarPos('valueMax','image')

    imgCanny = cv2.Canny(imgGray,min,max)

    dilKernel = np.ones((5,5))
    imgDil = cv2.dilate(imgCanny, dilKernel, iterations=1)
    
    getContours(imgDil,imgContour)

    cv2.imshow('binary', imgContour)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
