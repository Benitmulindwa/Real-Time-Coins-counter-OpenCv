import cv2 as cv 
import cvzone
import numpy as np
from cvzone.ColorModule import ColorFinder

cap=cv.VideoCapture(0)

cap.set(3,640)
cap.set(4,480)


myColorFinder=ColorFinder(False)

hsvVals = {'hmin': 23, 'smin': 0, 'vmin': 195, 'hmax': 179, 'smax': 255, 'vmax': 255}



def empty(a):
    pass
cv.namedWindow("Settings")
cv.resizeWindow("Settings", 640, 240)
cv.createTrackbar("Threshold1", "Settings", 27, 255, empty)
cv.createTrackbar("Threshold2", "Settings", 120, 255, empty)


def preProcessing(img):
    
    imgPre=cv.GaussianBlur(img, (5,5), 3)
    
    thresh1=cv.getTrackbarPos("Threshold1", "Settings")
    thresh2=cv.getTrackbarPos("Threshold2", "Settings")
    
    imgPre=cv.Canny(imgPre,thresh1,thresh2)
    
    kernel=np.ones((3,3),np.uint8)
    imgPre=cv.dilate(imgPre,kernel, iterations=1)
    
    imgPre=cv.morphologyEx(imgPre,cv.MORPH_CLOSE, kernel)
    
    return imgPre



while True:
    
    _,image=cap.read()
    imgPre=preProcessing(image)
    imgContours,conFound=cvzone.findContours(image,imgPre,minArea=20)
    money=0
    imgCount=np.zeros((480,640,3),np.uint8)
    
    if conFound:
        for count,contour in enumerate(conFound):
            peri = cv.arcLength(contour["cnt"], True)
            approx = cv.approxPolyDP(contour["cnt"], 0.02 * peri, True)
            
            if len(approx)>5:
                area=contour["area"]
                x,y,w,h=contour['bbox']
                imgCrop=image[y:y+h,x:x+w]
                
                imgColor, mask = myColorFinder.update(imgCrop, hsvVals)
                whitepixel=cv.countNonZero(mask)
                #print(whitepixel)
                
                # cv.imshow(str(count),imgCrop)
                # cv.imshow("Image Color",imgColor)
                
                #print(contour["area"])
                if area<4500:
                    money+=5
                elif 4500<area<5850:
                    money+=10
                elif 5890<area<7890:
                    money+=20
                elif area>8000:
                    money+=40
                # elif whitepixel>4000:
                #     money+=1
    #print(money) 
    cvzone.putTextRect(imgCount,f'ksh.{money}', (100,300),scale=10,colorR=(0,0,255),thickness=5)   
    imageStacked=cvzone.stackImages([image,imgPre,imgContours,imgCount],2,0.5)
    cvzone.putTextRect(imageStacked,f'ksh.{money}', (50,50),colorR=(0,0,255))
    cv.imshow("Money Counter",imageStacked)
    
    key=cv.waitKey(1)
    
    if key==ord("q"):
        break

cap.release() 
cv.destroyAllWindows()
