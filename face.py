#import dependencies
import cv2 as cv
#read image
img =cv.imread('imgees/h.jpg')
#redefine size of image read
def rescaleframe(frame, scale=0.75):
    width = int(frame.shape[1]*0.2)
    height = int(frame.shape[0]*0.2)
    dimensions =(width,height)
    return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)
resized_image =rescaleframe(img)

#change image to grayscale
gray = cv.cvtColor(resized_image, cv.COLOR_BGR2GRAY)
cv.imshow('gray',gray)

#link faces.py with data on images   NB: Download file from github
#https://github.com/opencv/opencv/blob/4.x/data/haarcascades/haarcascade_frontalface_default.xml
#download and save it in the same folder as his file
haar_cascade = cv.CascadeClassifier('haar_face.xml')

#place rectangle on face to show detection
faces_rect =haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=3)

#print number of faces detected
print(f'number of faces = {len(faces_rect)}')

#place rectangle on face to show detection
for (x,y,w,h) in faces_rect:
    cv.rectangle(resized_image,(x,y),(x+w,y+h),(0,255,0),thickness=3)
#show image
cv.imshow('detected image',resized_image)


cv.waitKey(0)
