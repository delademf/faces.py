import cv2 as cv
img =cv.imread('imgees/h.jpg')

def rescaleframe(frame, scale=0.75):
    width = int(frame.shape[1]*0.2)
    height = int(frame.shape[0]*0.2)
    dimensions =(width,height)
    return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)
resized_image =rescaleframe(img)

gray = cv.cvtColor(resized_image, cv.COLOR_BGR2GRAY)
cv.imshow('gray',gray)

#link file with data on images 
haar_cascade = cv.CascadeClassifier('haar_face.xml')

faces_rect =haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=3)

print(f'number of faces = {len(faces_rect)}')

for (x,y,w,h) in faces_rect:
    cv.rectangle(resized_image,(x,y),(x+w,y+h),(0,255,0),thickness=3)
cv.imshow('detected image',resized_image)


cv.waitKey(0)