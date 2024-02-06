import cv2
import numpy as np
import cvzone
from ultralytics import YOLO
########################
counter=0

########################
# load the model
model=YOLO("yolov8n.pt")

cap=cv2.VideoCapture("care.mp4")

def pre(result):
    for i in result:
        boxs=i.boxes
        for box in boxs:
            x,y,w,z=box.xyxy[0]
            x, y, w, z=int(x),int(y),int(w),int(z)

            return x,y,w,z




while (True):

    # frame1
    _,image1=cap.read()
    # frame2
    _,image2 = cap.read()
    # define the line on the road
    cv2.line(image2,(100,270),(450,270),(255,255,0),2)
    # the result of frame 1
    result1=model(image1,stream=True)
    # the result of frame 2
    result2 = model(image2, stream=True)
    # define the rectangle frame 1
    x1, y1, w1, z1=pre(result1)
    h1 = z1 - y1
    # define the rectangle frame 2
    x2, y2, w2, z2 = pre(result2)
    w2 = w2 - x2
    h2= z2 - y2

    #text on the frame2
    cv2.putText(image2, f"count {counter}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

    # x  of the care increase and y decrease
    if (x2-x1)>0 and (y2-y1)<0 and 0>h2-h1>-12:

        # save the image
        new_image = image2[y2:y2 + h2, x2:w2 + x2]
        cv2.imwrite(r"D:\\projects\under.jpg", new_image)

        # dfine the center of rectangle
        cx=x2+w2//2
        cy=y2+h2//2
        # if the center nearest the line
        if 240 >y2>230 :
            counter+=1

        cv2.circle(image2,(cx,cy),10,(255,0,255),-1)
        #draw the rectangle
        cvzone.cornerRect(image2, (x2, y2, w2, h2))

    cv2.imshow("image",image2)

    cv2.waitKey(1)
cap.release()
cv2.destroyAllWindows()

