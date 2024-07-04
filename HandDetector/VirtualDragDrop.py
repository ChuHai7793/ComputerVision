# from cvzone.HandTrackingModule import HandDetector
from HandDetectorClass import HandDetector
import cv2 as cv


capture  = cv.VideoCapture(0)
capture.set(3, 1280) # Set width
capture.set(4, 720) # Set height
detector = HandDetector()


# Rectangle Properties
cx, cy, w , h =  100,100,200,200 # center of rect and its width,height
Rec_color = (255,0,0)



class Drag_Rect():

    def __init__(self,Center_Pos = [100,100] ,size = [200,200]):
        self.Center_Pos = Center_Pos
        self.size = size

    def update(self,cursor):
        cx, cy = self.Center_Pos
        w, h = self.size
        # If the finger tip in the rectangle region then change the center position to cursor position
        if cx - w//2 < cursor[0] < cx + w//2 and cy - h//2 < cursor[1] < cy + h//2:
            self.Center_Pos = cursor

Rec = Drag_Rect()
while True:
    success, img = capture.read()
    flip_img = cv.flip(img, 1)
    img = detector.findHands(flip_img, success,draw =True)

    lmList = detector.findPosition(flip_img) # lmList is a list contain lists of [center_x,center_y]


    # DISPLAY ID NUMBER OF POSITION IN HAND
    # detector.Display_Hand_Position(flip_img, lmList)

    if lmList:

        distance,_,_ = detector.findDistance(lmList[4],lmList[12],flip_img) # Distance between middle finger tip and thumb

        cursor = lmList[8] # This is the point finger tip id

        if distance < 50:
            Rec.update(cursor)

    # print(Rec_color)
    cx, cy = Rec.Center_Pos
    w, h = Rec.size
    cv.rectangle(flip_img, (cx - w//2, cy - h//2),
                 (cx + w//2, cy + h//2), Rec_color, thickness=2)

    cv.imshow("Video", flip_img)
    cv.waitKey(1)