import cv2 as cv
from HandDetectorClass import HandDetector

capture = cv.VideoCapture(0)
capture.set(3, 1280) # Set width
capture.set(4, 720) # Set height
keys = [["q","w","e","r","t","y","u","i","o","p"],
        ["a","s","d","f","g","h","j","k","l"],
        ["z","x","c","v","b","n","m"]]


class Buttons():
    def __init__(self, img, center_pos = [200,200],size = [40,40],cursor=None,fontScale = 1):
        self.center_pos = center_pos
        self.size = size
        self.fontScale = fontScale
        self.cursor = cursor
        self.img = img
    def draw_key(self,img,cx,cy,w,h, button_name = None,tune = [-5,5] ,thickness = cv.FILLED):

        cv.rectangle(img, (cx-w//2,cy-h//2), (cx+w//2,cy+h//2), (0,0,255),thickness = thickness)
        cv.putText(img, button_name,(cx+tune[0],cy+tune[1]),cv.FONT_HERSHEY_PLAIN,
                   self.fontScale,(255,255,255),2)

    def hover_check(self, cx,cy,w,h,cursor = None):
        if cursor != None:
            # If the finger tip in the rectangle region then change the center position to cursor position
            if cx - w // 2 < cursor[0] < cx + w // 2 and cy - h // 2 < cursor[1] < cy + h // 2:
                return True
        else:
            return False

    def keyboard_create(self,img,keys=None, key_distance = 10):
        w, h = self.size
        cy = self.center_pos[1]
        for row in keys:
            cx = self.center_pos[0]
            for key in row:
                if self.hover_check(cx,cy,w,h,self.cursor) == True:
                    self.draw_key(self.img, cx, cy, w, h, key)
                else:
                    self.draw_key(self.img, cx, cy, w, h, key,thickness= 2)
                cx = cx + w + key_distance
            cy = cy + h + key_distance




while True:
    detector = HandDetector()
    success, img = capture.read()
    flip_img = cv.flip(img, 1)
    img = detector.findHands(flip_img,success)



    lmList = detector.findPosition(flip_img) # lmList is a list contain lists of [center_x,center_y]

    if lmList:
        distance,_,_ = detector.findDistance(lmList[4],lmList[12],flip_img) # Distance between middle finger tip and thumb

        cursor = lmList[8] # This is the point finger tip id

        Buttons(flip_img,[20, flip_img.shape[0] // 2], [70, 70],cursor, 2).keyboard_create(flip_img, keys)
    else:
        Buttons(flip_img, [20, flip_img.shape[0] // 2], [70, 70],None, 2).keyboard_create(flip_img, keys)

    cv.imshow("FrontCam",flip_img)
    
    cv.waitKey(1)
