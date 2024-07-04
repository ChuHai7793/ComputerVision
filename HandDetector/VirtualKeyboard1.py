import cv2 as cv
from HandDetectorClass import HandDetector
from time import sleep
capture = cv.VideoCapture(0)
capture.set(3, 1280)  # Set width
capture.set(4, 720)  # Set height
keys_list = [["q", "w", "e", "r", "t", "y", "u", "i", "o", "p"],
        ["a", "s", "d", "f", "g", "h", "j", "k", "l"],
        ["z", "x", "c", "v", "b", "n", "m"]]
finalText = ""


class Button():
    def __init__(self, center_pos,text, size=[40, 40], fontScale=1):
        self.center_pos = center_pos
        self.size = size
        self.text = text
        self.fontScale = fontScale

def Draw_Keyboards(img,buttons_list,tune=[-5, 5]):
    for button in  buttons_list:
        cx, cy = button.center_pos
        w, h = button.size
        cv.rectangle(img, (cx - w // 2, cy - h // 2), (cx + w // 2, cy + h // 2), (255, 0, 255), thickness=cv.FILLED)
        cv.putText(img, button.text, (cx + tune[0], cy + tune[1]), cv.FONT_HERSHEY_PLAIN,
                   button.fontScale, (255, 255, 255), 2)

    return img


buttons_list = []
initial_cy = 100
button_w, button_h = [60, 60]
for row in keys_list:
    initial_cx = 100
    for key in row:
        buttons_list.append(Button([initial_cx, initial_cy], key,size=[button_w, button_h]))
        initial_cx = initial_cx + button_w + 20
    initial_cy = initial_cy + button_h + 20

while True:
    detector = HandDetector(detectionConfidence = 0.8)
    success, img = capture.read()
    flip_img = cv.flip(img, 1)
    flip_img = detector.findHands(flip_img, success)
    lmList = detector.findPosition(flip_img)  # lmList is a list contain lists of [center_x,center_y]
    flip_img = Draw_Keyboards(flip_img, buttons_list)

    if lmList:
        cursor = lmList[8]
        for button in buttons_list:
            cx, cy = button.center_pos
            w, h = button.size
            if cx - w // 2 < cursor[0] < cx + w // 2 and cy - h // 2 < cursor[1] < cy + h // 2:
                cv.rectangle(flip_img, (cx - w // 2, cy - h // 2), (cx + w // 2, cy + h // 2), (100, 0, 100), thickness=cv.FILLED)
                cv.putText(flip_img, button.text, (cx - 5, cy +5), cv.FONT_HERSHEY_PLAIN,
                           button.fontScale, (255, 255, 255), 2)

                distance, _, _ = detector.findDistance(lmList[8], lmList[12],flip_img)  # Distance between middle finger tip and thumb

                if distance < 50:
                    cv.rectangle(flip_img, (cx - w // 2, cy - h // 2), (cx + w // 2, cy + h // 2), (0, 0, 255),
                                 thickness=cv.FILLED)
                    cv.putText(flip_img, button.text, (cx - 5, cy + 5), cv.FONT_HERSHEY_PLAIN,
                               button.fontScale, (255, 255, 255), 2)
                    finalText += button.text

                    sleep(0.5)
    print(finalText)
    cv.rectangle(flip_img, (50, 350), (700, 450), (175, 0, 175), cv.FILLED)
    cv.putText(flip_img, finalText, (60, 430),
                cv.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 5)

    cv.imshow("FrontCam", flip_img)

    cv.waitKey(1)
