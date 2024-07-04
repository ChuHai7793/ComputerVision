import cv2 as cv
import mediapipe as mp
import time
import math


class HandDetector():
    # Arguments of mp.solutions.hands.Hands() in def __init__
    def __init__(self,mode = False, maxHands = 2, detectionConfidence = 0.5 ,trackConfidence = 0.5):
        # Note: If FrontCamera == True. Must flip image beforehand. Except when using findHands
        self.mode = mode
        self.maxHands = maxHands
        self.detectionConfidence = detectionConfidence
        self.trackConfidence = trackConfidence

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode,self.maxHands,self.detectionConfidence,self.trackConfidence)
        self.mpDraw = mp.solutions.drawing_utils
        self.results = None

    def findHands(self,img,success,draw =True):
        if success == False:
            print("Out of frame!")
            return

        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # HandTracking module only use RGB image
        self.results = self.hands.process(imgRGB)
        # print(self.results.multi_hand_landmarks)

        if draw == True:
            if self.results.multi_hand_landmarks:
                for handLMS in self.results.multi_hand_landmarks:
                    self.mpDraw.draw_landmarks(img, handLMS, self.mpHands.HAND_CONNECTIONS)
        return img


    def findDistance(self,p1, p2, img=None,draw =None):
        """
        Find the distance between two landmarks based on their
        index numbers.
        :param p1: Point1
        :param p2: Point2
        :param img: Image to draw on.
        :param draw: Flag to draw the output on the image.
        :return: Distance between the points
                 Image with output drawn
                 Line information
        """

        x1, y1 = p1
        x2, y2 = p2
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        length = math.hypot(x2 - x1, y2 - y1)
        info = (x1, y1, x2, y2, cx, cy)
        if img is not None:
            if draw == None:
                return length,info, img
            else:
                cv.circle(img, (x1, y1), 15, (255, 0, 255), cv.FILLED)
                cv.circle(img, (x2, y2), 15, (255, 0, 255), cv.FILLED)
                cv.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                cv.circle(img, (cx, cy), 15, (255, 0, 255), cv.FILLED)
                return length,info, img
        else:
            return length, info

    def findPosition(self,img, Hand_Number = 0,id_highlight = 8):
        # Hand_Number = 1: Both Hands.
        # Hand_Number = 0: The latest detected Hand

        lmList =[]
        if Hand_Number == 1:
            if self.results.multi_hand_landmarks:
                for handLMS in self.results.multi_hand_landmarks:
                    for id, lm in enumerate(handLMS.landmark):
                        # print(id, lm)
                        height, width, channel = img.shape
                        center_x, center_y = int(lm.x * width), int(lm.y * height)

                        # print(id, center_x, center_y)
                        lmList.append([center_x, center_y])

                        if id == id_highlight:
                            cv.circle(img, (center_x, center_y), 10, (255, 0, 255), cv.FILLED)
        else:
            if self.results.multi_hand_landmarks:
                Chosen_Hand = self.results.multi_hand_landmarks[Hand_Number]
                for id, lm in enumerate(Chosen_Hand.landmark):
                    # print(id, lm)

                    height, width, channel = img.shape
                    center_x, center_y = int(lm.x * width), int(lm.y * height)

                    # print(id, center_x, center_y)
                    lmList.append([ center_x, center_y])
                    if id == id_highlight:
                        cv.circle(img, (center_x, center_y), 10, (255, 0, 255), cv.FILLED)
        return lmList

    def Display_Hand_Position(self,img,lmList):
        for id,(center_x, center_y) in enumerate(lmList):
            # if self.FrontCamera == True:
            # # Notice: img must be flipped first: img = cv.flip(img,1)
            #     cv.putText(img, str(int(id)), ((img.shape[1] - center_x), center_y), cv.FONT_HERSHEY_PLAIN,
            #                 1, (255, 0, 0), 1)
            # else:
            cv.putText(img, str(int(id)), (center_x, center_y), cv.FONT_HERSHEY_PLAIN,
                       1, (255, 0, 0), 1)

def main():
    detector = HandDetector()
    cap = cv.VideoCapture(0)
    while True:
        success,img = cap.read()
        flip_img = cv.flip(img, 1)
        flip_img = detector.findHands(flip_img,success)

        lmList = detector.findPosition(flip_img,Hand_Number = 0)
        print(lmList)

        detector.Display_Hand_Position(flip_img, lmList)
        cv.imshow("Webcam", flip_img)

        #  exit when press "q"
        if cv.waitKey(20) & 0xFF == ord('q'):
            print(cv.waitKey(0))
            break
    cap.release()
    cv.destroyAllWindows()
if __name__ == '__main__':
    main()