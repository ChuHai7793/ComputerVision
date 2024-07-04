import cv2 as cv
import mediapipe as mp
import time


class HandDetector():
    # Arguments of mp.solutions.hands.Hands() in def __init__
    def __init__(self,mode = False, maxHands = 2, detectionConfidence = 0.5 ,trackConfidence = 0.5):

        self.mode = mode
        self.maxHands = maxHands
        self.detectionConfidence = detectionConfidence
        self.trackConfidence = trackConfidence
        self.modelComplexity = 1

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode,self.maxHands,self.modelComplexity,self.detectionConfidence,self.trackConfidence)
        self.mpDraw = mp.solutions.drawing_utils
        self.results = None

    def findHands(self,img,success,draw =True):
        if success == False:
            print("Out of frame!")
            return

        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # HandTracking module only use RGB image
        self.results = self.hands.process(imgRGB)
        print(self.results.multi_hand_landmarks)

        if draw == True:
            if self.results.multi_hand_landmarks:
                for handLMS in self.results.multi_hand_landmarks:
                    self.mpDraw.draw_landmarks(img, handLMS, self.mpHands.HAND_CONNECTIONS)
        return img


    def findPosition(self,img, Hand_Number = 0):
        # Hand_Number = 1: Both Hands.
        # Hand_Number = 0: The latest detected Hand

        lmList =[]
        if Hand_Number == 2:
            if self.results.multi_hand_landmarks:
                for handLMS in self.results.multi_hand_landmarks:
                    for id, lm in enumerate(handLMS.landmark):
                        print(id, lm)

                        height, width, channel = img.shape
                        center_x, center_y = int(lm.x * width), int(lm.y * height)
                        print(id, center_x, center_y)
                        lmList.append([id, center_x, center_y])
                        if id == 0:
                            cv.circle(img, (center_x, center_y), 25, (255, 0, 255), cv.FILLED)
        else:
            if self.results.multi_hand_landmarks:
                Chosen_Hand = self.results.multi_hand_landmarks[Hand_Number]
                for id, lm in enumerate(Chosen_Hand.landmark):
                    print(id, lm)

                    height, width, channel = img.shape
                    center_x, center_y = int(lm.x * width), int(lm.y * height)
                    print(id, center_x, center_y)
                    lmList.append([id, center_x, center_y])
                    if id == 0:
                        cv.circle(img, (center_x, center_y), 25, (255, 0, 255), cv.FILLED)
        return lmList



def main():
    detector = HandDetector()
    cap = cv.VideoCapture(0)
    while True:
        success,img = cap.read()
        img = detector.findHands(img,success)
        lmList = detector.findPosition(img,Hand_Number = 0)
        flip_img = cv.flip(img, 1)

        cv.imshow("Webcam", flip_img)

        #  exit when press "q"
        if cv.waitKey(20) & 0xFF == ord('q'):
            print(cv.waitKey(0))
            break
    cap.release()
    cv.destroyAllWindows()
if __name__ == '__main__':
    main()