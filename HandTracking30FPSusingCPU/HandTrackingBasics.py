import cv2 as cv
import mediapipe as mp
import time





mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils


def main():
    ptime = 0
    ctime = 0
    cap = cv.VideoCapture(0)
    while True:
        success,img = cap.read()

        if success == False:
            print("Out of frame!")
            break

        imgRGB = cv.cvtColor(img,cv.COLOR_BGR2RGB) # HandTracking module only use RGB image
        results = hands.process(imgRGB)
        print(results.multi_hand_landmarks)


        if results.multi_hand_landmarks:
            for handLMS in results.multi_hand_landmarks:
                for id, lm in enumerate(handLMS.landmark):
                    print(id,lm)

                    height, width ,channel = img.shape
                    center_x, center_y = int(lm.x*width), int(lm.y*height)
                    print(id,center_x,center_y)
                    if id == 0:
                        cv.circle(img,(center_x,center_y),25,(255,0,255),cv.FILLED)
                mpDraw.draw_landmarks(img,handLMS,mpHands.HAND_CONNECTIONS)

        ctime = time.time()
        fps = 1/(ctime-ptime)
        ptime = ctime

        flip_img = cv.flip(img,1)
        cv.putText(flip_img, str(int(fps)), (10, 30), cv.FONT_HERSHEY_PLAIN,
                   1, (255, 0, 255), 3)
        cv.imshow("Webcam",flip_img)
        cv.waitKey(1)

if __name__ == '__main__':
    main()


