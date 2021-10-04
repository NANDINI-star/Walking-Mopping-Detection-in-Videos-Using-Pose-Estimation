import cv2
import numpy as np
import time
import poseModule as pm
from collections import deque
import pafy

url = input("Enter URL : ")
video = pafy.new(url)
best = video.getbest(preftype="mp4")

cap = cv2.VideoCapture()
cap.open(best.url)

detector = pm.poseDetector()
count = 0
dir = 0
Q = deque(maxlen=128)
Q_leg_results = deque(maxlen=128)

writer = None

while True:
    success, img = cap.read()
    # img = cv2.resize(img, (1280, 720))
    # img = cv2.imread("input/img1.PNG")
    img = detector.findPose(img, False)
    lmList = detector.findPostion(img, False)
    # print(lmList)
    if(len(lmList) != 0):
        h, w, c = img.shape
        #Right Arm
        angle_r = detector.findAngle(img, 12, 14,16)
        percent_r = np.interp(angle_r, (60, 0), (0, 100))

        #Right leg
        angle_rl = detector.findAngle(img, 24, 26, 28)
        percent_rl = np.interp(angle_rl, (170, 230), (0, 100))

        # Left Arm
        angle_l = detector.findAngle(img, 11, 13, 15)
        percent_l = np.interp(angle_l, (190, 240), (0,100))

        #Left leg
        angle_ll = detector.findAngle(img, 23, 25, 27)
        percent_ll = np.interp(angle_ll, (170, 230), (0, 100))



        max_per = max(percent_l, percent_r)
        Q.append(max_per)
        results = int(np.array(Q).mean(axis=0))

        max_leg_per = max(percent_ll, percent_rl)
        Q_leg_results.append(max_leg_per)
        leg_results = int(np.array(Q_leg_results).mean(axis=0))
        print(results, leg_results)

        min_per = min(percent_l, percent_r)

        if percent_l > 80 and percent_r > 80:
            if(dir == 0):
                count+=0.5
                dir = 1
        if percent_l == 0 or percent_r == 0:
            if(dir == 1):
                count-=0.5
                dir = 0


        #DRAW BAR
        bar_results = np.interp(results, (0, 100), (650, 100))
        leg_bar_results = np.interp(leg_results, (0, 100), (650, 100))
        cv2.rectangle(img, (1100,100), (1175,650), (0,255,0),3)
        print("count = ", count)

        if(results > 65 and leg_results < 30): #>=
            cv2.putText(img, "Mopping", (1000, 40), cv2.FONT_HERSHEY_PLAIN, 3,
                        (0, 255, 0), 4)
            cv2.rectangle(img, (1100, int(bar_results)), (1175, 650), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, f'{results} %', (1050, 85), cv2.FONT_HERSHEY_PLAIN, 3,
                        (0, 255, 0), 4)
        elif(leg_results >45):
            count = 1
            cv2.putText(img, "Walking", (1000, 40), cv2.FONT_HERSHEY_PLAIN, 3,
                        (255, 0, 0), 4)
            cv2.rectangle(img, (1100, int(leg_bar_results)), (1175, 650), (255, 0, 0), cv2.FILLED)

            cv2.putText(img, f'{leg_results} %', (1050, 85), cv2.FONT_HERSHEY_PLAIN, 3,
                        (255, 0, 0), 4)
        elif(results < 50 and leg_results < 30):
            count = 1
            cv2.putText(img, "Standing", (1000, 40), cv2.FONT_HERSHEY_PLAIN, 3,
                        (255, 0, 0), 4)
            cv2.rectangle(img, (1100, int(leg_bar_results)), (1175, 650), (255, 0, 0), cv2.FILLED)

            cv2.putText(img, f'{leg_results} %', (1050, 85), cv2.FONT_HERSHEY_PLAIN, 3,
                        (255, 0, 0), 4)


        # else:
        #     cv2.putText(img, "Analysing...", (1000, 40), cv2.FONT_HERSHEY_PLAIN, 3,
        #                 (0, 0, 255), 4)

    # check if the video writer is None
    if writer is None:
        # initialize our video writer
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter("output/v_output.avi", fourcc, 30, (w, h), True)
    # write the output frame to disk
    writer.write(img)
    cv2.imshow("Image", img)
    cv2.waitKey(1)