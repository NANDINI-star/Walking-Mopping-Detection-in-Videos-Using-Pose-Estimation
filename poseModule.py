import cv2
import mediapipe as mp
import time

class poseDetector():

    def __init__(self, mode = False, modelComplex = 1, smooth = True, enablseg = False, smoothseg = True, detectionConf = 0.5, trackConf  = 0.5):

        self.mode = mode
        self.modelComplex = modelComplex
        self.smooth = smooth
        self.enablseg = enablseg
        self.smoothseg = smoothseg
        self.detectionConf = detectionConf
        self.trackConf = trackConf

        self.mpPose = mp.solutions.pose
        self.mpDraw = mp.solutions.drawing_utils
        self.pose = self.mpPose.Pose(self.mode, self.modelComplex, self.smooth, self.enablseg, self.smoothseg, self.detectionConf, self.trackConf)

    def findPose(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)  # detection

        # drawing landmarks for the detection
        if (self.results.pose_landmarks):
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return img

    def findPostion(self, img, draw=True):
        lmList = []

        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                # print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)
        return lmList


def main():
    cap = cv2.VideoCapture('input/ResoluteAI_MLE_Assignment.mp4')
    pTime = 0
    detector = poseDetector()
    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        lmList = detector.findPostion(img, draw = False)
        if len(lmList) != 0:
            print(lmList)
            cv2.circle(img, (lmList[14][1], lmList[14][2]), 15, (0, 0, 0), cv2.FILLED)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()