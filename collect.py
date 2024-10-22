from app import normaliseGestureLandmarks
from app import normaliseHistoryLandmarks

import mediapipe
import cv2
import os
import numpy

from collections import deque


TOTALDATAPOINTS = 5000
HISTORYLENGTH = 16
ISHISTORYMODEL = True

LABELS = ["still", "clockwise", "anticlockwise", "move"]

# LABELS = [
#     "thumbsup",
#     "thumbsdown",
#     "pointing",
#     "victory",
#     "fist",
#     "palm",
#     "ILoveU",
#     "ok",
# ]

sampleCount = 0
labelCount = 0

fingerTipHistory = {"l": deque(maxlen=HISTORYLENGTH), "r": deque(maxlen=HISTORYLENGTH)}

STARTSAVING = False
DATASET = []

drawing = mediapipe.solutions.drawing_utils
styles = mediapipe.solutions.drawing_styles

hands = mediapipe.solutions.hands

SHAPE = (1280, 720)

cap = cv2.VideoCapture(0)

cap.set(3, SHAPE[0])
cap.set(4, SHAPE[1])

with hands.Hands(
    min_detection_confidence=0.4, min_tracking_confidence=0.4, max_num_hands=2
) as detector:
    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            continue

        cv2.flip(frame, 1, frame)

        frame.flags.writeable = False

        results = detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        frame.flags.writeable = True

        if len(LABELS) <= labelCount:
            STARTSAVING = False

        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    hands.HAND_CONNECTIONS,
                    styles.DrawingSpec((0, 0, 255), -1, 6),
                    styles.DrawingSpec((0, 255, 0), 3, -1),
                )

                handLabel = handedness.classification[0].label.lower()[0]

                landmarks = numpy.array(
                    [
                        [landmark.x * frame.shape[1], landmark.y * frame.shape[0]]
                        for landmark in hand_landmarks.landmark
                    ]
                )

                if ISHISTORYMODEL:
                    fingerTipHistory[handLabel].append(landmarks[8])

                if STARTSAVING:
                    if ISHISTORYMODEL:
                        normalisedHistoryLandmarks = normaliseHistoryLandmarks(
                            numpy.float64(fingerTipHistory[handLabel]), frame
                        )

                        if len(fingerTipHistory[handLabel]) == HISTORYLENGTH:
                            DATASET.append(normalisedHistoryLandmarks)

                            sampleCount += 1

                            if sampleCount >= TOTALDATAPOINTS:
                                numpy.save(
                                    f"./utils/history/samples/{LABELS[labelCount]}",
                                    DATASET,
                                )
                                STARTSAVING = False

                                labelCount += 1
                                sampleCount = 0

                                DATASET.clear()

                                for key in fingerTipHistory.keys():
                                    fingerTipHistory[key].clear()

                    else:
                        normalisedGestureLandmarks = normaliseGestureLandmarks(
                            landmarks
                        )

                        DATASET.append(normalisedGestureLandmarks)

                        sampleCount += 1

                        if sampleCount >= TOTALDATAPOINTS:
                            numpy.save(
                                f"./utils/gesture/samples/{LABELS[labelCount]}", DATASET
                            )
                            STARTSAVING = False

                            labelCount += 1
                            sampleCount = 0

                            DATASET.clear()

        elif ISHISTORYMODEL:
            for key in fingerTipHistory.keys():
                fingerTipHistory[key].append([0, 0])

        if ISHISTORYMODEL:
            for key in fingerTipHistory.keys():
                for index, point in enumerate(fingerTipHistory[key]):
                    if int(point[0]) != 0 and int(point[1]) != 0:
                        cv2.circle(
                            frame,
                            list(map(int, point)),
                            1 + int(index / 2),
                            (157, 255, 157),
                            2,
                        )

        width = max(
            cv2.getTextSize(
                LABELS[min(labelCount, len(LABELS) - 1)],
                cv2.FONT_HERSHEY_COMPLEX,
                0.9,
                2,
            )[0][0],
            cv2.getTextSize(
                str(sampleCount).zfill(len(str(TOTALDATAPOINTS))),
                cv2.FONT_HERSHEY_COMPLEX,
                0.9,
                2,
            )[0][0],
        )

        cv2.putText(
            frame, "Label:", (50, 65), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2
        )
        cv2.putText(
            frame,
            LABELS[min(labelCount, len(LABELS) - 1)],
            (160, 65),
            cv2.FONT_HERSHEY_COMPLEX,
            0.9,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame, "Count:", (50, 105), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 0, 255), 2
        )
        cv2.putText(
            frame,
            str(sampleCount if labelCount != len(LABELS) else TOTALDATAPOINTS).zfill(
                len(str(TOTALDATAPOINTS))
            ),
            (160, 105),
            cv2.FONT_HERSHEY_COMPLEX,
            0.9,
            (0, 255, 0),
            2,
        )

        cv2.rectangle(frame, (30, 20), (180 + width, 130), (255, 0, 0), 3)

        cv2.imshow("Data Collection", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            cap.release()

        elif key == ord("s"):
            STARTSAVING = not STARTSAVING

cv2.destroyAllWindows()
