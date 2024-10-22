from utils import GestureClassifier
from utils import HistoryClassifier

import mediapipe
import cv2
import numpy

from collections import deque
from collections import Counter


def normaliseGestureLandmarks(landmarks: numpy.ndarray) -> numpy.ndarray:
    """normalise landmarks

    Args:
        landmarks (numpy.ndarray): landmarks array

    Returns:
        numpy.ndarray: normalised landmarks array
    """
    landmarksCopy = landmarks.copy()

    landmarksCopy = (
        landmarksCopy - numpy.array([landmarks[0]] * len(landmarks))
    ).flatten()

    return landmarksCopy / abs(landmarksCopy).max()


def normaliseHistoryLandmarks(
    landmarks: numpy.ndarray, frame: numpy.ndarray
) -> numpy.ndarray:
    """normalise landmarks

    Args:
        landmarks (numpy.ndarray): landmarks array
        frame (numpy.ndarray): camera frame

    Returns:
        numpy.ndarray: normalised landmarks array
    """
    landmarksCopy = landmarks.copy()

    landmarksCopy = (
        landmarksCopy - numpy.array([landmarks[0]] * len(landmarks))
    ) / numpy.array([frame.shape[-2::-1]] * len(landmarks))

    return landmarksCopy.flatten()


if __name__ == "__main__":
    gestureModel = GestureClassifier()
    historyModel = HistoryClassifier()

    with open(r"./utils/gesture/models/sample/labels.txt") as file:
        GESTURELABELS = file.read().split("\n")

    with open(r"./utils/history/models/sample/labels.txt") as file:
        HISTORYLABELS = file.read().split("\n")

    HISTORYLENGTH = 16

    fingerHistory = {"l": deque(maxlen=HISTORYLENGTH), "r": deque(maxlen=HISTORYLENGTH)}
    resultHistory = {"l": deque(maxlen=HISTORYLENGTH), "r": deque(maxlen=HISTORYLENGTH)}

    drawing = mediapipe.solutions.drawing_utils
    styles = mediapipe.solutions.drawing_styles

    hands = mediapipe.solutions.hands

    PADDING = 20
    BOXCOLOR = (255, 255, 0)
    LENGTH = 50

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

                    normalisedGestureLandmarks = normaliseGestureLandmarks(landmarks)
                    normalisedHistoryLandmarks = normaliseHistoryLandmarks(
                        numpy.float64(fingerHistory[handLabel]), frame
                    )

                    gestureIndex = gestureModel(normalisedGestureLandmarks)

                    fingerHistory[handLabel].append(
                        list(landmarks[8]) if gestureIndex == 2 else [0, 0]
                    )

                    if (
                        len(fingerHistory[handLabel]) == HISTORYLENGTH
                        and gestureIndex == 2
                    ):
                        resultHistory[handLabel].append(
                            historyModel(normalisedHistoryLandmarks)
                        )

                        historyIndex = Counter(resultHistory[handLabel]).most_common()[
                            0
                        ][0]

                    x1, y1 = (
                        int(landmarks[:, 0].min()) - PADDING,
                        int(landmarks[:, 1].min()) - PADDING,
                    )
                    x2, y2 = (
                        int(landmarks[:, 0].max()) + PADDING,
                        int(landmarks[:, 1].max()) + PADDING,
                    )

                    cv2.line(frame, (x2 - LENGTH, y2), (x2, y2), BOXCOLOR, 3)
                    cv2.line(frame, (x2, y2), (x2, y2 - LENGTH), BOXCOLOR, 3)

                    cv2.line(frame, (x1, y2), (x1 + LENGTH, y2), BOXCOLOR, 3)
                    cv2.line(frame, (x1, y2), (x1, y2 - LENGTH), BOXCOLOR, 3)

                    if gestureIndex != 2 or (gestureIndex == 2 and handLabel == "r"):
                        cv2.line(frame, (x2 - LENGTH, y1), (x2, y1), BOXCOLOR, 3)
                        cv2.line(frame, (x2, y1), (x2, y1 + LENGTH), BOXCOLOR, 3)

                    if gestureIndex != 2 or (gestureIndex == 2 and handLabel == "l"):
                        cv2.line(frame, (x1 + LENGTH, y1), (x1, y1), BOXCOLOR, 3)
                        cv2.line(frame, (x1, y1), (x1, y1 + LENGTH), BOXCOLOR, 3)

                    (width, height), baseline = cv2.getTextSize(
                        (
                            GESTURELABELS[gestureIndex]
                            if gestureIndex != 2
                            else (
                                HISTORYLABELS[historyIndex]
                                + " "
                                + (
                                    GESTURELABELS[gestureIndex]
                                    if historyIndex == 0
                                    else ""
                                )
                            )
                        ),
                        cv2.FONT_HERSHEY_COMPLEX,
                        1,
                        2,
                    )

                    cv2.rectangle(
                        frame,
                        (x1, y1 - (height + baseline) - 20),
                        (x1 + width + 60, y1 - 10),
                        BOXCOLOR,
                        -1,
                    )

                    cv2.putText(
                        frame,
                        (
                            GESTURELABELS[gestureIndex]
                            if gestureIndex != 2
                            else (
                                HISTORYLABELS[historyIndex]
                                + " "
                                + (
                                    GESTURELABELS[gestureIndex]
                                    if historyIndex == 0
                                    else ""
                                )
                            )
                        ),
                        (x1 + 30, y1 - baseline // 2 - 15),
                        cv2.FONT_HERSHEY_COMPLEX,
                        1,
                        (255, 255, 255),
                        2,
                    )

            else:
                for key in fingerHistory.keys():
                    fingerHistory[key].append([0, 0])

            for key in fingerHistory.keys():
                for index, point in enumerate(fingerHistory[key]):
                    if int(point[0]) != 0 and int(point[1]) != 0:
                        cv2.circle(
                            frame,
                            list(map(int, point)),
                            1 + int(index / 2),
                            (157, 255, 157),
                            2,
                        )

            cv2.imshow("Hand Gesture Classification", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                cap.release()

    cv2.destroyAllWindows()
