import cv2
import mediapipe as mp
import mouse

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

points = [0 for i in range(21)]

while True:
    success, img = cap.read()
    # imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgRGB = cv2.flip(img, 1)
    # cv2.imshow("Image", img)
    results = hands.process(imgRGB)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(imgRGB, handLms, mpHands.HAND_CONNECTIONS)
            for id, point in enumerate(handLms.landmark):
                width, height, color = imgRGB.shape
                width, height = int(point.x * height), int(point.y * width)

                points[id] = height

                if id == 8:
                    cv2.circle(imgRGB, (width, height), 15, (255, 0, 255), cv2.FILLED)

                    distance_0_5 = abs(points[0] - points[5])
                    distance_0_8 = abs(points[0] - points[8])
                    distanceGood = distance_0_5 + (distance_0_5 / 2)

                    if distance_0_8 > distanceGood:
                        start_x, start_y = mouse.get_position()

                        res_x, res_y = abs(start_x - width * 2.4), abs(start_y - height * 2.4)

                        if res_x > 10 or res_y > 10:
                            # if mouse.is_pressed():
                            #     mouse.release()
                            mouse.move(start_x, start_y, True, 0)
                            mouse.move(width * 2.4, height * 2.4, True, 0)
                            # mouse.release()
                    # mouse.drag(x, y, width, width)

    # cv2.namedWindow("Image", cv2.WND_PROP_FULLSCREEN)
    # cv2.setWindowProperty("Image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Image", imgRGB)

    if cv2.waitKey(1) == ord('q'):
        break
