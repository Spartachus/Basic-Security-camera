import numpy as np
import cv2
import time
import datetime

vid = cv2.VideoCapture(0)


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")

detection = False
detection_stopped_time = None
timer_started = False
Seconds_to_record_after_detection = 2

frame_size = (int(vid.get(3)), int(vid.get(4)))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

while True:
    ret, frame = vid.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#graying images
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    body = body_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) + len(body) > 0:
        if detection:
            timer_started = False
        else:
            detection = True
            current_time = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
            output = cv2.VideoWriter(f"{current_time}.mp4", fourcc, 60, frame_size)
            print("°Recording°")
    elif detection:
        if timer_started:
            if time.time() - detection_stopped_time >= Seconds_to_record_after_detection:
                detection = False
                timer_started = False
                output.release()
                print("°Stopped°")
        else:
            timer_started =True
            detection_stopped_time = time.time()

    if detection:
        output.write(frame)

    cv2.imshow("frame", frame)

    if cv2.waitKey(1) == ord("q"):
        break

vid.release()
cv2.destroyAllWindows()