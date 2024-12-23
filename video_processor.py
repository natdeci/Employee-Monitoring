import cv2
import time


class VideoProcessor:
    def __init__(self, yolo, face_recognition, activity_recognition):
        self.yolo = yolo
        self.face_recognition = face_recognition
        self.activity_recognition = activity_recognition
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise ValueError("Error: Could not open webcam.")

    def process_video(self):
        prev_time = time.time()
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Failed to read frame.")
                break

            detections = self.yolo.detect_people(frame)
            for x1, y1, x2, y2 in detections:
                person_image = frame[y1:y2, x1:x2]

                bbox, face_label = self.face_recognition.recognize(person_image)
                probable_activity = self.activity_recognition.recognize(person_image)

                if bbox is not None:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, face_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                cv2.putText(frame, probable_activity, (x1, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            fps = 1 / (time.time() - prev_time)
            prev_time = time.time()

            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow("Real-Time Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()
