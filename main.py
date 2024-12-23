from face_database import FaceDatabase
from face_recognition import FaceRecognitionSystem
from activity_recognition import ActivityRecognition
from yolo_inference import YOLOInference
from video_processor import VideoProcessor

if __name__ == "__main__":
    database = FaceDatabase('Database')
    face_recognition_system = FaceRecognitionSystem(database)
    activity_recognition = ActivityRecognition(["working in office", "sleeping on desk", "eating a snack", "using their phone"])
    yolo_inference = YOLOInference('yolov8s.pt')

    processor = VideoProcessor(yolo_inference, face_recognition_system, activity_recognition)
    processor.process_video()
