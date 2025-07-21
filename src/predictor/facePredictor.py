from mtcnn import MTCNN
from src.insightface.deploy import face_model
from src.detectfaces_mtcnn.Configurations import ConfigurationsPOJO
import warnings
import sys
import dlib
import pickle
import numpy as np
import cv2
from keras.models import load_model
import face_preprocess

warnings.filterwarnings('ignore')

sys.path.append('../insightface/deploy')
sys.path.append('../insightface/src/common')


class FacePredictor():
    def __init__(self):
        try:
            # Face embedding model config
            self.image_size = '112,112'  # format expected by face_model.FaceModel
            self.model_path = "src/insightface/models/model-y1-test2/model,0"  # Check this path is correct!
            self.threshold = 1.24
            self.det = 0

            # Initialize MTCNN detector
            self.detector = MTCNN()

            # Initialize face embedding model
            self.embedding_model = face_model.FaceModel(self.image_size, self.model_path, self.threshold, self.det)

            # Load embeddings and label encoder
            self.embeddings_path = "./faceEmbeddingModels/embeddings.pickle"
            self.le_path = "./faceEmbeddingModels/le.pickle"

            self.data = pickle.loads(open(self.embeddings_path, "rb").read())
            self.le = pickle.loads(open(self.le_path, "rb").read())

            self.embeddings = np.array(self.data['embeddings'])
            self.labels = self.le.fit_transform(self.data['names'])

            # Load classifier model for face recognition
            self.model = load_model(ConfigurationsPOJO.clssfr_ModelPath)

            # Thresholds
            self.cosine_threshold = 0.8
            self.proba_threshold = 0.85
            self.comparing_num = 5

            # Tracker lists for tracking faces in video
            self.trackers = []
            self.texts = []
        except Exception as e:
            print("Error in FacePredictor init:", e)

    @staticmethod
    def findCosineDistance(vector1, vector2):
        """
        Calculate cosine distance between two vectors
        """
        vec1 = vector1.flatten()
        vec2 = vector2.flatten()
        a = np.dot(vec1.T, vec2)
        b = np.dot(vec1.T, vec1)
        c = np.dot(vec2.T, vec2)
        return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

    @staticmethod
    def CosineSimilarity(test_vec, source_vecs):
        """
        Verify the similarity of one vector to group vectors of one class
        """
        cos_dist = 0
        for source_vec in source_vecs:
            cos_dist += FacePredictor.findCosineDistance(test_vec, source_vec)
        return cos_dist / len(source_vecs)

    def detectFace(self):
        cap = cv2.VideoCapture(0)
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        print(f"Camera resolution: {frame_width} x {frame_height}")

        save_width = 800
        save_height = int(800 / frame_width * frame_height)

        frames = 0
        trackers = []
        texts = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frames += 1
            frame = cv2.resize(frame, (save_width, save_height))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if frames % 3 == 0:
                trackers = []
                texts = []

                bboxes = self.detector.detect_faces(frame)

                if len(bboxes) > 0:
                    for bboxe in bboxes:
                        bbox = bboxe['box']
                        bbox = np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
                        landmarks = bboxe['keypoints']
                        landmarks = np.array([
                            landmarks["left_eye"][0], landmarks["right_eye"][0], landmarks["nose"][0],
                            landmarks["mouth_left"][0], landmarks["mouth_right"][0],
                            landmarks["left_eye"][1], landmarks["right_eye"][1], landmarks["nose"][1],
                            landmarks["mouth_left"][1], landmarks["mouth_right"][1]
                        ])
                        landmarks = landmarks.reshape((2, 5)).T

                        nimg = face_preprocess.preprocess(frame, bbox, landmarks, image_size='112,112')
                        nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
                        nimg = np.transpose(nimg, (2, 0, 1))

                        embedding = self.embedding_model.get_feature(nimg).reshape(1, -1)

                        text = "Unknown"

                        preds = self.model.predict(embedding).flatten()
                        j = np.argmax(preds)
                        proba = preds[j]

                        match_class_idx = np.where(self.labels == j)[0]
                        selected_idx = np.random.choice(match_class_idx, self.comparing_num)
                        compare_embeddings = self.embeddings[selected_idx]

                        cos_similarity = self.CosineSimilarity(embedding, compare_embeddings)

                        if cos_similarity < self.cosine_threshold and proba > self.proba_threshold:
                            name = self.le.classes_[j]
                            text = f"{name}"
                            print(f"Recognized: {name} <{proba * 100:.2f}%>")

                        tracker = dlib.correlation_tracker()
                        rect = dlib.rectangle(bbox[0], bbox[1], bbox[2], bbox[3])
                        tracker.start_track(rgb, rect)

                        trackers.append(tracker)
                        texts.append(text)

                        y = bbox[1] - 10 if bbox[1] - 10 > 10 else bbox[1] + 10
                        cv2.putText(frame, text, (bbox[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 1)
                        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (179, 0, 149), 4)
            else:
                for tracker, text in zip(trackers, texts):
                    pos = tracker.get_position()

                    startX = int(pos.left())
                    startY = int(pos.top())
                    endX = int(pos.right())
                    endY = int(pos.bottom())

                    cv2.rectangle(frame, (startX, startY), (endX, endY), (179, 0, 149), 4)
                    cv2.putText(frame, text, (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 1)

            cv2.imshow("Face Recognition", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
