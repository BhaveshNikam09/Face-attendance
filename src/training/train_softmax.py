from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import numpy as np
import pickle

from src.detectfaces_mtcnn.Configurations import get_logger
from src.training.softmax import SoftMax


class TrainFaceRecogModel:

    def __init__(self, args):
        self.args = args
        self.logger = get_logger()
        # Load the face embeddings
        self.data = pickle.loads(open(args["embeddings"], "rb").read())

    def trainKerasModelForFaceRecognition(self):
        # Encode the labels
        le = LabelEncoder()
        labels = le.fit_transform(self.data["names"])
        num_classes = len(np.unique(labels))
        labels = labels.reshape(-1, 1)
        
        one_hot_encoder = OneHotEncoder(categories='auto')
        labels = one_hot_encoder.fit_transform(labels).toarray()

        embeddings = np.array(self.data["embeddings"])

        # Initialize training parameters
        BATCH_SIZE = 8
        EPOCHS = 5
        input_shape = embeddings.shape[1]

        # Build softmax classifier
        softmax = SoftMax(input_shape=(input_shape,), num_classes=num_classes)
        model = softmax.build()

        # K-Fold cross validation
        cv = KFold(n_splits=5, random_state=42, shuffle=True)
        history = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}

        # Training loop
        for train_idx, valid_idx in cv.split(embeddings):
            X_train, X_val = embeddings[train_idx], embeddings[valid_idx]
            y_train, y_val = labels[train_idx], labels[valid_idx]

            his = model.fit(
                X_train, y_train,
                batch_size=BATCH_SIZE,
                epochs=EPOCHS,
                verbose=1,
                validation_data=(X_val, y_val)
            )

            print(his.history['accuracy'])
            self.logger.info(his.history['accuracy'])

            history['accuracy'] += his.history['accuracy']
            history['val_accuracy'] += his.history['val_accuracy']
            history['loss'] += his.history['loss']
            history['val_loss'] += his.history['val_loss']

        # Save model and label encoder
        model.save(self.args['model'])
        with open(self.args["le"], "wb") as f:
            f.write(pickle.dumps(le))

        # (Optional) Save history or plot
        # self.plot_history(history)

    # Optional: Plot training curve
    def plot_history(self, history):
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history['accuracy'], label='Train Accuracy')
        plt.plot(history['val_accuracy'], label='Val Accuracy')
        plt.legend()
        plt.title("Accuracy")

        plt.subplot(1, 2, 2)
        plt.plot(history['loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Val Loss')
        plt.legend()
        plt.title("Loss")

        plt.tight_layout()
        plt.show()
