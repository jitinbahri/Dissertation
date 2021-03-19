import cv2
from time import sleep

CLASSES = ['SAFE', 'DANGER']
NEG_IDX = 0
POS_IDX = 1
FRAMES_PER_VIDEO = 100
VIDEOS_PER_CLASS = 2


def capture(num_frames, path='out.avi'):

    # Create a VideoCapture object
    cap = cv2.VideoCapture(0)

    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Unable to read camera feed")

    # Default resolutions of the frame are obtained.The default resolutions are system dependent.
    # We convert the resolutions from float to integer.
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(
        'M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

    print('Recording started')
    for i in range(num_frames):

        ret, frame = cap.read()

        if ret == True:
            # Write the frame into the file 'output.avi'
            out.write(frame)

    # When everything done, release the video capture and video write objects
    cap.release()
    out.release()


for take in range(VIDEOS_PER_CLASS):
    for cla in CLASSES:
        path = 'C:/Users/jitin/Documents/dissertation/implementation/data/{}{}.avi'.format(
            cla, take)
        print('Get ready to act:', cla)
        capture(FRAMES_PER_VIDEO, path=path)


# Create X, y series
import cv2
import numpy as np
from glob import *
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input


class VGGFramePreprocessor():

    def __init__(self, vgg_model):
        self.vgg_model = vgg_model

    def process(self, frame):
        img_data = cv2.resize(frame, (224, 224))
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)
        x = self.vgg_model.predict(img_data).flatten()
        x = np.expand_dims(x, axis=0)
        return x


def get_video_frames(video_path):
    vidcap = cv2.VideoCapture(video_path)
    success, frame = vidcap.read()
    while success:
        yield frame
        success, frame = vidcap.read()
    vidcap.release()


frame_preprocessor = VGGFramePreprocessor(
    VGG16(weights='imagenet', include_top=False))


# Load movies and transform frames to features
movies = []
X = []
y = []
for video_path in glob('C:/Users/jitin/Documents/dissertation/implementation/data/*.avi'):
    print('preprocessing', video_path)
    positive = CLASSES[POS_IDX] in video_path
    _X = np.concatenate([frame_preprocessor.process(frame)
                         for frame in get_video_frames(video_path)])
    _y = np.array(_X.shape[0] * [[int(not positive), int(positive)]])
    X.append(_X)
    y.append(_y)

X = np.concatenate(X)
y = np.concatenate(y)
print(X.shape)
print(y.shape)




from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

MODEL_PATH = 'model.h5'
EPOCHS = 20
HIDDEN_SIZE = 64

model = Sequential()
model.add(Dense(HIDDEN_SIZE, input_shape=(X.shape[1],)))
model.add(Dense(HIDDEN_SIZE//2))
model.add(Dropout(0.2))
model.add(Dense(HIDDEN_SIZE))
model.add(Dropout(0.25))
model.add(Dense(len(CLASSES), activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=42)

model.fit(x_train, y_train,
          batch_size=32, epochs=EPOCHS,
          validation_split=0.2)

model.save(MODEL_PATH)
y_true = [np.argmax(y) for y in y_test]
y_pred = [np.argmax(pred) for pred in model.predict(x_test)]
score = f1_score(y_true, y_pred)
print('F1:', score)

# Use this to load the model
model = load_model(MODEL_PATH)



# Infer on live video
from math import ceil
import subprocess
import cv2
import sys

TEST_FRAMES = 500

# Initialize camera
cap = cv2.VideoCapture(0)
# Check if camera opened successfully
if (cap.isOpened() == False):
    print("Unable to read camera feed")
    TEST_FRAMES = 0

# Start processing video
for i in range(TEST_FRAMES):
    ret, frame = cap.read()
    if not ret:
        continue
    x_pred = frame_preprocessor.process(frame)
    y_pred = model.predict(x_pred)[0]
    conf_negative = y_pred[NEG_IDX]
    conf_positive = y_pred[POS_IDX]
    cla = CLASSES[np.argmax(y_pred)]

    progress = int(100 * (i / TEST_FRAMES))
    message = 'testing {}%  conf_neg = {:.02f} conf_pos = {:.02f}   class = {}         \r'.format(
        progress, conf_negative, conf_positive, cla)
    sys.stdout.write(message)
    sys.stdout.flush()

cap.release()

