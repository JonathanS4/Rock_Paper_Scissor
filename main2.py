import cv2
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import pandas as pd

df=pd.DataFrame()

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

def draw_landmarks_on_image(rgb_image, detection_result):
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)

  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

    # Draw the hand landmarks.
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      hand_landmarks_proto,
      solutions.hands.HAND_CONNECTIONS,
      solutions.drawing_styles.get_default_hand_landmarks_style(),
      solutions.drawing_styles.get_default_hand_connections_style())

    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN

    # Draw handedness (left or right hand) on the image.
    cv2.putText(annotated_image, f"{handedness[0].category_name}",
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

  return annotated_image

base_options = python.BaseOptions(model_asset_path='MiniProject/hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)
cap=cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        mp_image=mp.Image(image_format=mp.ImageFormat.SRGB,data=frame)
        detection_result = detector.detect(mp_image)
        annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), detection_result)
        cv2.imshow("Video",cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        keys=cv2.waitKey(1)
        d={}
        if len(detection_result.hand_landmarks)>0:
          if keys==ord("r"):
            print(detection_result,"r")
            for index,i in enumerate(detection_result.hand_landmarks[0]):
              d["x"+str(index)]=i.x
              d["y"+str(index)]=i.y
              d["z"+str(index)]=i.z
            d["class"]=0
            df=pd.concat([df,pd.DataFrame([d])])
            print(df)
          elif keys==ord("p"):
            for index,i in enumerate(detection_result.hand_landmarks[0]):
              d["x"+str(index)]=i.x
              d["y"+str(index)]=i.y
              d["z"+str(index)]=i.z
            d["class"]=1
            df=pd.concat([df,pd.DataFrame([d])])
            print(df)
          elif keys==ord("s"):
            for index,i in enumerate(detection_result.hand_landmarks[0]):
              d["x"+str(index)]=i.x
              d["y"+str(index)]=i.y
              d["z"+str(index)]=i.z
            d["class"]=2
            df=pd.concat([df,pd.DataFrame([d])])
            print(df)
          elif keys==ord("q"):
            df.to_csv("data.csv")
            break
        
            
cap.release()
cv2.destroyAllWindows()