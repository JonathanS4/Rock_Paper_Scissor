import cv2
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import mediapipe as mp
import cv2
import pandas as pd
import time

from joblib import load
with open("LogisticModel.joblib", "rb") as f:
    model = load(f)

df=pd.DataFrame()

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

def draw_landmarks_on_image(rgb_image, detection_result: mp.tasks.vision.HandLandmarkerResult):
    try:
      if detection_result.hand_landmarks == []:
         return rgb_image
      else:
         hand_landmarks_list = detection_result.hand_landmarks
         handedness_list = detection_result.handedness
         annotated_image = np.copy(rgb_image)

         # Loop through the detected hands to visualize.
         for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]
            
            # Draw the hand landmarks.
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
               landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks])
            mp.solutions.drawing_utils.draw_landmarks(
               annotated_image,
               hand_landmarks_proto,
               mp.solutions.hands.HAND_CONNECTIONS,
               mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
               mp.solutions.drawing_styles.get_default_hand_connections_style())

         return annotated_image
    except:
        return rgb_image

class landmarker_and_result():
   def __init__(self):
      self.result = mp.tasks.vision.HandLandmarkerResult
      self.landmarker = mp.tasks.vision.HandLandmarker
      self.createLandmarker()
   
   def createLandmarker(self):
      # callback function
      def update_result(result: mp.tasks.vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
         self.result = result

      # HandLandmarkerOptions (details here: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker/python#live-stream)
      options = mp.tasks.vision.HandLandmarkerOptions( 
         base_options = mp.tasks.BaseOptions(model_asset_path="finger-counter/hand_landmarker.task"), # path to model
         running_mode = mp.tasks.vision.RunningMode.LIVE_STREAM, # running on a live stream
         num_hands = 2, # track both hands
        #  min_hand_detection_confidence = 0.5, # lower than value to get predictions more often
        #  min_hand_presence_confidence = 0.5, # lower than value to get predictions more often
        #  min_tracking_confidence = 0.5, # lower than value to get predictions more often
         result_callback=update_result)
      
      # initialize landmarker
      self.landmarker = self.landmarker.create_from_options(options)
   
   def detect_async(self, frame):
      # convert np frame to mp image
      mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
      # detect landmarks
      self.landmarker.detect_async(image = mp_image, timestamp_ms = int(time.time() * 1000))

   def close(self):
      # close landmarker
      self.landmarker.close()

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode
def print_result(result: HandLandmarkerResult, output_image: mp.Image):
    print("0")
    annotated_image = draw_landmarks_on_image(output_image.numpy_view(), result)
    cv2.imshow("Video",cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    keys=cv2.waitKey(1)
    l=[]
    print("1")
    if len(result.hand_landmarks)>0:
        print("2")
        if keys==ord("j"):
            for index,i in enumerate(result.hand_landmarks[0]):
                print("3")
                l=l+[i.x,i.y,i.z]
                pred=model.predict([l])
                if pred==[0]:
                    print("r")
                elif pred==[1]:
                    print("p")
                elif pred==[2]:
                    print("q")
        elif keys==ord("q"):
            return "q"
    print("4")


# options = HandLandmarkerOptions(base_options=BaseOptions(model_asset_path='MiniProject/hand_landmarker.task'),
#     running_mode=VisionRunningMode.LIVE_STREAM,
#     result_callback=print_result)
# timestamp_ms = 0
# with HandLandmarker.create_from_options(options) as landmarker:
cap=cv2.VideoCapture(0)
hand_landmarker = landmarker_and_result()
while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if ret:
        hand_landmarker.detect_async(frame)
        frame = draw_landmarks_on_image(frame,hand_landmarker.result)
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) == ord('q'):
            break
            
            
cap.release()
cv2.destroyAllWindows()