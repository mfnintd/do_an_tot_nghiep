import tkinter as tk
from tkinter import Label, Entry, Button, ttk
import cv2
from PIL import Image, ImageTk
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import pandas as pd
import helper
import datetime
from time import gmtime, strftime

"""
h1: khoảng cách từ mắt cá chân đến lòng bàn chân: 28 -> 30-32
h2: khoảng cách từ đầu gối đến mắt cá chân: 26 -> 28
h3: khoảng cách từ hông đến đầu gối: 24 -> 26
h4: khoảng cách từ vai đến hông: 11-12 -> 23-24
h5: khoảng cách từ mũi đến trung điểm của vai: 0 -> 11-12
"""
h1 = h2 = h3 = h4 = h5 = height = 0

df = pd.read_csv('data/data.csv')
print(df.shape[0])
model_path = "./landmark_model/pose_landmarker_heavy.task"
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options)

def show_camera():
    global cur_frame
    _, frame = cap.read()
    #frame = cv2.cvtColor(frame)
    #print(frame.shape)
    frame = frame[:, 200:440].copy()
    #print(frame)
    #cv2.imshow('hehe', frame)
    #print(frame)
    #
    #TODO: Chỗ này xử lý với frame
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    detection_result = detector.detect(img)
    #print(len(detection_result))
    frame = draw_landmarks_on_image(img.numpy_view(), detection_result)

    #frame = cv2.flip(frame, 1)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #cv2_imshow(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    #
    cur_frame = frame

    img = Image.fromarray(frame)
    img = img.resize((350, 700))
    imgtk = ImageTk.PhotoImage(image=img)
    lbl.imgtk = imgtk
    lbl.configure(image=imgtk)
    lbl.after(10, show_camera)

id = 0

def draw_landmarks_on_image(rgb_image, detection_result):
  global h1, h2, h3, h4, h5, height, id

  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)
  
  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
      pose_landmarks = pose_landmarks_list[idx]

      # Draw the pose landmarks.
      pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
      # pose_landmarks_proto.landmark.extend([
      # landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
      # ])
      pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=0) for landmark in pose_landmarks
      ])

      #landmark_coordinates = [np.array((landmark.x, landmark.y, landmark.z)) for landmark in pose_landmarks] 
      landmark_coordinates = [np.array((landmark.x, landmark.y, 0)) for landmark in pose_landmarks] 

      h1 = helper.disR2PQ(landmark_coordinates[30], landmark_coordinates[32], landmark_coordinates[28])

      h2 = helper.disPoint2Point(landmark_coordinates[26], landmark_coordinates[28])

      h3 = helper.disPoint2Point(landmark_coordinates[24], landmark_coordinates[26])

      h4 = helper.disPoint2Point(helper.middlePoint(landmark_coordinates[11], landmark_coordinates[12]),
                                 helper.middlePoint(landmark_coordinates[23], landmark_coordinates[24]))

      h5 = helper.disPoint2Point(landmark_coordinates[0], 
                                 helper.middlePoint(landmark_coordinates[11], landmark_coordinates[12]))
      #print(h1, h2, h3, h4, h5)
      id += 1
      if (id % 10 == 0):
         height = 110.02 * h4 + 256.61 * h3 + 87.81 * h2 -107.5 * h1 + 86.66 * h5 + 66.57
         height = 188.5 / 160 * height
      #print(height)
      solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
      cv2.putText(annotated_image, str(round(height,1)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
  return annotated_image

def add_data():
    distance = entry_distance.get()
    if distance == '':
       distance = '300'
    camera_height = entry_height.get()
    if camera_height == '':
       camera_height = '101'
    person_height = entry_person_height.get()
    if person_height == '':
       person_height = '0'
    type = label_combo.get()
    if type == '':
       type = 'unknown'
    # print(f"Distance from camera to person: {distance}")
    # print(f"Camera height: {camera_height}")
    # print(f"Person height: {person_height}")
    # print(f"type: {type}")
    global h1, h2, h3, h4, h5, df

    global cur_frame
    #print(f'data/image/{type}/{df.shape[0]}_{strftime("%Y%m%d%H%M%S", gmtime())}')
    cv2.imwrite(f'data/image/{type}/{df.shape[0]}_{strftime("%Y%m%d%H%M%S", gmtime())}.png', cv2.cvtColor(cur_frame, cv2.COLOR_RGB2BGR))

    df = pd.concat([df, pd.DataFrame({
          "distance": [distance],
          "camera_height": [camera_height],
          "person_height": [person_height],
          "type": [type],
          "h1": [h1],
          "h2": [h2],
          "h3": [h3],
          "h4": [h4],
          "h5": [h5]
        })],
        ignore_index = True
    )
    df.to_csv('./data/data.csv', index=False)
    print(df)
    print('add thành công')


# Create the main window
root = tk.Tk()
root.title("Camera Viewer")
root.geometry("1500x700")

# Camera display label
lbl = Label(root)
lbl.place(x=0, y=0, width=640, height=700)

# Distance entry
label_distance = Label(root, text="Distance from camera to person:")
label_distance.place(x=800, y=64)
entry_distance = Entry(root)
entry_distance.place(x=1000, y=64)

# Height entry
label_height = Label(root, text="Camera height:")
label_height.place(x=800, y=100)
entry_height = Entry(root)
entry_height.place(x=1000, y=100)

# person height
label_person_height = Label(root, text="Person height:")
label_person_height.place(x=800, y=164)
entry_person_height = Entry(root)
entry_person_height.place(x=1000, y=164)

# type
label_type = Label(root, text="Type:")
label_type.place(x=800, y=200)
label_combo = ttk.Combobox(
   state="readonly",
   values=['front', 'beside', 'random']
)
label_combo.place(x=1000, y=200)

# Print button
btn_print = Button(root, text="Take photo", command=add_data)
btn_print.place(x=1000, y=250)

# Open the camera
cap = cv2.VideoCapture(0)

# Start displaying the camera
show_camera()

# Run the application
root.mainloop()

# Release the camera when the application is closed
cap.release()
cv2.destroyAllWindows()