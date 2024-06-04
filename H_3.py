###############################################################################################################
# DO CHIEU CAO NGUOI: TINH TOAN THEO KHUNG XUONG
# XUAT TOA DO VA CHIEU CAO -> CSV -> EXCEL
# XUAT ANH
# -------------------------------------------------------------------------------------------------------------
import math
import cv2
import mediapipe as mp
import pandas as pd
import csv
import openpyxl

# Initialize mediapipe pose class.
mp_pose = mp.solutions.pose

# Setup the Pose function for images - independently for the images standalone processing.
pose_image = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# Initialize mediapipe drawing class - to draw the landmarks points.
mp_drawing = mp.solutions.drawing_utils

lm_list = []
lmhb_list = []
def make_landmark_timestep(results):
    c_lm = []
    for id, lm in enumerate(results.pose_landmarks.landmark):
        c_lm.append(lm.x)
        c_lm.append(lm.y)
        c_lm.append(lm.z)
    return c_lm

def draw_landmark_on_image(mp_drawing, results, img):
    # Vẽ các đường nối
    mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    # Vẽ các điểm nút
    for id, lm in enumerate(results.pose_landmarks.landmark):
        h, w, c = img.shape
        print(id, lm)
        print(h)
        print(w)
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(img, (cx, cy), 10, (0, 0, 255), cv2.FILLED)
    return img

def detectPose(image_pose, pose, draw=False, display=False):
    image = image_pose.copy()

    image_in_RGB = cv2.cvtColor(image_pose, cv2.COLOR_BGR2RGB)

    results = pose.process(image_in_RGB)
    if results.pose_landmarks and draw:
        mp_drawing.draw_landmarks(image=image, landmark_list=results.pose_landmarks,
                                  connections=mp_pose.POSE_CONNECTIONS)
        lst = []
        n = 0
        for id, lm in enumerate(results.pose_landmarks.landmark):
            lst[n] = lst.append([id, lm.x, lm.y])
            n + 1
            h, w, c = image.shape
            if id == 0:
                x0, y0 = int(lm.x * w), int(lm.y * h)
            if id == 11:
                x11, y11 = int(lm.x * w), int(lm.y * h)
            if id == 12:
                x12, y12 = int(lm.x * w), int(lm.y * h)
                xh5 = ((x11 + x12) / 2) - x0
                yh5 = ((y11 + y12) / 2) - y0
                h5 = (xh5 ** 2 + yh5 ** 2) ** 0.5  # Khoảng cách giữa trung điểm vai đến đỉnh mũi
            if id == 23:
                x23, y23 = int(lm.x * w), int(lm.y * h)
            if id == 24:
                x24, y24 = int(lm.x * w), int(lm.y * h)
                # h1 = (11+12,23+24)
                xh1 = (x23 + x24 - x11 - x12) / 2
                yh1 = (y23 + y24 - y11 - y12) / 2
                h1 = (xh1 ** 2 + yh1 ** 2) ** 0.5  # Khoảng cách giữa trung điểm vai đến trung điểm hông
            if id == 26:
                x26, y26 = int(lm.x * w), int(lm.y * h)
                # h2 = (24,26)
                xh2 = (x26 - x24)
                yh2 = (y26 - y24)
                h2 = (xh2 ** 2 + yh2 ** 2) ** 0.5  # Độ dài đùi
            if id == 28:
                x28, y28 = int(lm.x * w), int(lm.y * h)
                # h3 = (26,28)
                xh3 = (x28 - x26)
                yh3 = (y28 - y26)
                h3 = (xh3 ** 2 + yh3 ** 2) ** 0.5  # Độ dài bắp chân
            if id == 30:
                x30, y30 = int(lm.x * w), int(lm.y * h)
            if id == 32:
                x32, y32 = int(lm.x * w), int(lm.y * h)
                # h4 = (28,30+32)
                # đt đi qua 30 và 32 có phương trình ax + by + c = 0
                a = y32 - y30
                b = x30 - x32
                c = -x30 * (y32 - y30) - y30 * (x30 - x32)
                h4 = math.fabs(a * x28 + b * y28 + c) / ((a ** 2 + b ** 2) ** 0.5)  # Khoảng cách từ cổ chân đến bàn chân
                # tính chiều cao
                hb = h5 + h1 + h2 + h3 + h4
                hi = round(hb, 3)
                lmhb_list.append([h5,h1,h2,h3,h4,hi])
        # Ghi nhận thông số khung xương
        lm = make_landmark_timestep(results)
        lm_list.append(lm)

        # Vẽ khung xương lên ảnh
        image = draw_landmark_on_image(mp_drawing, results, image)
    else:

        return image, results

    image = cv2.resize(image, (405, 720))
    output_path = ".\h_IMG_3872_frame_350.jpg"
    cv2.imshow('.\IMG_3872_frame_350.jpg', image)
    cv2.imwrite(output_path, image)
    cv2.waitKey(10)
# Here we will read our image from the specified path to detect the pose
image_path = 'D:\IMG\IMG_3872_frame_350.jpg'
output = cv2.imread(image_path)
detectPose(output, pose_image, draw=True, display=True)

a="350"
# Write vào file csv
df = pd.DataFrame(lm_list)
df.to_csv(a+"H3_xyz.csv")
df_hb = pd.DataFrame(lmhb_list)
df_hb.to_csv(a+"h5_h1_h2_h3_h4_hi.csv")

cv2.destroyAllWindows()

def csv_to_excel(csv_file, excel_file):
    csv_data = []
    with open(csv_file) as file_obj:
        reader = csv.reader(file_obj)
        for row in reader:
            csv_data.append(row)
    workbook = openpyxl.load_workbook(filename=a+"H3_xyz.xlsx")
    workbook_hb = openpyxl.load_workbook(filename=a+"h5_h1_h2_h3_h4_hi.xlsx")
    sheet1 = workbook.active
    sheet2 = workbook_hb.active
    for row in csv_data:
        sheet1.append(row)
        sheet2.append(row)
    workbook.save(excel_file)
    workbook_hb.save(excel_file)
    if __name__ == "__main__":
        csv_to_excel(a+"H3_xyz.csv", a+"H3_xyz.xlsx")
        csv_to_excel(a+"h5_h1_h2_h3_h4_hi.csv", a+"h5_h1_h2_h3_h4_hi.xlsx")
    cv2.destroyAllWindows()


