import os
import cv2
import numpy as np

# โหลดไฟล์ cascade สำหรับการตรวจจับใบหน้า
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# กำหนด directory ที่เก็บภาพ
image_dir = 'Image_training'

if not os.path.exists(image_dir):
    print(f"โฟลเดอร์ {image_dir} ไม่พบ!")
else:
    print(f"โฟลเดอร์ {image_dir} พบ!")
    files = os.listdir(image_dir)
    faces = []
    labels = []

    for file in files:
        # ตรวจสอบว่าไฟล์เป็น .jpg หรือ .jpeg หรือไม่
        if file.endswith('.jpg') or file.endswith('.jpeg'):
            file_path = os.path.join(image_dir, file)
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

            if image is None:
                print(f"ไม่สามารถโหลดไฟล์: {file_path}")
                continue  # ถ้าภาพไม่สามารถโหลดได้ให้ข้ามไฟล์นี้ไป

            # ตรวจจับใบหน้าจากภาพ
            faces_detected = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces_detected:
                face = image[y:y + h, x:x + w]
                faces.append(face)
                labels.append(1)  # ปรับให้ตรงกับป้ายที่คุณต้องการ เช่น 1 สำหรับคน

    # ถ้ามีข้อมูลใน faces และ labels ก็สามารถฝึกโมเดลได้
    if len(faces) > 0 and len(labels) > 0:
        # สร้าง recognizer สำหรับการฝึก
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.train(faces, np.array(labels))

        # บันทึกโมเดลลงในไฟล์ .yml
        recognizer.save('face_model.yml')
        print("การฝึกอบรมโมเดลเสร็จสิ้นและบันทึกลงใน face_model.yml")
    else:
        print("ไม่พบใบหน้าสำหรับการฝึกอบรม")
