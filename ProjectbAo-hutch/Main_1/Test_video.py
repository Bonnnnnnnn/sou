import cv2
import face_recognition
import pickle
import numpy as np
import time
import os

# โหลดโมเดลใบหน้าจาก pickle
def load_face_model(model_path='face_encodings.pkl'):
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    return model_data['encodings'], model_data['labels']

# ฟังก์ชันเปรียบเทียบใบหน้า
def compare_faces(known_encodings, known_labels, face_encoding):
    if len(known_encodings) == 0:
        return "Unknown", 0.0

    face_distances = face_recognition.face_distance(known_encodings, face_encoding)
    best_match_index = np.argmin(face_distances)

    if face_recognition.compare_faces(known_encodings, face_encoding)[best_match_index]:
        person_name = known_labels[best_match_index].split('.')[0]
        person_name = ''.join([i for i in person_name if not i.isdigit()])
        match_score = 100 - (face_distances[best_match_index] * 100)
        return person_name, match_score

    return "Unknown", 0.0

# ฟังก์ชันตรวจจับมนุษย์ (ใช้ HOG + SVM ของ OpenCV)
def detect_humans(frame):
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    boxes, _ = hog.detectMultiScale(gray, winStride=(4, 4), padding=(8, 8), scale=1.05)
    
    return boxes

# ฟังก์ชันบันทึกภาพ Unknown Face
def save_unknown_face(frame, face_location):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    save_path = "unknown_faces"
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    (top, right, bottom, left) = face_location
    face_img = frame[top:bottom, left:right]
    filename = os.path.join(save_path, f"unknown_{timestamp}.jpg")
    
    cv2.imwrite(filename, face_img)
    print(f"📸 Unknown face saved: {filename}")

# เริ่มต้นระบบ
def start_face_recognition_system():
    known_encodings, known_labels = load_face_model('face_encodings.pkl')
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image from webcam")
            break

        # ตรวจจับมนุษย์ก่อน
        human_boxes = detect_humans(frame)

        # ตรวจจับใบหน้า
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        print(f"Detected humans: {len(human_boxes)}, Detected faces: {len(face_locations)}")

        # วาดกรอบรอบมนุษย์
        for (x, y, w, h) in human_boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # สีน้ำเงิน

        # วาดกรอบรอบใบหน้า + ใส่ชื่อ + แจ้งเตือน Unknown
        for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
            person_name, match_score = compare_faces(known_encodings, known_labels, face_encoding)

            if person_name == "Unknown":
                print("🚨 Warning! Unknown Face Detected!")
                save_unknown_face(frame, (top, right, bottom, left))

            color = (0, 255, 0) if person_name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, f"{person_name} ({match_score:.2f}%)", (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        cv2.imshow("Human & Face Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    start_face_recognition_system()
