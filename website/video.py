from flask import Blueprint, Response, jsonify, render_template
import cv2
from ultralytics import YOLO
from collections import Counter
from flask_login import login_required, current_user
from datetime import datetime
from . import db
from .models import Report
import json

video = Blueprint('video', __name__)

# Load models once
model_face = YOLO('website/yolo_model/YOLO11_10B_face.pt')
model_emotion = YOLO('website/yolo_model/YOLO11_20B_emotion.pt')

emotion_log = []
streaming = False


@video.route('/video')
@login_required
def video_page():
    return render_template("video.html")


@video.route('/video_feed')
@login_required
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@video.route('/start_stream')
@login_required
def start_stream():
    global streaming
    streaming = True
    return jsonify({'status': 'started'})


@video.route('/stop_stream')
@login_required
def stop_stream():
    global streaming
    streaming = False
    return jsonify({'status': 'stopped'})


@video.route('/get_report')
@login_required
def get_report():
    global emotion_log
    summary = dict(Counter(emotion_log))
    emotion_log.clear()

    user_id = current_user.id
    data = json.dumps(summary)  

    new_report = Report(data=data, user_id=user_id)
    db.session.add(new_report)
    db.session.commit()

    print(f"[INFO] Report saved for user {user_id}")
    return jsonify(summary)


def generate_frames():
    global streaming, emotion_log
    print("[INFO] Starting video capture...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[ERROR] Could not access the webcam.")
        return

    while streaming:
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Frame capture failed.")
            break

        frame = cv2.resize(frame, (640, 480))
        face_results = model_face(frame)[0]

        for box in face_results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue

            emotion_results = model_emotion(face_crop)[0]

            if emotion_results.boxes is not None and len(emotion_results.boxes) > 0:
                best_emotion = emotion_results.boxes[0]
                label = model_emotion.names[int(best_emotion.cls[0])]
                confidence = float(best_emotion.conf[0])
                emotion_log.append(label)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    print("[INFO] Releasing video capture.")
    cap.release()
