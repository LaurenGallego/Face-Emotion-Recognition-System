from flask import Flask, render_template, Response, jsonify
import cv2
from ultralytics import YOLO
from collections import Counter

app = Flask(__name__)

# Load models (YOLOv8 style)
model_face = YOLO('yolo_model/YOLO11_10B_face.pt')
model_emotion = YOLO('yolo_model/YOLO11_20B_emotion.pt')

cap = cv2.VideoCapture(0)
emotion_log = []

def generate_frames():

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize for performance (you can try 320x240 if needed)
        frame = cv2.resize(frame, (640, 480))


        # Detect faces every 3rd frame
        face_results = model_face(frame)[0]

        for box in face_results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            face_crop = frame[y1:y2, x1:x2]

            if face_crop.size == 0:
                continue

            # Detect emotion on cropped face
            emotion_results = model_emotion(face_crop)[0]

            if emotion_results.boxes is not None and len(emotion_results.boxes) > 0:
                best_emotion = emotion_results.boxes[0]
                label = model_emotion.names[int(best_emotion.cls[0])]
                confidence = float(best_emotion.conf[0])

                # Save to log
                emotion_log.append(label)

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Encode frame for browser streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_results')
def get_results():
    emotion_count = dict(Counter(emotion_log))
    return jsonify(emotion_count)

@app.route('/reset')
def reset_log():
    emotion_log.clear()
    return "Log cleared"

if __name__ == '__main__':
    app.run(debug=True)