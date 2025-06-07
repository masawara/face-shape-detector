from flask import Flask, request, Response, send_file
from flask_cors import CORS  # ← 追加！
import cv2
import mediapipe as mp
import numpy as np
import math
import tempfile
import json
import io

app = Flask(__name__)
CORS(app)  # ← この1行を追加でCORS有効化！
mp_face_mesh = mp.solutions.face_mesh

def detect_face_shape(landmarks):
    zygoma_left = landmarks[454]
    zygoma_right = landmarks[234]
    jaw_left = landmarks[435]
    jaw_right = landmarks[215]
    forehead_top = landmarks[10]
    forehead_bottom = landmarks[152]

    face_width = math.dist(zygoma_left, zygoma_right)
    jaw_width = math.dist(jaw_left, jaw_right)
    face_height = math.dist(forehead_top, forehead_bottom)

    face_ratio = face_height / face_width
    face_to_jaw_ratio = face_width / jaw_width

    if face_to_jaw_ratio < 1.13:
        face_shape = "四角型"
        reason = "顔幅と顎幅が近く、角ばった印象があります。"
    elif face_ratio > 1.2:
        face_shape = "面長型"
        reason = "顔の縦の長さが横幅に比べて長く、面長な印象です。"
    elif 1.1 <= face_ratio <= 1.2:
        face_shape = "卵型"
        reason = "顔の縦横比のバランスが取れており、滑らかな卵型です。"
    else:
        face_shape = "丸型"
        reason = "顔の縦横比が低く、丸みを帯びた印象です。"

    return {
        "face_shape": face_shape,
        "reason": reason,
        "metrics": {
            "face_width": round(face_width, 2),
            "face_height": round(face_height, 2),
            "face_ratio": round(face_ratio, 3),
            "jaw_width": round(jaw_width, 2),
            "face_to_jaw_ratio": round(face_to_jaw_ratio, 3)
        }
    }

@app.route("/upload", methods=["POST"])
def upload():
    if 'image' not in request.files:
        return Response(json.dumps({"error": "画像ファイルが必要です"}, ensure_ascii=False), mimetype="application/json", status=400)

    file = request.files['image']
    if file.filename == '':
        return Response(json.dumps({"error": "ファイル名がありません"}, ensure_ascii=False), mimetype="application/json", status=400)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        file.save(tmp.name)
        image = cv2.imread(tmp.name)

    if image is None:
        return Response(json.dumps({"error": "画像の読み込みに失敗しました"}, ensure_ascii=False), mimetype="application/json", status=400)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
        result = face_mesh.process(image_rgb)

        if not result.multi_face_landmarks:
            return Response(json.dumps({"error": "顔が検出されませんでした"}, ensure_ascii=False), mimetype="application/json", status=400)

        face_landmarks = result.multi_face_landmarks[0]
        h, w, _ = image.shape
        landmark_list = [(lm.x * w, lm.y * h) for lm in face_landmarks.landmark]

        shape_result = detect_face_shape(landmark_list)
        json_str = json.dumps(shape_result, ensure_ascii=False)
        return Response(json_str, mimetype="application/json")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)