from flask import Flask, request, jsonify
import os
import torch
from ultralytics import YOLO
import csv
from pathlib import Path
import re

def next_fname(base: Path, style: str = "plain") -> Path:
    parent, stem, suffix = base.parent, base.stem, base.suffix

    if style == "plain":                           # good, good1, good2 …
        pattern = rf"{re.escape(stem)}(\d*){re.escape(suffix)}$"
        extract = lambda m: int(m.group(1) or 0)

    else:                                          # good, good(1), good(2) …
        pattern = rf"{re.escape(stem)}(?:\((\d+)\))?{re.escape(suffix)}$"
        extract = lambda m: int(m.group(1) or 0)

    # 이미 존재하는 파일들에서 가장 큰 번호를 찾음
    max_idx = -1
    for p in parent.glob(f"{stem}*{suffix}"):
        m = re.fullmatch(pattern, p.name)
        if m:
            max_idx = max(max_idx, extract(m))

    # 다음 번호로 파일 이름 생성
    idx = max_idx + 1
    if idx == 0:                                   # 첫 파일은 원본 이름 유지
        return parent / f"{stem}{suffix}"
    if style == "plain":
        return parent / f"{stem}{idx}{suffix}"
    return parent / f"{stem}({idx}){suffix}"

BASE = Path(r"C:/Users/pizza/Desktop/flat_and_hills_whole/good1.csv")

def save_hits(lidar_points, player_pose, *, style="plain"):
    """
    매 호출 시 새로운 CSV(good.csv, good1.csv …)에
    'isDetected' 포인트만 저장합니다.
    style: "plain" | "paren"
    """
    fname = next_fname(BASE, style=style)
    with fname.open("w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["angle", "vert_angle", "dist", "x", "y", "z", "height"])  # 헤더
        for p in lidar_points:
            if p["isDetected"]:
                pos = p["position"]
                wr.writerow([p["angle"], p["verticalAngle"], p["distance"],
                            pos["x"], pos["y"], pos["z"], player_pose['y']])
    # print(f"▶ 저장 완료: {fname.name}")
    
    
# FNAME = Path("C:/Users/pizza/Desktop/hi/good.csv")
# def save_hits(lidar_points, *, append=True):
#     """isDetected가 True인 포인트만 CSV에 스트림 저장."""
#     mode = "a" if append and FNAME.exists() else "w"
#     with FNAME.open(mode, newline="") as f:
#         wr = csv.writer(f)
#         if mode == "w":                # 헤더 최초 1회 기록
#             wr.writerow(["angle","vert_angle","dist","x","y","z"])
#         for p in lidar_points:
#             if p["isDetected"]:        # 가벼운 조건문
#                 pos = p["position"]
#                 wr.writerow([p["angle"], p["verticalAngle"], p["distance"],
#                             pos["x"],  pos["y"], pos["z"]])

app = Flask(__name__)
# model = YOLO('yolov8n.pt')
model = YOLO('C:\\pycode\\3rd_Project\\detection\\tank_detection_v1\\runs\\detect\\yolov8n_coco17val_tank_freeze20\\weights\\best.pt')

# Move commands with weights (11+ variations)
move_command = [
    {"move": "W", "weight": 1.0},
    {"move": "W", "weight": 0.6},
    {"move": "W", "weight": 0.3},
    {"move": "D", "weight": 1.0},
    {"move": "D", "weight": 0.6},
    {"move": "D", "weight": 0.4},
    {"move": "A", "weight": 1.0},
    {"move": "A", "weight": 0.3},
    {"move": "S", "weight": 0.5},
    {"move": "S", "weight": 0.1},
    {"move": "STOP"}
]

# Action commands with weights (15+ variations)
action_command = [
    {"turret": "Q", "weight": 1.0},
    {"turret": "Q", "weight": 0.8},
    {"turret": "Q", "weight": 0.6},
    {"turret": "Q", "weight": 0.4},
    {"turret": "E", "weight": 1.0},
    {"turret": "E", "weight": 1.0},
    {"turret": "E", "weight": 1.0},
    {"turret": "E", "weight": 1.0},
    {"turret": "F", "weight": 0.5},
    {"turret": "F", "weight": 0.3},
    {"turret": "R", "weight": 1.0},
    {"turret": "R", "weight": 0.7},
    {"turret": "R", "weight": 0.4},
    {"turret": "R", "weight": 0.2},
    {"turret": "FIRE"}
]

@app.route('/detect', methods=['POST'])
def detect():
    image = request.files.get('image')
    if not image:
        return jsonify({"error": "No image received"}), 400

    image_path = 'temp_image.jpg'
    image.save(image_path)

    results = model(image_path)
    detections = results[0].boxes.data.cpu().numpy()

    target_classes = {0: "person", 2: "car", 7: "truck", 15: "rock", 80:"tank"}
    filtered_results = []
    for box in detections:
        class_id = int(box[5])
        if class_id in target_classes:
            filtered_results.append({
                'className': target_classes[class_id],
                'bbox': [float(coord) for coord in box[:4]],
                'confidence': float(box[4]),
                'color': '#00FF00',
                'filled': False,
                'updateBoxWhileMoving': False
            })

    return jsonify(filtered_results)

@app.route('/info', methods=['POST'])
def info():
    data = request.get_json(force=True)
    # print(data['playerPos']['y'])
    save_hits(data['lidarPoints'], data['playerPos'])
    if not data:
        return jsonify({"error": "No JSON received"}), 400

  #  print("📨 /info data received:", data)

    # Auto-pause after 15 seconds
    #if data.get("time", 0) > 15:
    #    return jsonify({"status": "success", "control": "pause"})
    # Auto-reset after 15 seconds
    #if data.get("time", 0) > 15:
    #    return jsonify({"stsaatus": "success", "control": "reset"})
    return jsonify({"status": "success", "control": ""})

@app.route('/update_position', methods=['POST'])
def update_position():
    data = request.get_json()
    if not data or "position" not in data:
        return jsonify({"status": "ERROR", "message": "Missing position data"}), 400

    try:
        x, y, z = map(float, data["position"].split(","))
        current_position = (int(x), int(z))
        print(f"📍 Position updated: {current_position}")
        return jsonify({"status": "OK", "current_position": current_position})
    except Exception as e:
        return jsonify({"status": "ERROR", "message": str(e)}), 400

@app.route('/get_move', methods=['GET'])
def get_move():
    global move_command
    if move_command:
        command = move_command.pop(0)
        print(f"🚗 Move Command: {command}")
        return jsonify(command)
    else:
        return jsonify({"move": "STOP", "weight": 1.0})

@app.route('/get_action', methods=['GET'])
def get_action():
    global action_command
    if action_command:
        command = action_command.pop(0)
        print(f"🔫 Action Command: {command}")
        return jsonify(command)
    else:
        return jsonify({"turret": "", "weight": 0.0})

@app.route('/update_bullet', methods=['POST'])
def update_bullet():
    data = request.get_json()
    if not data:
        return jsonify({"status": "ERROR", "message": "Invalid request data"}), 400

    print(f"💥 Bullet Impact at X={data.get('x')}, Y={data.get('y')}, Z={data.get('z')}, Target={data.get('hit')}")
    return jsonify({"status": "OK", "message": "Bullet impact data received"})

@app.route('/set_destination', methods=['POST'])
def set_destination():
    data = request.get_json()
    if not data or "destination" not in data:
        return jsonify({"status": "ERROR", "message": "Missing destination data"}), 400

    try:
        x, y, z = map(float, data["destination"].split(","))
        print(f"🎯 Destination set to: x={x}, y={y}, z={z}")
        return jsonify({"status": "OK", "destination": {"x": x, "y": y, "z": z}})
    except Exception as e:
        return jsonify({"status": "ERROR", "message": f"Invalid format: {str(e)}"}), 400

@app.route('/update_obstacle', methods=['POST'])
def update_obstacle():
    data = request.get_json()
    if not data:
        return jsonify({'status': 'error', 'message': 'No data received'}), 400

    print("🪨 Obstacle Data:", data)
    return jsonify({'status': 'success', 'message': 'Obstacle data received'})

@app.route('/collision', methods=['POST']) 
def collision():
    data = request.get_json()
    if not data:
        return jsonify({'status': 'error', 'message': 'No collision data received'}), 400

    object_name = data.get('objectName')
    position = data.get('position', {})
    x = position.get('x')
    y = position.get('y')
    z = position.get('z')

    print(f"💥 Collision Detected - Object: {object_name}, Position: ({x}, {y}, {z})")

    return jsonify({'status': 'success', 'message': 'Collision data received'})

#Endpoint called when the episode starts
@app.route('/init', methods=['GET'])
def init():
    config = {
        "startMode": "start",  # Options: "start" or "pause"
        "blStartX": 60,  #Blue Start Position
        "blStartY": 10,
        "blStartZ": 27.23,
        "rdStartX": 59, #Red Start Position
        "rdStartY": 10,
        "rdStartZ": 280,
        "trackingMode": True,
        "detactMode": False,
        "logMode": True,
        "enemyTracking": True,
        "saveSnapshot": False,
        "saveLog": True,
        "saveLidarData": False,
        "lux": 30000
    }
    print("🛠️ Initialization config sent via /init:", config)
    return jsonify(config)

@app.route('/start', methods=['GET'])
def start():
    print("🚀 /start command received")
    return jsonify({"control": ""})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
