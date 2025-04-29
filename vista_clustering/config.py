# config.py
# 역할: 전체 파이프라인에서 사용되는 설정값들을 정의합니다.

import os

# --- 서버 설정 ---
SERVER_IP = "192.168.0.118"  # 실제 컴퓨터 1의 IP 주소로 변경 필요
SERVER_PORT = 5051
LIDAR_DATA_URL = f"http://{SERVER_IP}:{SERVER_PORT}/lidar_data"
TRAJECTORY_LOG_URL = f"http://{SERVER_IP}:{SERVER_PORT}/trajectory_log" # 궤적 로그 URL
KKKK_URL = f"http://{SERVER_IP}:{SERVER_PORT}/kkkk" # /info 데이터 GET URL (현재 사용 안함)
REQUEST_TIMEOUT = 20 # 데이터 요청 타임아웃 (초)

# --- 데이터베이스 설정 ---
DATABASE_FILE = "lidar_database.db" # 저장될 데이터베이스 파일 이름 (필요시 robot_data.db 등으로 변경)

# --- 포인트 처리 설정 ---
# LiDAR 스캔 데이터 처리 시 'isDetected'가 True인 포인트만 사용할지 여부
USE_ONLY_DETECTED_POINTS = True

# --- 지면 제거 설정 ---
ENABLE_GROUND_REMOVAL = True  # 지면 제거 활성화 여부
GROUND_REMOVAL_HEIGHT_THRESHOLD = 8.5 # 지면으로 간주할 최대 Y값 (m) - 환경에 맞게 조정

# --- 클러스터링 (DBSCAN) 설정 ---
# !!! 이 값들은 데이터 특성에 맞게 반드시 튜닝해야 합니다 !!!
DBSCAN_EPS = 1.0        # 클러스터 내 점 간 최대 거리 (m) - 예: 0.5 ~ 1.5 사이 값 시도
DBSCAN_MIN_SAMPLES = 10 # 클러스터를 형성하기 위한 최소 점 개수 - 예: 5 ~ 30 사이 값 시도

# --- 시각화 설정 ---
PLOT_AREA_LIMIT = 50    # 플롯 X, Z 축 기본 범위 (데이터 없거나 범위 작을 때 사용)
PLOT_HEIGHT_LIMIT = 10  # 플롯 Y(높이) 축 기본 최대값