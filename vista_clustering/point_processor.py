# point_processor.py (PyVista 호환 좌표 순서, 중심점 계산 포함)
# 역할: LiDAR 포인트 데이터를 처리합니다 (추출, 지면 제거, 클러스터링, 경계 상자/중심점 계산).

import numpy as np
from sklearn.cluster import DBSCAN
import config # 설정값 가져오기
import time

# def extract_points_from_data(lidar_data_list):
#     """LiDAR 데이터에서 유효 3D 좌표 (X, Y, Z) 추출 (PyVista 호환)"""
#     points_3d = []
#     for p in lidar_data_list:
#         if isinstance(p, dict) and 'position' in p and isinstance(p['position'], dict) and all(k in p['position'] for k in ('x','y','z')):
#             include = not config.USE_ONLY_DETECTED_POINTS or p.get('isDetected') is True
#             if include:
#                 try: x,y,z = float(p['position']['x']), float(p['position']['y']), float(p['position']['z']); points_3d.append([x, y, z]); valid += 1
#                 except (ValueError, TypeError): pass # 변환 오류 시 조용히 건너뜀
#     return np.array(points_3d) if points_3d else np.empty((0, 3))

def perform_clustering(points_3d):
    """3D 포인트 클라우드에 지면 제거(선택) 및 DBSCAN 클러스터링 수행."""
    if points_3d.shape[0] == 0: return np.array([], dtype=int), 0
    points_for_clus = points_3d; ground_idx = np.zeros(points_3d.shape[0], dtype=bool); n_ground = 0
    if config.ENABLE_GROUND_REMOVAL:
        print(f" - 지면 제거 (Y <= {config.GROUND_REMOVAL_HEIGHT_THRESHOLD})...")
        ground_idx = points_3d[:, 1] <= config.GROUND_REMOVAL_HEIGHT_THRESHOLD # Y는 인덱스 1
        points_for_clus = points_3d[~ground_idx]; n_ground = np.sum(ground_idx)
        print(f" - 클러스터링 대상: {points_for_clus.shape[0]} (지면 제거: {n_ground})")
    else: print(" - 지면 제거 비활성화.")
    labels = np.full(points_3d.shape[0], -1, dtype=int); n_clusters = 0
    if points_for_clus.shape[0] >= config.DBSCAN_MIN_SAMPLES:
        try:
            print(f" - DBSCAN 실행 (eps={config.DBSCAN_EPS}, min={config.DBSCAN_MIN_SAMPLES})..."); t_s = time.time()
            db = DBSCAN(eps=config.DBSCAN_EPS, min_samples=config.DBSCAN_MIN_SAMPLES).fit(points_for_clus)
            t_e = time.time(); labels[~ground_idx] = db.labels_
            unique_lbls = set(db.labels_) - {-1}; n_clusters = len(unique_lbls)
            n_noise = np.sum(db.labels_ == -1)
            print(f" - DBSCAN 완료 ({t_e - t_s:.2f}초). {n_clusters} 클러스터 발견.")
            print(f"   (클러스터링 노이즈: {n_noise}, 지면 제거: {n_ground})")
        except Exception as e: print(f"❌ DBSCAN 오류: {e}")
    else: print(f"ℹ️ 클러스터링 대상 포인트 부족.")
    return labels, n_clusters

def calculate_bounding_boxes(points_3d, labels):
    """클러스터링 결과로부터 각 클러스터의 AABB 및 중심점 계산."""
    bboxes = []
    if points_3d.shape[0] == 0 or labels.shape[0] != points_3d.shape[0]: return bboxes
    unique_labels = set(labels); print(f" - 경계 상자 계산 시작 (고유 레이블: {len(unique_labels)})...")
    for k in unique_labels:
        if k == -1: continue
        mask = (labels == k); cluster_pts = points_3d[mask]
        if cluster_pts.shape[0] > 0:
            try:
                min_c = np.min(cluster_pts, axis=0); max_c = np.max(cluster_pts, axis=0); cen_c = (min_c + max_c) / 2.0
                bboxes.append({'label': k, 'min_coords': min_c.tolist(), 'max_coords': max_c.tolist(), 'center': cen_c.tolist()})
            except Exception as e: print(f"⚠️ 클러스터 {k} BBox 계산 오류: {e}")
    print(f" - 경계 상자 계산 완료: {len(bboxes)}개 생성.")
    return bboxes                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 