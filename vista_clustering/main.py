import os # 파일 입출력
import glob # 경로 다루는 라이브러리
import csv # csv 파일 읽기
import numpy as np
import pyvista as pv # GPU 가속화하여 3D 시각화

import point_processor
import visualizer

# json 파일들 경로
lidar_data_folder = '../lidar_data(100_100)'#'C:/Users/pizza/Documents/Tank Challenge/lidar_data'
csv_files = glob.glob(os.path.join(lidar_data_folder, '*.csv'))

global_point_cloud = []
for f in csv_files: # 파일 열기
    
    with open(f, mode='r', newline='') as file: # csv파일 1개씩 열기
        reader = csv.DictReader(file)
        
        for row in reader: # 360줄 중 한 줄 씩 읽기
            x = float(row['x'])
            y = float(row['y'])
            z = float(row['z'])
            t = str(row['isDetected'])
            
            if(t == 'True'): # detect 된 경우만 읽어들이기
                global_point_cloud.append((x, y, z))
# print(len(global_point_cloud))
# print(global_point_cloud[0])
# raw data --> (59.35704, 8.002197, 32.08697)

global_point_cloud = [tuple(round(coord, 1) for coord in point) for point in global_point_cloud]
# 리스트 → 넘파이 배열 변환
global_point_cloud = np.array(global_point_cloud)

converted_points = np.zeros_like(global_point_cloud)
converted_points[:, 0] = global_point_cloud[:, 0]        # X
converted_points[:, 1] = global_point_cloud[:, 2]        # Y자리에 Z 넣음
converted_points[:, 2] = global_point_cloud[:, 1]        # Z자리에 Y 넣음

# ====================  포인트 클라우드 시각화
cloud = pv.PolyData(converted_points)
plotter = pv.Plotter()
plotter.add_mesh(cloud, point_size=3, render_points_as_spheres=True)
# 축 라벨 추가
plotter.add_axes()
plotter.show_axes()  # 축 방향 위젯
plotter.show_bounds(
    grid='front', location='outer', all_edges=True,
    xtitle='X Axis', ytitle='Y Axis', ztitle='Z Axis'
)
# plotter.show()


# --- LiDAR 데이터 처리 ---
print("\n--- 포인트 처리 시작 ---")
# 8. 유효 포인트 추출 (point_processor는 X, Y, Z 순서로 반환해야 함)
# points_3d = point_processor.extract_points_from_data(process_list)

points_3d = global_point_cloud #converted_points

print('bbbbb', points_3d)
# 9. 클러스터링 수행
labels, num_clusters = point_processor.perform_clustering(points_3d)
# 10. 경계 상자 계산 (point_processor는 결과 딕셔너리에 'center' 포함)
bounding_boxes = point_processor.calculate_bounding_boxes(points_3d, labels)

# --- 처리된 객체 정보 저장 ---
# if bounding_boxes and current_ts: # 유효한 결과와 타임스탬프가 있을 때만 저장
#     print("\n--- 감지된 객체 정보 저장 ---")
#     # detected_objects 테이블에 저장
#     database_handler.save_detected_objects(current_ts, bounding_boxes, labels, points_3d)
# elif not bounding_boxes:
#     print("ℹ️ 저장할 감지된 객체 정보가 없습니다.")

# --- 중심점 추출/출력 (디버깅/활용 목적) ---
box_centers = []
if bounding_boxes:
    print("\n--- 계산된 경계 상자 중심점 ---")
    for i, bbox in enumerate(bounding_boxes):
            center = bbox.get('center')
            if center and isinstance(center, list) and len(center) == 3:
                box_centers.append(center)
                print(f"  - 객체({bbox.get('label','?')}) 중심: X={center[0]:.2f}, Y={center[1]:.2f}, Z={center[2]:.2f}")
            else: print(f"⚠️ BBox {i} 중심점 오류")
    print(f"✅ 총 {len(box_centers)}개 중심점 추출 완료.")
else: print("ℹ️ 계산된 경계 상자 없음.")

# ====> 추가: DB에서 궤적 데이터 로드 (시각화용) <====
# print("\n--- DB 궤적 데이터 로드 (시각화용) ---")
# load_trajectory_points 함수는 (N, 3) NumPy 배열 또는 None 반환
# trajectory_coords = database_handler.load_trajectory_points()


# --- 시각화 (궤적 데이터 함께 전달) ---
print("\n--- PyVista 시각화 시작 ---")
# ====> visualizer 호출 시 trajectory_coords 전달 <====
# points_3d가 비어있더라도 궤적은 있을 수 있으므로 호출은 함
visualizer.plot_with_pyvista(points_3d, labels, bounding_boxes)