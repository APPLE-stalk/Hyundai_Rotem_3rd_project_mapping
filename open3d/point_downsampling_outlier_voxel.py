# 3d 라이다 활용하여 mapping하려 함
# 우선 라이다에 감지된 점의 절대 좌표를 활용함


import os # 파일 입출력
import glob # 경로 다루는 라이브러리
import csv # csv 파일 읽기
import numpy as np


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # 3차원 시각화
import pyvista as pv # GPU 가속화하여 3D 시각화

from collections import defaultdict # voxel 제작에 사용
import open3d as o3d # downsampling & outlier 제거


lidar_data_folder = '../lidar_data_10'#'C:/Users/pizza/Documents/Tank Challenge/lidar_data'
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

# 해상도 0.1m로 하기 위해 소숫점 2째자리 버림
global_point_cloud = [tuple(round(coord, 1) for coord in point) for point in global_point_cloud] 
# print('global point cloud[0]: ', global_point_cloud[0])

# 튜플 → 넘파이 배열 변환
global_point_cloud = np.array(global_point_cloud)
print(type(global_point_cloud))

# 왼손 좌표계를 오른손 좌표계로 표현하기 위해  x, y, z -> x, z, y
converted_points = np.zeros_like(global_point_cloud)
converted_points[:, 0] = global_point_cloud[:, 0]        # X
converted_points[:, 1] = global_point_cloud[:, 2]        # Y자리에 Z 넣음
converted_points[:, 2] = global_point_cloud[:, 1]        # Z자리에 Y 넣음


# downsampling & outliner 제거
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(converted_points)

# Downsampling (Voxel Grid)
# 3D공간을 0.5m*0.5m*0.5m로 정육면체로 나누고 각 공간에서 대표 포인트 1개만 고름
pcd = pcd.voxel_down_sample(voxel_size=0.5) 


# Outlier 제거 (statistical filter)
pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
# 검사할 점 주변의 nb_neighbors 갯수의 점을 검사
# 점들의 평균 거리의 평균값과 표준편차 계산
# 평균거리 > (전체 평균거리 + std_ration*표준편차) 인 점들을 outlier로 간주하고 제거

# point clouds저장
o3d.io.write_point_cloud("downsampling_outliner_point_cloud.ply", pcd)




# 2. VoxelGrid로 변환
voxel_size = 0.5  # 원하는 voxel 크기 지정 (단위: meter)
voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)

# 3. 시각화
o3d.visualization.draw_geometries([voxel_grid])


