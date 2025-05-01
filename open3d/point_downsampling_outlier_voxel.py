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

# ==================== LiDAR log .csv 불러와 데이터 읽기  ====================
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
print('LiDAR log raw', global_point_cloud[0])


# ====================  전처리  ====================
global_point_cloud = [tuple(round(coord, 1) for coord in point) for point in global_point_cloud] # 소숫점 둘째자리에서 반올림하여 데이터 크기 줄임
# print('before point cloud downsampling', len(global_point_cloud))
# global_point_cloud = list(set(global_point_cloud)) # set을 이용하여 겹치는 점 제거, 순서 보장 안 됨!
# print('after point cloud downsampling', len(global_point_cloud))
global_point_cloud = np.array(global_point_cloud)  



converted_points = np.zeros_like(global_point_cloud)                                             # /\/\/\/\/\/\/\ 시각화 위해 축 변환 /\/\/\/\/\/\/\
converted_points[:, 0] = global_point_cloud[:, 0]        
converted_points[:, 1] = global_point_cloud[:, 2]        
converted_points[:, 2] = global_point_cloud[:, 1] 

# ====================  point cloud 객체 생성  ====================
pcd = o3d.geometry.PointCloud() # open 3d 이용하여 points 객체 생성
pcd.points = o3d.utility.Vector3dVector(converted_points) # open3d 내부에서 처리 가능한 포인트 벡터로 변환



# ====================  point cloud Downsampling for voxel  ====================
# 3D공간을 0.5m*0.5m*0.5m로 정육면체로 나누고 각 공간에서 대표 포인트 1개만 고름
print('before voxel downsampling', len(pcd.points))
pcd = pcd.voxel_down_sample(voxel_size=0.5) 
print('after voxel downsampling', len(pcd.points))

# ====================  point cloud Outlier 제거  ====================
# 통계적 방식 이용 statistical filter
print('before remove_outlier', len(pcd.points))
pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
print('after remove_outlier', len(pcd.points))
# 검사할 점 주변의 nb_neighbors 갯수의 점을 검사
# 점들의 평균 거리의 평균값과 표준편차 계산
# 평균거리 > (전체 평균거리 + std_ration*표준편차) 인 점들을 outlier로 간주하고 제거

# ====================  point cloud 시각화  ====================
# point cloud 시각화
o3d.visualization.draw_geometries([pcd])

# point clouds저장
# o3d.io.write_point_cloud("flat(100x100)_pc(0.5x0.5)_downsampling_outliner_rm_visual.ply", pcd)




# ====================  voxel 시각화  ====================
# voxel grid로 변환
voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.5) 

# draw
o3d.visualization.draw_geometries([voxel_grid])


