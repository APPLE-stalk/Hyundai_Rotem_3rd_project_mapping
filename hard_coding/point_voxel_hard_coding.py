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

# json 파일들 경로
lidar_data_folder = './lidar_data_10'#'C:/Users/pizza/Documents/Tank Challenge/lidar_data'
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
plotter.show()



# PCL(Point Cloud Library) 없이 vexel 만드는 코드
def create_voxel_occupancy_map(points, voxel_size):
    """
    points: (N, 3) np.ndarray - global point cloud
    voxel_size: float - 크기 (ex: 0.2m)
    
    return:
        occupancy: set of voxel indices (x, y, z)
    """
    voxel_indices = np.floor(points / voxel_size).astype(int)
    occupancy = set(map(tuple, voxel_indices))  # 중복 제거
    return occupancy

# point cloud는 (N, 3) numpy 배열이어야 함
voxel_size = 0.5  # 1:1m, 0.1:10cm 크기의 큐브
occupancy_map = create_voxel_occupancy_map(global_point_cloud, voxel_size)

print(f"점유된 voxel 수: {len(occupancy_map)}")

voxels = []
for x, y, z in occupancy_map:
    center = np.array([x, z, y]) * voxel_size + voxel_size / 2
    # *voxel_size: [0.1m] -> [m] 단위로 변환
    # (1, 2, 3) -> 직육면체의 좌측 하단 모서리이다. + vexel_size/2 => 직육면체의 중심으로 이동함
    # y, z 바꿔서 넣기
    cube = pv.Cube(center=center, x_length=voxel_size, y_length=voxel_size, z_length=voxel_size)
    # 중심, 크기 지정하면 Cube 생성해 줌
    voxels.append(cube)

# 모든 큐브 합치기
map_mesh = pv.MultiBlock(voxels).combine() 
# 여러 mesh 객체(블록)를 하나의 묶음으로 처리하는 PyVista의 컨테이너 클래스
# combine(): 여러개의 cube를 단일 mesh로 병합
# 성능개선: 100개 따로 랜더링 vs 하나로 랜더링

# map_mesh파일을 저장하기
map_mesh.save("voxel_map_0.5.vtk")  # 또는 .ply, .obj, .vtp

