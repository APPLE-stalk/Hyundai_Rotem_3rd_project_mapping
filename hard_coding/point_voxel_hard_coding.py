# 3d 라이다 활용하여 mapping하려 함
# 우선 라이다에 감지된 점의 절대 좌표를 활용함


import os # 파일 입출력
import glob # 경로 다루는 라이브러리
import csv # csv 파일 읽기
import numpy as np

import pyvista as pv # GPU 가속화하여 3D 시각화

# ==================== LiDAR 정보 저장된 .csv 불러와 데이터 읽기  ====================
lidar_data_folder = '../lidar_data(100_100)' #'C:/Users/pizza/Documents/Tank Challenge/lidar_data'
csv_files = glob.glob(os.path.join(lidar_data_folder, '*.csv'))

global_point_cloud = []
for f in csv_files: # 파일 열기
    
    with open(f, mode='r', newline='') as file:                   
        reader = csv.DictReader(file)
        
        for row in reader:                                        
            x = float(row['x'])
            y = float(row['y'])
            z = float(row['z'])
            t = str(row['isDetected'])
            
            if(t == 'True'): # detect 된 경우만 읽어들이기
                global_point_cloud.append((x, y, z)) # 왼손 좌표계로 정의된 데이터이다



# ====================  전처리  ====================
global_point_cloud = [tuple(round(coord, 1) for coord in point) for point in global_point_cloud] # 소숫점 둘째자리에서 반올림하여 데이터 크기 줄임
global_point_cloud = list(set(global_point_cloud))                                               # set을 이용하여 겹치는 점 제거, 순서 보장 안 됨!
global_point_cloud = np.array(global_point_cloud)                                                # 리스트 → 넘파이 배열 변환



# ====================  포인트 클라우드 시각화 ====================
converted_points = np.zeros_like(global_point_cloud)                                             # /\/\/\/\/\/\/\ 시각화 위해 축 변환 /\/\/\/\/\/\/\
converted_points[:, 0] = global_point_cloud[:, 0]        
converted_points[:, 1] = global_point_cloud[:, 2]        
converted_points[:, 2] = global_point_cloud[:, 1] 
cloud = pv.PolyData(converted_points)                                                            # pyvista 라이브러리 활용하여 point cloud 다룸
plotter = pv.Plotter(title="point_cloud(0.1x0.1)_PolyData_Visual")
plotter.add_mesh(cloud, point_size=2, render_points_as_spheres=True)
# 축 라벨 추가
plotter.add_axes()
plotter.show_axes()  # 축 방향 위젯
plotter.show_bounds(
    grid='front', location='outer', all_edges=True,
    xtitle='X Axis', ytitle='Y Axis', ztitle='Z Axis'
)
plotter.show()

# ====================  시각화용 PolyData 저장하기 ====================
cloud.save("flat(100x100)_pc(0.1x0.1)_PolyData_LEFT.ply")



# ====================  voxel 만들기 ====================
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
voxel_size = 1  # 1:1m 크기의 큐브
occupancy_map = create_voxel_occupancy_map(converted_points, voxel_size) # /\/\/\/\/\/\/\ 시각화 위해 축 변환 /\/\/\/\/\/\/\

print(f"점유된 voxel 수: {len(occupancy_map)}")
print('vexels 만드는 중...')
voxels = []
for x, y, z in occupancy_map:
    center = np.array([x, y, z]) * voxel_size + voxel_size / 2
    # (1, 2, 3) -> 직육면체의 좌측 하단 모서리이다. + vexel_size/2 => 직육면체의 중심으로 이동함
    cube = pv.Cube(center=center, x_length=voxel_size, y_length=voxel_size, z_length=voxel_size)
    # 중심, 크기 지정하면 Cube 생성
    voxels.append(cube)
print('vexels 만들기 완료')



# ====================  mesh 만들기 ====================
print('vexels 합쳐서 mesh 만드는 중...')
map_mesh = pv.MultiBlock(voxels).combine()
# voxel 합치기
# voxel은 부피 개념, mesh는 표면 개념
# 여러 mesh 객체(블록)를 하나의 묶음으로 처리하는 PyVista의 컨테이너 클래스
# combine(): 여러개의 cube를 단일 mesh로 병합
# 성능개선: 100개 따로 랜더링 vs 하나로 랜더링
print('mesh 만들기 완료')


# ====================  mesh 시각화 ====================
plotter = pv.Plotter(title="mesh(1.0x1.0)_MultiBlock_Visual")
plotter.add_mesh(map_mesh, color='light_gray', opacity=1) # 색갈, 투명도
# 축 라벨 추가
plotter.add_axes()
plotter.show_axes()  # 축 방향 위젯
plotter.show_bounds(
    grid='front', location='outer', all_edges=True,
    xtitle='X Axis', ytitle='Y Axis', ztitle='Z Axis'
)
plotter.show()



# map_mesh파일을 저장하기
map_mesh.save("flat(100x100)_voxel(1.0x1.0)_MultiBlock_LEFT.vtk")  # 또는 .ply, .obj, .vtp