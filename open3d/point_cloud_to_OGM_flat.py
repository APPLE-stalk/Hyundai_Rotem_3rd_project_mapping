# 3d 라이다 활용하여 mapping하려 함
# 우선 라이다에 감지된 점의 절대 좌표를 활용함


import os # 파일 입출력
import glob # 경로 다루는 라이브러리
import csv # csv 파일 읽기
import numpy as np # numpy로 데이터 다루기
import time # 알고리즘 실행 시간 측정 용도

import pyvista as pv # GPU 가속화하여 3D 시각화
import open3d as o3d # point cloude 가공, downsampling & outlier 제거



# ==================== LiDAR log .csv 불러와 데이터 읽기  ====================
st_time = time.perf_counter()
# lidar_data_path = '../lidar_data_maze1'      
# lidar_data_path = '../lidar_data_rugged_pre1(100_100)'
# lidar_data_path = '../hi'
lidar_data_path = '../lidar_data(100_100)'
# lidar_data_path = '../lidar_data_kd_concept'
csv_files = glob.glob(os.path.join(lidar_data_path, '*.csv'))

global_point_cloud = []
for f in csv_files: # 파일 열기
    with open(f, mode='r', newline='') as file: # csv파일 1개씩 열기
        reader = csv.DictReader(file)
        for row in reader:                      # 360줄 중 한 줄 씩 읽기
            x = float(row['x'])
            y = float(row['y'])
            z = float(row['z'])
            # t = str(row['isDetected'])
            
            # if((t == 'TRUE') or (t == 'True')):                     # detect 된 경우만 읽어들이기
            global_point_cloud.append((x, y, z))
# print('LiDAR log raw', global_point_cloud[0])

ed_time = time.perf_counter()
print(f'맵의 point cloud csv파일 읽는데 걸리는 시간: {ed_time - st_time:.6f}초')


# ====================  전처리  ====================
st_time = time.perf_counter()
print(f'raw point cloud 갯수: {len(global_point_cloud)}개')
global_point_cloud = [tuple(round(coord, 2) for coord in point) for point in global_point_cloud] # 소숫점 둘째자리에서 반올림하여 데이터 크기 줄임
# print('before point cloud downsampling', len(global_point_cloud))
# global_point_cloud = list(set(global_point_cloud)) # set을 이용하여 겹치는 점 제거, 순서 보장 안 됨!, 추후 down sampling 과정과 겹침
# print('after point cloud downsampling', len(global_point_cloud))
global_point_cloud = np.array(global_point_cloud)  


converted_points = np.zeros_like(global_point_cloud)                                             
converted_points[:, 0] = global_point_cloud[:, 0]   # 우     
converted_points[:, 1] = global_point_cloud[:, 2]   # 전방   
converted_points[:, 2] = global_point_cloud[:, 1]   # 높이
ed_time = time.perf_counter()
print(f'point cloud 전처리에 걸리는 시간: {ed_time - st_time:.6f}초')


# ====================  point cloud 객체 생성  ====================
st_time = time.perf_counter()
pcd = o3d.geometry.PointCloud()                                 # open 3d 이용하여 points 객체 생성
pcd.points = o3d.utility.Vector3dVector(converted_points)       # open3d 내부에서 처리 가능한 포인트 벡터로 변환


# ====================  point cloud Downsampling for voxel  ====================
# 3D공간을 0.5m*0.5m*0.5m로 정육면체로 나누고 각 공간에서 대표 포인트 1개만 고름
print('before voxel downsampling', len(pcd.points))
voxel_size = 0.5
pcd = pcd.voxel_down_sample(voxel_size=voxel_size) 
print('after voxel downsampling', len(pcd.points))



# ====================  point cloud Outlier 제거  ====================
# 통계적 방식 이용 statistical filter
print('before remove_outlier', len(pcd.points))
pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
print('after remove_outlier', len(pcd.points))
# 검사할 점 주변의 nb_neighbors 갯수의 점을 검사
# 점들의 평균 거리의 평균값과 표준편차 계산
# 평균거리 > (전체 평균거리 + std_ration*표준편차) 인 점들을 outlier로 간주하고 제거

ed_time = time.perf_counter()
print(f'open3d를 이용하여 downsampling, outlier제거: {ed_time - st_time:.6f}초')



# ====================  point cloud 시각화  ====================
# point cloud 시각화
o3d.visualization.draw_geometries([pcd])

# point clouds저장
# o3d.io.write_point_cloud("flat(100x100)_pc(0.5x0.5)_downsampling_outliner_rm_visual.ply", pcd)
# o3d.io.write_point_cloud("flat(100x100)_pc(0.5x0.5)_downsampling_outliner_rm_LH.ply", pcd)
# o3d.io.write_point_cloud("flat(100x100)_pc(0.5x0.5)_LH.ply", pcd)
# o3d.io.write_point_cloud("rugged_pre1(100_100).ply", pcd)




# ====================  voxel grid 생성  ====================
# voxel grid로 변환
start_mk_voxelgrid = time.perf_counter()
voxel_size = 0.5 # 2, 0.5 ..
voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size) 
end_mk_voxelgrid = time.perf_counter()
print(f'voxelgrid 변환 소요시간: {end_mk_voxelgrid - start_mk_voxelgrid:.6f}초')
print("voxel_size:", voxel_grid.voxel_size)
print("origin:", voxel_grid.origin)
print("origin cell center:", voxel_grid.origin + voxel_size * 0.5)

# voxel grid 시각화
# 흑색
# o3d.visualization.draw_geometries([voxel_grid])

# voxel grid 저장
# o3d.io.write_voxel_grid("flat(100x100)_vg(2.0x2.0)_LH.ply", voxel_grid)



# ====================== OGM 만들기 ======================
st_time = time.perf_counter()
voxels = voxel_grid.get_voxels()
print('voxel 갯수: ', len(voxels)) # 2944개
# print('voxel 1개 예시: ', voxels[0])
# print('voxel 좌표', voxels[0].grid_index, 'voxel 색상', voxels[0].color) # voxel 데이터 접근 방법: 


# 맵 크기 계산 (셀 수)
# 각 voxel.grid_index 는 (i,j,k) 정수 인덱스
indices = np.array([v.grid_index for v in voxels])
min_idx = indices.min(axis=0) # axis=0의미: 같은 열끼리 비교하여 min값을 구하시오
max_idx = indices.max(axis=0) + 1 # 슬라이싱 할 때 마지막 인덱시 미포함 고려하여 +1씩 해주기
dims = (max_idx - min_idx).astype(int)  # (nx, ny, nz) 각 축별 길이값 구하기
nx, nz, ny = dims


# Occupancy 배열 (0=free, 1=occupied)
# 0==free => 갈 수 있음 // 1==occupied => 갈 수 없음
occ = np.zeros(dims, dtype=np.uint8)
for v in voxels:
    i, k, j = np.array(v.grid_index)
    i, k, j = np.array(v.grid_index)  - min_idx
    occ[i, k, j] = 1
ed_time = time.perf_counter()
print(f'OGM만들기 소요시간: {ed_time - st_time:.6f}초')



# ====================== OGM 저장 ======================
# OCC맵 저장 
# np.save("OGM_2.0x2.0_LH.npy", occ)         # .npy 형식으로 저장
# np.savez('OGM_2.0x2.0_LH_with_meta.npz',
#         data       = occ,
#         origin     = voxel_grid.origin,
#         resolution = voxel_size)

# np.save("OGM_kd_concept.npy", occ)         # .npy 형식으로 저장
# np.savez('OGM_kd_concept_with_meta.npz',
#         data       = occ,
#         origin     = voxel_grid.origin,
#         resolution = voxel_size)

# np.save("OGM_rugged_pre1(100_100).npy", occ)         # .npy 형식으로 저장
# np.savez('OGM_maze1_with_meta.npz',
#         data       = occ,
#         origin     = voxel_grid.origin,
#         resolution = voxel_size)

# ====================== OGM 시각화 ======================
# ImageData 생성
shift = np.array([0.0, 0.0, 0.5]) * voxel_size
origin = voxel_grid.origin +shift
grid = pv.ImageData(
    dimensions=(nx+1, nz+1, ny+1), # 셀 단위로 사용할 땐 +1
    spacing=(voxel_size,)*3,       # 각 축 voxel 크기
    origin=tuple(origin)           # 맵의 원점
)


# 1D로 펼쳐서 grid 객체에 추가하기 
grid.cell_data["occ"] = occ.ravel(order="F") # order="F": 평탄화

# grid객체 중 장애물(occ>0.5)
voxels = grid.threshold(0.5, scalars="occ") # occ값이 임계값 0.5 이상(1인 voxel만)인 것들만 추려서 voxels에 담음
# print("cell center:", voxel_grid.origin + voxel_size * 0.5)
# Plotter로 인터랙티브 시각화
plotter = pv.Plotter()
plotter.add_mesh(
    voxels,
    color="lightgray",
    opacity=0.8,
    show_edges=False
)
plotter.show_grid(
    xtitle="X (m)",
    ytitle="Z (m)",
    ztitle="Y (m)",
    show_xaxis=True,
    show_yaxis=True,
    show_zaxis=True,
    show_xlabels=True,
    show_ylabels=True,
    show_zlabels=True
)
plotter.show(title="3D Occupancy Grid")
plotter.close() 