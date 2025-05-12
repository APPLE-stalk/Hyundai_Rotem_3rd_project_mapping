import pyvista as pv
import open3d as o3d
import numpy as np


# point cloud ply파일 불러오기 파일 불러오기
# pcd = o3d.io.read_point_cloud("./pc/flat(100x100)_pc(0.5x0.5)_downsampling_outliner_rm_visual.ply")
# pcd = o3d.io.read_point_cloud("./pc/flat(100x100)_pc(0.5x0.5)_visual.ply")
# pcd = o3d.io.read_point_cloud("./pc/flat(100x100)_pc(0.5x0.5)_downsampling_outliner_rm_visual.ply")


# pcd = o3d.io.read_point_cloud("./pc/flat(100x100)_pc(0.5x0.5)_downsampling_outliner_rm_LH.ply")


# point cloud 시각화
# o3d.visualization.draw_geometries([pcd])


# voxel_size = 0.5  # 원하는 voxel 크기 지정 (단위: meter)
# voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)


# voxel 시각화
# o3d.visualization.draw_geometries([voxel_grid])


# ====================== OGM 시각화 ======================
# 1) OGM 불러오기
occ = np.load("./OGM_2.0x2.0_LH.npy")
voxel_size = 2
print(occ.shape)
nx, nz, ny = occ.shape

# ImageData 생성
origin = (0, 0, 0)
grid = pv.ImageData(
    dimensions=(nx+1, nz+1, ny+1), # 셀 단위로 사용할 땐 +1
    spacing=(voxel_size,)*3,       # 각 축 voxel 크기
    origin=tuple(origin)           # 맵의 원점
)


# 1D로 펼쳐서 grid 객체에 추가하기 
grid.cell_data["occ"] = occ.ravel(order="F") # order="F": 평탄화

# grid객체 중 장애물(occ>0.5)
voxels = grid.threshold(0.5, scalars="occ") # occ값이 임계값 0.5 이상(1인 voxel만)인 것들만 추려서 voxels에 담음

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