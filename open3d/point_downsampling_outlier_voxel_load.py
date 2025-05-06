import pyvista as pv
import open3d as o3d


# point cloud ply파일 불러오기 파일 불러오기
# pcd = o3d.io.read_point_cloud("./pc/flat(100x100)_pc(0.5x0.5)_downsampling_outliner_rm_visual.ply")
# pcd = o3d.io.read_point_cloud("./pc/flat(100x100)_pc(0.5x0.5)_visual.ply")
pcd = o3d.io.read_point_cloud("./pc/flat(100x100)_pc(0.5x0.5)_downsampling_outliner_rm_visual.ply")
# pcd = o3d.io.read_point_cloud("./pc/flat(100x100)_pc(0.5x0.5)_downsampling_outliner_rm_LH.ply")


# point cloud 시각화
o3d.visualization.draw_geometries([pcd])


voxel_size = 0.5  # 원하는 voxel 크기 지정 (단위: meter)
voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)


# voxel 시각화
o3d.visualization.draw_geometries([voxel_grid])
