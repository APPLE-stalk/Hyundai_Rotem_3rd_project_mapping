import os # 파일 입출력
import glob # 경로 다루는 라이브러리
import csv # csv 파일 읽기
import numpy as np


import open3d as o3d
import numpy as np
from open3d import pipelines as pp

# 1. TSDF 볼륨 생성 파라미터
# TSDF 볼륨 파라미터
voxel_length = 0.05         # [m]
sdf_trunc    = voxel_length * 5
volume_unit_resolution = 16 # 내부 버퍼 단위 격자 해상도
depth_sampling_stride = 4   # depth 이미지 샘플링 스텝

tsdf_volume = pp.integration.ScalableTSDFVolume(
    voxel_length,                  # voxel 한 변 길이
    sdf_trunc,                     # truncation 거리
    pp.integration.TSDFVolumeColorType.None,  # 컬러 사용 안 함
    volume_unit_resolution=volume_unit_resolution,
    depth_sampling_stride=depth_sampling_stride)





# 2. 정합된 포인트 클라우드 리스트 로드
pcd = o3d.io.read_point_cloud("downsampling_outliner_point_cloud_(100_100)_1.0.ply")
pts = np.asarray(pcd.points)


# 3. 가상 Depth 이미지 생성
#    실제 시뮬레이터가 Depth 이미지를 제공한다면 이 단계는 제거 가능
intrinsic = o3d.camera.PinholeCameraIntrinsic(
    o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
# 깊이 이미지를 uniform sampling으로 생성
depth = pp.integration.make_uniform_depth_image(
    pts, intrinsic, depth_scale=1000.0, depth_max=5.0)
# RGB 정보가 없으므로 빈 이미지 사용
color = o3d.geometry.Image(np.zeros(depth.shape, dtype=np.uint8))

rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color, depth,
    depth_scale=1000.0,
    depth_trunc=5.0,
    convert_rgb_to_intensity=False)

# 4. 통합 (extrinsic: scan의 월드 좌표가 이미 적용된 경우엔 identity)
extrinsic = np.eye(4)
tsdf_volume.integrate(rgbd, intrinsic, extrinsic)

# 5. 메쉬 추출 및 저장
mesh = tsdf_volume.extract_triangle_mesh()
mesh.compute_vertex_normals()
o3d.io.write_triangle_mesh("tsdf_mesh.ply", mesh)

print("TSDF 메쉬 tsdf_mesh.ply로 저장 완료")
