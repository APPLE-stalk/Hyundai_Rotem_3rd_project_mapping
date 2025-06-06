import os # 파일 입출력
import glob # 경로 다루는 라이브러리
import csv # csv 파일 읽기
import numpy as np


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # 3차원 시각화
import pyvista as pv # GPU 가속화하여 3D 시각화


# ====================  포인트 클라우드 시각화 ====================
cloud = pv.read("flat(100x100)_pc(0.1x0.1)_PolyData_LEFT.ply")                                                          # pyvista 라이브러리 활용하여 point cloud 다룸
plotter = pv.Plotter()
plotter.add_mesh(
    cloud,
    point_size=4,
    render_points_as_spheres=True,
    color="white"
)
plotter.add_axes()    # 축 표시
plotter.show_bounds(
    grid="front", location="outer", all_edges=True,
    xtitle="X", ytitle="Y", ztitle="Z"
)

# 3) 렌더링 실행
plotter.show()




map_mesh = pv.read("flat(100x100)_voxel(1.0x1.0)_MultiBlock_LEFT.vtk")
plotter = pv.Plotter()
plotter.add_mesh(map_mesh, color='light_gray', opacity=1) # 색갈, 투명도
# 축 라벨 추가
plotter.add_axes()
plotter.show_axes()  # 축 방향 위젯
plotter.show_bounds(
    grid='front', location='outer', all_edges=True,
    xtitle='X Axis', ytitle='Y Axis', ztitle='Z Axis'
)
plotter.show()