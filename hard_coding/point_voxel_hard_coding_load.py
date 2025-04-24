import os # 파일 입출력
import glob # 경로 다루는 라이브러리
import csv # csv 파일 읽기
import numpy as np


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # 3차원 시각화
import pyvista as pv # GPU 가속화하여 3D 시각화

map_mesh = pv.read("voxel_map_0.5.vtk")
plotter = pv.Plotter()
plotter.add_mesh(map_mesh, color='gray', opacity=1) # 색갈, 투명도
# 축 라벨 추가
plotter.add_axes()
plotter.show_axes()  # 축 방향 위젯
plotter.show_bounds(
    grid='front', location='outer', all_edges=True,
    xtitle='X Axis', ytitle='Y Axis', ztitle='Z Axis'
)
plotter.show()