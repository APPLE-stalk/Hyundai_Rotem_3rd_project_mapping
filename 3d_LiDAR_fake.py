# 3d 라이다 활용하여 mapping하려 함
# 우선 라이다에 감지된 점의 절대 좌표를 활용함


import os # 파일 입출력
import glob # 경로 다루는 라이브러리
import csv # csv 파일 읽기
import numpy as np


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # 3차원 시각화

# json 파일들 경로
lidar_data_folder = './lidar_data'#'C:/Users/pizza/Documents/Tank Challenge/lidar_data'
csv_files = glob.glob(os.path.join(lidar_data_folder, '*.csv'))

global_point_cloudes = []
for f in csv_files: # 파일 열기
    # print(f) # f는 csv 파일 절대주소
    
    with open(f, mode='r', newline='') as file: # csv파일 1개씩 열기
        reader = csv.DictReader(file)
        # scan = []
        for row in reader: # 360줄 중 한 줄 씩 읽기
            # print(row)
            # print('hi')
            x = float(row['x'])
            y = float(row['y'])
            z = float(row['z'])
            t = str(row['isDetected'])
            
            if(t == 'True'): # detect 된 경우만 읽어들이기
                global_point_cloudes.append((x, y, z))
# print(len(global_point_cloudes))
# print(global_point_cloudes[0])


global_point_cloudes = [tuple(round(coord, 1) for coord in point) for point in global_point_cloudes]

# Create the 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# ax.scatter(x, y, z)  # Plot the single point
for(x, y, z) in global_point_cloudes:
    ax.plot(x, z, y, linestyle='', marker='.', markersize=2, color='blue')



# Label axes for clarity
ax.set_xlabel('X')
ax.set_ylabel('z')
ax.set_zlabel('Y')

# Optional: Set equal aspect ratio for all axes
ax.set_xlim([0, 120]) # x축
ax.set_ylim([0, 70]) # z축
ax.set_zlim([0, 100]) # y축

plt.title("3D Point Visualization")
plt.show()