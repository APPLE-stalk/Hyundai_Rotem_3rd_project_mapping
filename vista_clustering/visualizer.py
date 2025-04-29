# visualizer.py (PyVista 사용, 궤적 플로팅 추가)
# 역할: 처리된 포인트 클라우드, 객체, 궤적 정보를 PyVista로 시각화합니다.

import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt # 컬러맵 사용 위해 유지
import config

def plot_with_pyvista(points_3d, labels, bounding_boxes, trajectory_points=None):
    """PyVista로 3D 포인트 클라우드, 경계 상자, 궤적 시각화"""
    print("📊 PyVista 시각화 생성 중...")
    has_lidar_points = points_3d is not None and points_3d.shape[0] > 0
    has_trajectory = trajectory_points is not None and trajectory_points.shape[0] > 1

    if not has_lidar_points and not has_trajectory: print("⚠️ 시각화할 데이터 없음."); return

    try: plotter = pv.Plotter(window_size=[2400, 1600])
    except Exception as e: print(f"❌ PyVista Plotter 생성 실패: {e}"); return

    # --- 포인트 클라우드 플로팅 ---
    if has_lidar_points:
        point_cloud = pv.PolyData(points_3d); point_cloud['labels'] = labels
        unique_labels = set(labels); num_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        valid_labels = sorted([l for l in unique_labels if l != -1])
        cmap_clusters = plt.cm.get_cmap("rainbow", num_clusters if num_clusters > 0 else 1)

        noise_mask = (labels == -1)
        if np.any(noise_mask): plotter.add_mesh(pv.PolyData(points_3d[noise_mask]), color='lightgrey', point_size=2, style='points', render_points_as_spheres=False, label='Noise/Ground')
        cluster_mask = (labels != -1)
        if np.any(cluster_mask):
            cluster_points = pv.PolyData(points_3d[cluster_mask]); cluster_points['labels'] = labels[cluster_mask]
            plotter.add_mesh(cluster_points, scalars='labels', cmap='rainbow', clim=[0, num_clusters-1] if num_clusters>0 else [0,0], point_size=4, render_points_as_spheres=True, scalar_bar_args={'title': 'Cluster ID'})
        print(f" - 포인트 클라우드 플롯: {points_3d.shape[0]} points")
    else: print(" - LiDAR 포인트 데이터 없음.")

    # --- 경계 상자 플로팅 ---
    if bounding_boxes:
        print(f" - 경계 상자 플롯: {len(bounding_boxes)} objects")
        bbox_label_added = False
        for i, bbox in enumerate(bounding_boxes):
            label = bbox.get('label', -1)
            if label != -1:
                min_c, max_c, cen_c = bbox['min_coords'], bbox['max_coords'], bbox['center']
                lengths = (max_c[0]-min_c[0], max_c[1]-min_c[1], max_c[2]-min_c[2])
                box_mesh = pv.Cube(center=cen_c, x_length=lengths[0], y_length=lengths[1], z_length=lengths[2])
                try: c_idx = valid_labels.index(label) if label in valid_labels else 0; color = cmap_clusters(c_idx/num_clusters if num_clusters>0 else .5)[:3]
                except: color = (.5,.5,.5)
                lbl_txt = 'Detected Object (AABB)' if not bbox_label_added else None; plotter.add_mesh(box_mesh, color=color, style='wireframe', line_width=3, label=lbl_txt)
                if not bbox_label_added: bbox_label_added = True
    else: print(" - 경계 상자 데이터 없음.")

    # --- 궤적 데이터 플로팅 ---
    if has_trajectory:
        try:
            traj_line = pv.lines_from_points(trajectory_points)
            plotter.add_mesh(traj_line, color='magenta', line_width=5, label='Player Trajectory')
            plotter.add_mesh(pv.PolyData(trajectory_points[0]), color='lime', point_size=10, render_points_as_spheres=True, label='Traj Start') # 라임색 시작
            plotter.add_mesh(pv.PolyData(trajectory_points[-1]), color='blue', point_size=10, render_points_as_spheres=True, label='Traj End')
            print(f" - 궤적 플롯: {trajectory_points.shape[0]} points")
        except Exception as e: print(f"⚠️ 궤적 플로팅 오류: {e}")
    else: print(" - 궤적 데이터 없음 또는 부족.")

    # --- 플로터 설정 및 실행 ---
    plotter.set_background('white'); plotter.show_grid()
    plotter.xlabel = 'World X (m)'; plotter.ylabel = 'World Y (m)'; plotter.zlabel = 'World Z (Height, m)'
    plotter.add_legend(bcolor=(0.9, 0.9, 0.9), border=True, size=(0.15, 0.25)) # 범례 배경색/테두리 추가
    print(" - 카메라 뷰 자동 조정..."); plotter.reset_camera()
    print("✅ PyVista 플롯 생성 완료."); plotter.show()