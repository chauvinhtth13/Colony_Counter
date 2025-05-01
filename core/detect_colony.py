import sys
import os
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

def resource_path(relative_path):
    """Lấy đường dẫn tuyệt đối cho tài nguyên, hoạt động cho cả dev và PyInstaller"""
    if hasattr(sys, '_MEIPASS'):
        # Đường dẫn tới thư mục tạm khi chạy file thực thi
        base_path = sys._MEIPASS
    else:
        # Đường dẫn gốc trong môi trường phát triển
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

def colony_counting_yolo(img_line, model, conf=0.5):

    results = model.predict(source=img_line, conf=conf, verbose=False)
    coords = results[0].boxes.xywh.cpu().numpy()

    num_detections = coords.shape[0]
    if num_detections == 0:
        # Trả về số lượng 0 và danh sách rỗng theo logic ban đầu
        return 0, []
    
    x_c = coords[:, 0]
    y_c = coords[:, 1]
    w = coords[:, 2]
    h = coords[:, 3]

    w_half = w / 2.0
    h_half = h / 2.0

    max_half_dim = np.maximum(w_half, h_half)

    x_min = x_c - w_half # Tọa độ x cạnh trái
    y_min = y_c - h_half # Tọa độ y cạnh trên
    x_max = x_c + w_half # Tọa độ x cạnh phải
    y_max = y_c + h_half # Tọa độ y cạnh dưới

    list_centroids = np.column_stack((
        x_c,
        y_c,
        max_half_dim,
        x_min,
        y_min,
        x_max,
        y_max
    ))
    
    return num_detections, list_centroids

def sort_lines(lines):
    """Sort lines based on their x-coordinates.
    
    Args:
        lines (list): List of (x_top, y_top, x_bottom, y_bottom) tuples.
    
    Returns:
        list: Sorted list of lines based on x-coordinates.
    """
    return sorted(lines, key=lambda line: line[0])

def detect_colony_lines(centroids_colony, silhouette_weight=1.0, bic_weight=0.1):
    """Cluster colonies into vertical lines based on x-coordinates and normalize y-boundaries.
    
    Args:
        centroids_colony (np.ndarray): Array of [cx, cy, radius, x, y, x+w, y+h] for each colony.
    
    Returns:
        list: Sorted list of (x_top, y_top, x_bottom, y_bottom) tuples representing colony lines.
    """
    if centroids_colony.size == 0:
        return []

    coords = centroids_colony[:, 0:1]
    n_samples = len(coords)

    best_score = float('-inf')
    best_labels = np.zeros(n_samples, dtype=int)

    max_clusters = min(n_samples, 5)
    for k in range(1, max_clusters):
        gmm = GaussianMixture(
            n_components=k,
            random_state=42,
            covariance_type="full",
            n_init=5,
            reg_covar=1e-5
        )
        labels = gmm.fit_predict(coords)
        bic = gmm.bic(coords)

        if len(np.unique(labels)) > 1:
            sil_score = silhouette_score(coords, labels)
        else:
            sil_score = 0  # no silhouette for k=1
        
        combined_score = (silhouette_weight * sil_score) - (bic_weight * bic)

        if combined_score > best_score:
            best_score = combined_score
            best_labels = labels

    line_coords = []
    for label in np.unique(best_labels):
        cluster_points = centroids_colony[best_labels == label]
        x_min, y_min = np.min(cluster_points[:, 3:5], axis=0)
        x_max, y_max = np.max(cluster_points[:, 5:7], axis=0)
        line_coords.append((int(x_min), int(y_min), int(x_max), int(y_max)))

    if not line_coords:
        return []

    # Normalize y-boundaries across all lines
    y_min = min(line[1] for line in line_coords)
    y_max = max(line[3] for line in line_coords)
    line_coords = [(x_min, y_min, x_max, y_max) for x_min, _, x_max, _ in line_coords]

    return sort_lines(line_coords)