import cv2
import numpy as np
import math
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from .image_processing import preprocess_image

def calculate_angle(p1, p2, p3):
    """Calculate the angle in degrees between three points with p2 as the vertex.
    
    Args:
        p1 (array-like): First point coordinates (x, y).
        p2 (array-like): Vertex point coordinates (x, y).
        p3 (array-like): Third point coordinates (x, y).
    
    Returns:
        float: Angle in degrees between the vectors p1-p2 and p2-p3.
    """
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0)) * 180 / np.pi
    return angle

def remove_label(
    img_org, 
    min_area=10000, 
    max_area=40000, 
    min_solidity=0.8, 
    max_circularity=0.75, 
    approx_epsilon=0.02, 
    angle_threshold_min=10, 
    angle_threshold_max=170,
    min_aspect_ratio=1.0, 
    max_aspect_ratio=2.5
):
    """
    Remove label-like contours from an image based on geometric properties.

    Args:
        img_org (np.ndarray): Input BGR image.
        min_area (float): Minimum contour area.
        max_area (float): Maximum contour area.
        min_solidity (float): Minimum solidity.
        max_circularity (float): Maximum circularity.
        approx_epsilon (float): Polygon approximation factor.
        angle_threshold_min (float): Minimum vertex angle.
        angle_threshold_max (float): Maximum vertex angle.
        min_aspect_ratio (float): Minimum aspect ratio of bounding box.
        max_aspect_ratio (float): Maximum aspect ratio of bounding box.

    Returns:
        np.ndarray: Binary image with label-like contours removed.
    """
    img_denoise = preprocess_image(img_org)
    contours, _ = cv2.findContours(img_denoise, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_filtered = np.zeros_like(img_denoise)

    for contour in contours:
        area = cv2.contourArea(contour)
        if not (min_area <= math.floor(area) <= max_area):
            continue

        rect = cv2.minAreaRect(contour)
        width, height = rect[1]
        if width == 0 or height == 0:
            continue

        aspect_ratio = max(width, height) / min(width, height)
        
        if not (min_aspect_ratio <= aspect_ratio <= max_aspect_ratio):
            continue

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, approx_epsilon * perimeter, True)
        vertices = approx.reshape(-1, 2)

        # Check angles only if sufficient vertices
        if len(vertices) >= 3:
            angles = [
                calculate_angle(vertices[i], vertices[(i + 1) % len(vertices)], vertices[(i + 2) % len(vertices)])
                for i in range(len(vertices))
            ]
            if not all(angle_threshold_min <= angle <= angle_threshold_max for angle in angles):
                continue
        
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            continue

        solidity = area / hull_area
        circularity = (4 * np.pi * area) / (perimeter ** 2)

        if solidity >= min_solidity and round(circularity,2) <= max_circularity:
            cv2.drawContours(mask_filtered, [contour], -1, 255, cv2.FILLED)

    mask_inv = cv2.bitwise_not(mask_filtered)
    return cv2.bitwise_and(img_denoise, img_denoise, mask=mask_inv)

def find_colonies(img_org):
    """Detect colonies in an image and return their centroids and bounding boxes.
    
    Args:
        img_org (np.ndarray): Input BGR image.
    
    Returns:
        np.ndarray or None: Array of [cx, cy, radius, x, y, x+w, y+h] for each colony, or None if no colonies found.
    """
    img_bin = remove_label(img_org)
    contours, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centroids_colony = []

    for contour in contours:
        area = cv2.contourArea(contour)
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            x, y, w, h = cv2.boundingRect(contour)
            radius = int(math.sqrt(area / np.pi))
            centroids_colony.append([cx, cy, radius, x, y, x + w, y + h])

    return np.array(centroids_colony) if centroids_colony else None

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