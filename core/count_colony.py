import math
import numpy as np
import cv2
from scipy.ndimage import maximum_filter
from scipy import optimize
from pyscipopt import Model, quicksum
from ultralytics import YOLO

def intersection_area(r1, r2, d, epsilon=1e-6):
    """Compute the intersection area of two circles.
    
    Args:
        r1 (float): Radius of the first circle.
        r2 (float): Radius of the second circle.
        d (float): Distance between the centers of the two circles.
    
    Returns:
        float: Area of intersection between the two circles.
    """
    if r1 < 0 or r2 < 0 or d < 0:
        raise ValueError("Radii and distance must be non-negative.")

    if r1 < r2:
        r1, r2 = r2, r1 

    if d >= r1 + r2 - epsilon: 
        return 0.0
    if d <= r1 - r2 + epsilon: 
        return math.pi * r2**2
    if r1 <= epsilon or r2 <= epsilon: 
        return 0.0
        
    r1_sq = r1**2
    r2_sq = r2**2
    d_sq = d**2

    phi_arg = (d_sq + r1_sq - r2_sq) / (2 * d * r1)
    phi_arg = max(-1.0, min(1.0, phi_arg)) 
    phi_half = math.acos(phi_arg) 

    theta_arg = (d_sq + r2_sq - r1_sq) / (2 * d * r2)
    theta_arg = max(-1.0, min(1.0, theta_arg))
    theta_half = math.acos(theta_arg) 

    area_sector1 = r1_sq * phi_half
    area_sector2 = r2_sq * theta_half
    
    term_inside = (-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2)
    term_inside = max(0.0, term_inside) 
    area_kite = 0.5 * math.sqrt(term_inside)

    return area_sector1 + area_sector2 - area_kite

def filter_circles(circles, duplicate_rel_tol=0.15, overlap_thresh=0.85, epsilon=1e-6, min_radius=5):
    """Filter overlapping or duplicate circles based on geometric criteria.
    
    Args:
        circles (list): List of [x, y, radius] for each circle.
        duplicate_rel_tol (float): Relative tolerance for identifying duplicates (default: 0.15).
        overlap_thresh (float): Threshold for excessive overlap (default: 0.8).
        epsilon (float): Small value for numerical stability (default: 1e-3).
        min_radius (int): Minimum radius to keep a circle (default: 10).
    
    Returns:
        list: Filtered list of (x, y, radius) tuples with integer coordinates.
    """
    valid_circles = [c for c in circles if c[2] >= min_radius]
    if not valid_circles:
        return []
        
    circles_np = np.array(valid_circles, dtype=np.float64)
    
    sort_indices = np.argsort(-circles_np[:, 2])
    circles_np = circles_np[sort_indices]
    
    n = len(circles_np)
    keep = np.ones(n, dtype=bool) 

    for i in range(n):
        if not keep[i]: 
            continue
            
        xA, yA, rA = circles_np[i]
        
        candidates_indices = np.arange(i + 1, n)
        if len(candidates_indices) == 0: 
            break 
            
        candidates = circles_np[candidates_indices]
        xB, yB, rB = candidates[:, 0], candidates[:, 1], candidates[:, 2]
        
        dist_sq = (xA - xB)**2 + (yA - yB)**2
        dist = np.sqrt(np.maximum(dist_sq, 0)) 
        
        duplicate = (dist <= duplicate_rel_tol * rA + epsilon) & \
                    (np.abs(rA - rB) <= duplicate_rel_tol * rA + epsilon)

        contained = dist + rB <= rA + epsilon

        intersects_mask = (dist > np.abs(rA - rB) + epsilon) & \
                          (dist < rA + rB - epsilon)
                          
        remove_by_overlap = np.zeros(len(candidates), dtype=bool) 
        
        intersecting_subset_indices = np.where(intersects_mask)[0]
        
        if len(intersecting_subset_indices) > 0:
            rB_intersect = rB[intersecting_subset_indices]
            dist_intersect = dist[intersecting_subset_indices]
            
            inter_areas = np.array([
                intersection_area(rA, rb_i, d_i, epsilon) 
                for rb_i, d_i in zip(rB_intersect, dist_intersect)
            ])
            
            small_areas = math.pi * np.minimum(rA, rB_intersect)**2
            small_areas = np.maximum(small_areas, epsilon) 

            excessive_overlap_subset = (inter_areas / small_areas) >= overlap_thresh
            
            remove_by_overlap[intersecting_subset_indices] = excessive_overlap_subset

        remove_mask = duplicate | contained | remove_by_overlap
        
        keep[candidates_indices] &= ~remove_mask 

    filtered_circles_np = circles_np[keep]
    
    result = [
        (np.int64(round(x)), np.int64(round(y)), np.int64(round(r)))
        for x, y, r in filtered_circles_np
    ]
    
    return result

def find_maxima(img):
    """Identify local maxima in a distance-transformed image to detect potential colony centers.
    
    Args:
        img (np.ndarray): Binary image to analyze.
    
    Returns:
        tuple: (count, maxima, centroids, distance_map) where:
            - count (int): Number of detected maxima.
            - maxima (np.ndarray): Boolean mask of local maxima.
            - centroids (list): List of (x, y, radius) for detected circles.
            - distance_map (np.ndarray): Distance transform of the input image.
    """
    dist = cv2.distanceTransform(img, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    maxima = (dist > 0) & (dist == maximum_filter(dist, footprint=np.ones((3, 3)), mode='constant', cval=-1))
    y, x = np.nonzero(maxima)
    centroids = list(zip(x, y, dist[y, x]))
    return len(centroids), maxima, centroids, dist

def distance(cnt_i, cnt_j, cnt_k):
    """Compute the perpendicular distance from a point to a line defined by two points.
    
    Args:
        cnt_i (list): Point coordinates [x, y] to measure distance from.
        cnt_j (list): Start point of the line [x, y].
        cnt_k (list): End point of the line [x, y].
    
    Returns:
        float: Perpendicular distance from cnt_i to the line cnt_j-cnt_k.
    """
    x1, y1 = cnt_i[0][0], cnt_i[0][1]
    line_start_x, line_start_y = cnt_j[0][0], cnt_j[0][1]
    x2, y2 = cnt_k[0][0], cnt_k[0][1]
    
    array_longi = np.array([x2 - x1, y2 - y1])
    array_trans = np.array([x2 - line_start_x, y2 - line_start_y])
    array_temp = (float(array_trans.dot(array_longi)) / array_longi.dot(array_longi))
    array_temp = array_longi.dot(array_temp)
    return np.sqrt((array_trans - array_temp).dot(array_trans - array_temp))

def clockwise_angle(v1, v2):
    """Compute the clockwise angle between two 2D vectors in radians.
    
    Args:
        v1 (list): First vector [x, y].
        v2 (list): Second vector [x, y].
    
    Returns:
        float: Clockwise angle in radians between v1 and v2.
    """
    x1, y1 = v1
    x2, y2 = v2
    dot = x1 * x2 + y1 * y2
    det = x1 * y2 - x2 * y1
    theta = np.arctan2(det, dot)
    return theta

def calc_R(x, y, xc, yc):
    """Calculate the Euclidean distance of points from a center.
    
    Args:
        x (array-like): X-coordinates of points.
        y (array-like): Y-coordinates of points.
        xc (float): X-coordinate of the center.
        yc (float): Y-coordinate of the center.
    
    Returns:
        np.ndarray: Array of distances from each point to (xc, yc).
    """
    return np.sqrt((x - xc) ** 2 + (y - yc) ** 2)

def f(c, x, y):
    """Compute the algebraic distance for circle fitting.
    
    Args:
        c (tuple): Center coordinates (xc, yc).
        x (array-like): X-coordinates of points.
        y (array-like): Y-coordinates of points.
    
    Returns:
        np.ndarray: Algebraic distances from points to the fitted circle.
    """
    Ri = calc_R(x, y, *c)
    return Ri - Ri.mean()

def leastsq_circle(x, y):
    """Fit a circle to 2D points using least squares optimization.
    
    Args:
        x (array-like): X-coordinates of points.
        y (array-like): Y-coordinates of points.
    
    Returns:
        tuple: (xc, yc, R, residu) where:
            - xc (int): X-coordinate of the circle center.
            - yc (int): Y-coordinate of the circle center.
            - R (float): Radius of the fitted circle.
            - residu (float): Sum of squared residuals.
    """
    x_m = np.mean(x)
    y_m = np.mean(y)
    center_estimate = (x_m, y_m)
    center, _ = optimize.leastsq(f, center_estimate, args=(x, y))
    
    xc, yc = center
    Ri = calc_R(x, y, *center)
    R = Ri.mean()
    residu = np.sum((Ri - R) ** 2)
    return int(xc), int(yc), R, residu

def seg_counting(labels, label_val, lam=38, d_t=0.5, min_radius=10):
    """Segment and count circular colonies in a labeled image region using optimization.
    
    Args:
        labels (np.ndarray): Labeled image where each region has a unique integer value.
        label_val (int): Label value of the region to process.
        lam (float): Penalty term for the optimization objective (default: 38).
        d_t (float): Distance threshold for contour splitting (default: 0.5).
        min_radius (int): Minimum radius for detected circles (default: 10).
    
    Returns:
        tuple: (count, centroids) where:
            - count (int): Number of detected colonies.
            - centroids (list): List of (x, y, radius) tuples for detected circles.
    """
    img_c = (labels == label_val).astype(np.uint8) * 255
    count, m, centroids, dist = find_maxima(np.uint8(img_c))
    contours, _ = cv2.findContours(img_c, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    contours_split = []
    for cnt in contours:
        turning_points = [cnt[0]]
        i, k = 0, 2
        while k < len(cnt):
            d_max, ind = 0, 0
            for j in range(i + 1, k):
                d = distance(cnt[i], cnt[j], cnt[k])
                if d > d_max:
                    d_max = d
                    ind = j
            if d_max > d_t:
                turning_points = np.append(turning_points, [cnt[ind]], axis=0)
                i, k = ind, ind + 2
            else:
                k = k + 1
        turning_points = np.append(turning_points, [cnt[-1]], axis=0)

        angles = []
        concave_points = [turning_points[0]]
        for i in range(len(turning_points) - 2):
            v1 = [
                turning_points[i + 1][0][0] - turning_points[i][0][0],
                turning_points[i + 1][0][1] - turning_points[i][0][1]
            ]
            v2 = [
                turning_points[i + 2][0][0] - turning_points[i + 1][0][0],
                turning_points[i + 2][0][1] - turning_points[i + 1][0][1]
            ]
            angle = clockwise_angle(v1, v2) / math.pi * 180
            angles.append(angle)
            if angle > 0:
                concave_points = np.append(concave_points, [turning_points[i + 1]], axis=0)

        index = []
        k = 0
        for i in range(len(cnt)):
            x, y = concave_points[k][0][0], concave_points[k][0][1]
            x_now, y_now = cnt[i][0][0], cnt[i][0][1]
            if x == x_now and y == y_now:
                index.append(i)
                k = k + 1
            if k == len(concave_points):
                break

        if not contours_split:
            contours_split = [[cnt[index[x]:index[x + 1]]] for x in range(len(index) - 1)]
        else:
            contours_split += [[cnt[index[x]:index[x + 1]]] for x in range(len(index) - 1)]
        contours_split.append([cnt[index[-1]:]])

    for ss in range(len(contours_split)):
        x = [i[0][0] for i in contours_split[ss][0]]
        y = [i[0][1] for i in contours_split[ss][0]]
        if len(x) <= 3:
            continue
        
        xc, yc, _, _ = leastsq_circle(x, y)
        y_max, x_max = m.shape
        if yc < y_max and yc >= 0 and xc < x_max and xc >= 0:
            if img_c[yc][xc] > 0 and dist[yc][xc] > 0:
                count = count + 1 - int(m[yc][xc])
                m[yc][xc] = 1
                centroids.append((np.int64(xc), np.int64(yc), np.int64(dist[yc][xc])))

    d = np.zeros((len(contours_split), count))
    for i in range(len(contours_split)):
        for j in range(count):
            for z in range(len(contours_split[i][0])):
                point = [contours_split[i][0][z][0][0], contours_split[i][0][z][0][1]]
                center = [centroids[j][0], centroids[j][1]]
                d[i][j] += abs(math.dist(point, center) - centroids[j][2])

    model = Model("contour_mapping")
    model.hideOutput()

    x = {
        (i, j): model.addVar(vtype="B", name=f"x_{i}_{j}")
        for i in range(len(contours_split))
        for j in range(count)
    }
    z = {j: model.addVar(vtype="B", name=f"z_{j}") for j in range(count)}

    model.setObjective(
        quicksum(d[i, j] * x[i, j] for i in range(len(contours_split)) for j in range(count)) +
        quicksum(lam * z[j] for j in range(count)),
        "minimize"
    )

    for i in range(len(contours_split)):
        model.addCons(
            quicksum(x[i, j] for j in range(count)) == 1,
            f"contour_{i}"
        )
    for j in range(count):
        model.addCons(
            quicksum(x[i, j] for i in range(len(contours_split))) >= z[j],
            f"z_lower_{j}"
        )
        model.addCons(
            quicksum(x[i, j] for i in range(len(contours_split))) <= len(contours_split) * z[j],
            f"z_upper_{j}"
        )

    model.optimize()

    var_z = [model.getVal(z[j]) for j in range(count)]

    list_centroids = [
        (int(centroids[i][0]), int(centroids[i][1]), int(centroids[i][2]))
        for i in range(count) if var_z[i] > 0.5
    ]
    list_centroids = filter_circles(list_centroids, min_radius)

    return len(list_centroids), list_centroids

def colony_counting(img_line_bin, lam=38, d=0.5, min_radius=0):
    """Count colonies in a binary image by segmenting and analyzing connected components.
    
    Args:
        img_line_bin (np.ndarray): Binary image containing colony regions.
        lam (float): Penalty term for the optimization objective (default: 38).
        d (float): Distance threshold for contour splitting (default: 0.5).
        min_radius (int): Minimum radius for detected circles (default: 10).
    
    Returns:
        tuple: (count, centroids) where:
            - count (int): Total number of detected colonies.
            - centroids (list): List of (x, y, radius) tuples for detected circles.
    """
    counting_result = 0
    list_centroids = []
    num_labels, labels, _, _ = cv2.connectedComponentsWithStats(img_line_bin)

    for j in range(1, num_labels):
        counted_cell_number, centroids = seg_counting(labels, j, lam, d, min_radius)
        counting_result += counted_cell_number
        list_centroids.extend(centroids)

    return counting_result, list_centroids

def count_colony_dl(img_line,model):
    """Count colonies in a binary image using a deep learning model.
    
    Args:
        img_line (np.ndarray): Binary image containing colony regions.
    
    Returns:
        tuple: (count, centroids) where:
            - count (int): Total number of detected colonies.
            - centroids (list): List of (x, y, radius) tuples for detected circles.
    """
    results = model.predict(source=img_line, conf=0.25)
    count = len(results[0].boxes.xywh.tolist())

    centroids = [(int(x), int(y), min(int(w), int(h))/2) for x, y, w, h in results[0].boxes.xywh.cpu().numpy()]

    return count, centroids