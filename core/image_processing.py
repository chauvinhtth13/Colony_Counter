import cv2
import numpy as np
# Define constants for reusability
DEFAULT_BLUR_SIZE = (9, 9)
DEFAULT_KERNEL_SIZE = (3, 3)
DEFAULT_CENTER = (600, 600)
DEFAULT_IMAGE_SIZE = (1200, 1200)

__all__ = ["preprocess_image", "crop_plate", "detect_splitting_line", 
           "convert_bboxes_to_original"]

def preprocess_image(img_org, blur_size=DEFAULT_BLUR_SIZE, kernel_size=DEFAULT_KERNEL_SIZE, sigmaX=2.0):
    """Preprocesses an image to enhance features and reduce noise.
    
    Args:
        img_org (np.ndarray): Input BGR image.
        blur_size (tuple): Gaussian blur kernel size (default: (9, 9)).
        kernel_size (tuple): Morphological kernel size (default: (3, 3)).
        sigmaX (float): Gaussian blur sigma (default: 2.0).
    
    Returns:
        np.ndarray: Denoised binary image.
    """
    img_gray = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY)
    img_blurred = cv2.GaussianBlur(img_gray, blur_size, sigmaX)
    _, img_bin = cv2.threshold(img_blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # kernel = np.ones(kernel_size, np.uint8)
    # img_denoise = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernel, iterations=1)
    
    return img_bin

def crop_plate(img_org, min_radius=400, max_radius=500, center=None):
    """Crops a circular region (e.g., a plate) from an image.
    
    Args:
        img_org (np.ndarray): Input BGR image (default size: 1200x1200).
        min_radius (int): Minimum radius for circle detection (default: 400).
        max_radius (int): Maximum radius for circle detection (default: 500).
        center (tuple): Center of the image (default: (600, 600)).
    
    Returns:
        tuple: (cropped_image, radius) - Cropped region and detected radius.
    """
    height, width = img_org.shape[:2]
    center = center or DEFAULT_CENTER
    
    roi_size = max_radius * 2
    roi_start = center[0] - max_radius
    if roi_start < 0 or roi_start + roi_size > height:
        raise ValueError("ROI exceeds image bounds")

    img_roi = img_org[roi_start:roi_start + roi_size, roi_start:roi_start + roi_size]
    img_gray = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
    
    # Downscale for faster processing (optional, adjust scale as needed)
    scale = 0.5
    small_gray = cv2.resize(img_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    edges = cv2.Canny(small_gray, 50, 150, apertureSize=3, L2gradient=True)
    
    circles = cv2.HoughCircles(
        edges, cv2.HOUGH_GRADIENT, dp=2, minDist=50, param1=80, param2=15,
        minRadius=int(min_radius * scale), maxRadius=int(max_radius * scale)
    )
    
    radius = max_radius
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int") / scale
        distances = np.sqrt((circles[:, 0] - max_radius) ** 2 + (circles[:, 1] - max_radius) ** 2)
        radius = int(circles[distances.argmin()][2])

    crop_start_y = max(0, center[1] - radius)
    crop_end_y = min(height, center[1] + radius)
    crop_start_x = max(0, center[0] - radius)
    crop_end_x = min(width, center[0] + radius)
    img_cropped = img_org[crop_start_y:crop_end_y, crop_start_x:crop_end_x]

    crop_h, crop_w = img_cropped.shape[:2]
    mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
    cv2.circle(mask, (center[0] - crop_start_x, center[1] - crop_start_y), radius, 255, -1)
    
    img_cropped = cv2.bitwise_and(img_cropped, img_cropped, mask=mask)
    img_cropped = cv2.convertScaleAbs(img_cropped, alpha=1.3, beta=-30)

    return img_cropped, radius

def detect_splitting_line(img_bin, thresh_factor=0.8):
    """Segments touching objects in a binary image using distance transform.
    
    Args:
        img_bin (np.ndarray): Binary image.
        thresh_factor (float): Threshold factor for foreground detection (default: 0.7).
    
    Returns:
        np.ndarray: Mask of splitting lines or unknown regions.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    sure_bg = cv2.dilate(img_bin, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(img_bin, cv2.DIST_L2, 5)
    dist_mean = np.mean(dist_transform[dist_transform > 0])
    dist_std = np.std(dist_transform[dist_transform > 0])
    thresh_val = max(thresh_factor * dist_transform.max(), dist_mean + dist_std)
    _, sure_fg = cv2.threshold(dist_transform, thresh_val, 255, 0)
    sure_fg = cv2.erode(sure_fg.astype(np.uint8), kernel, iterations=1)
    
    unknown = cv2.subtract(sure_bg, sure_fg)
    return cv2.bitwise_and(img_bin, img_bin, mask=unknown)

def convert_bboxes_to_original(bbox_list, original_size, crop_info, bbox_type="rect"):
    """Converts bounding box coordinates from cropped to original image space.
    
    Args:
        bbox_list (list): List of bboxes (format depends on bbox_type).
        original_size (tuple): (height, width) of original image.
        crop_info (int or tuple): Crop radius or (x_min, y_min, _, _).
        bbox_type (str): "rect" for (x1, y1, x2, y2), "circle" for (cx, cy, r).
    
    Returns:
        list: Adjusted bboxes in original coordinates.
    """
    bboxes = np.array(bbox_list, dtype=np.float32)
    height, width = original_size
    
    if bbox_type == "rect":
        shift = np.array([original_size[0] // 2 - crop_info, original_size[1] // 2 - crop_info])
        bboxes += np.tile(np.concatenate((shift, shift)), (len(bboxes), 1))
    elif bbox_type == "circle":
        if isinstance(crop_info, int):
            shift = np.array([height // 2 - crop_info, width // 2 - crop_info])
            bboxes[:, :2] += shift
        else:
            x_min, y_min, _, _ = crop_info
            bboxes[:, 0] += x_min
            bboxes[:, 1] += y_min
        bboxes[:, 0] = np.clip(bboxes[:, 0], bboxes[:, 2], width - bboxes[:, 2])
        bboxes[:, 1] = np.clip(bboxes[:, 1], bboxes[:, 2], height - bboxes[:, 2])
    
    return [tuple(bbox) for bbox in bboxes.astype(int)]