import numpy as np

def preprocess_image(img_org):
    """Preprocesses an image to enhance features and reduce noise.
    ToDo: Add more details about the preprocessing steps.
    """
    
    return img_org

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