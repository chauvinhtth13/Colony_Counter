from PyQt6.QtWidgets import QMessageBox
from core.image_processing import crop_plate, convert_bboxes_to_original
from core.detect_colony_lines import remove_label, find_colonies, detect_colony_lines, sort_lines
from core.count_colony import colony_counting
from ultralytics import YOLO

MODEL_PATH = "models/best.pt"
class ColonyDetector:
    def __init__(self, parent):
        self.parent = parent
        self.model = YOLO(MODEL_PATH)

    def detect_lines(self):
        """Detect colony lines and draw them on the image."""
        if self.parent.original_image is None:
            QMessageBox.warning(self.parent, "Warning", "Please load an image first")
            return
        
        self.parent.cropped_image, self.parent.cropped_radius = crop_plate(self.parent.original_image)
        self.parent.binary_image = remove_label(self.parent.cropped_image)
        #self.parent.binary_image = detect_splitting_line(img_bin)
        centroids = find_colonies(self.parent.cropped_image)
        self.parent.lines_coords = detect_colony_lines(centroids)
        self.parent.view_lines_coords = convert_bboxes_to_original(
            self.parent.lines_coords, self.parent.original_image.shape[:2], self.parent.cropped_radius, bbox_type="rect"
        )
        self.parent.layout_manager.update_spinboxes(self.parent.lines_coords)
        self.parent.image_utils.draw_lines(self.parent.view_lines_coords)
        self.parent.layout_manager.set_button_states(detecting=True)

    def count_colony(self):
        """Count colonies in detected lines and update the results table."""
        if not self.parent.lines_coords:
            QMessageBox.warning(self.parent, "Warning", "Please detect lines first")
            return

        params = self.parent.layout_manager.get_all_spinbox_values()
        number_colony = []
        list_centroids_crop = []
        
        for i, (x_min, y_min, x_max, y_max) in enumerate(self.parent.lines_coords):
            img_bin_line = self.parent.binary_image[y_min:y_max, x_min:x_max]
            img_line = self.parent.cropped_image[y_min:y_max, x_min:x_max]
            p = params[i] if i < len(params) and len(params[i]) == 3 else [
                self.parent.default_params["Lambda"],
                self.parent.default_params["Spacing"],
                self.parent.default_params["Min Radius"]
            ]
            count, centroids = colony_counting(img_bin_line, *p)
            number_colony.append(count)
            centroids_crop = convert_bboxes_to_original(
                centroids, self.parent.binary_image.shape[:2], (x_min, y_min, x_max, y_max), bbox_type="circle"
            )
            list_centroids_crop.extend(centroids_crop)
            self.parent.layout_manager.progress_bar.setValue(int((i + 1) / len(self.parent.lines_coords) * 100))
            
        self.parent.layout_manager.progress_bar.setValue(100)

        self.parent.colony_coords = convert_bboxes_to_original(
            list_centroids_crop, self.parent.original_image.shape[:2], self.parent.cropped_radius, bbox_type="circle"
        )
        self.parent.image_utils.draw_lines(sort_lines(self.parent.view_lines_coords))
        self.parent.image_utils.draw_colony(sort_lines(self.parent.view_lines_coords),number_colony, self.parent.colony_coords)
        self.parent.data_handler.update_table(self.parent.image_paths[self.parent.current_index], number_colony)