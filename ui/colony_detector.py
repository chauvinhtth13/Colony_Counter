from PyQt6.QtWidgets import QMessageBox
from core.image_processing import convert_bboxes_to_original
from core.detect_colony import detect_colony_lines, sort_lines, colony_counting_yolo, resource_path
from ultralytics import YOLO

MODEL_PATH = resource_path("models/best.pt")
class ColonyDetector:
    def __init__(self, parent):
        self.parent = parent
        self.model = YOLO(MODEL_PATH)

    def detect_lines(self):
        """Detect colony lines and draw them on the image."""
        if self.parent.original_image is None:
            QMessageBox.warning(self.parent, "Warning", "Please load an image first")
            return
        
        _, coords = colony_counting_yolo(self.parent.original_image, self.model, 0.5)

        self.parent.lines_coords = detect_colony_lines(coords)
        self.parent.view_lines_coords = self.parent.lines_coords
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

            img_line = self.parent.original_image[y_min:y_max, x_min:x_max]

            p = params[i] if i < len(params) and len(params[i]) == 1 else [
                self.parent.default_params["Confidence"]
            ]
            count, centroids = colony_counting_yolo(img_line, self.model, *p)

            number_colony.append(count)

            centroids_crop = convert_bboxes_to_original(
                centroids, self.parent.original_image.shape[:2], (x_min, y_min, x_max, y_max), bbox_type="circle"
            )
            list_centroids_crop.extend(centroids_crop)
            self.parent.layout_manager.progress_bar.setValue(int((i + 1) / len(self.parent.lines_coords) * 100))
            
        self.parent.layout_manager.progress_bar.setValue(100)

        self.parent.colony_coords = list_centroids_crop

        self.parent.image_utils.draw_lines(sort_lines(self.parent.view_lines_coords))
        self.parent.image_utils.draw_colony(sort_lines(self.parent.view_lines_coords),number_colony, self.parent.colony_coords)
        self.parent.data_handler.update_table(self.parent.image_paths[self.parent.current_index], number_colony)