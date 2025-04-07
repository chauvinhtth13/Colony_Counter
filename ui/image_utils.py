from PyQt6.QtGui import QPixmap, QPainter, QPen, QImage, QBrush, QFont
from PyQt6.QtCore import Qt

class ImageUtils:
    def __init__(self, parent):
        self.parent = parent

    def cv2_to_qimage(self, cv2_image):
        """Convert a cv2 image to a QImage."""
        h, w, c = cv2_image.shape
        return QImage(cv2_image.data, w, h, w * c, QImage.Format.Format_BGR888)

    def display_image(self, image):
        """Display a cv2 image in the QLabel."""
        pixmap = QPixmap.fromImage(self.cv2_to_qimage(image))
        self.parent.layout_manager.image_label.setPixmap(pixmap.scaled(
            self.parent.layout_manager.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation
        ))

    def draw_lines(self, lines_coords):
        """Draw detected lines on the original image without handles."""
        image = self.parent.original_image.copy()
        pixmap = QPixmap.fromImage(self.cv2_to_qimage(image))
        
        painter = QPainter(pixmap)
        pen = QPen(Qt.GlobalColor.red, 2)
        painter.setPen(pen)
        for x1, y1, x2, y2 in lines_coords:
            painter.drawRect(x1, y1, x2 - x1, y2 - y1)
        
        painter.end()
        self.parent.layout_manager.image_label.setPixmap(pixmap.scaled(
            self.parent.layout_manager.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation
        ))

    def draw_text_lines(self, lines_coords, number_colony=None):
        """
        Draw lines with text labels on the original image.
        
        Args:
            lines_coords (list): List of tuples (x1, y1, x2, y2) representing bounding boxes.
        """
        pixmap = self.parent.layout_manager.image_label.pixmap()
        
        painter = QPainter(pixmap)
        pen = QPen(Qt.GlobalColor.blue, 2)
        painter.setPen(pen)
        
        # Set font for text
        font = QFont("Arial", 18, QFont.Weight.Bold)
        painter.setFont(font)
        
        # Draw each line with a text label
        for i, (x1, y1, x2, y2) in enumerate(lines_coords, 1):

            # Draw text (e.g., "Line 1") above the top-left corner
            text = f"Line {i}: {number_colony[i-1]}" if number_colony else f"Line {i}"
            text_y = y1 - 30  # Position text slightly above the box
            painter.drawText(x1, text_y, text)
        
        painter.end()
        self.parent.layout_manager.image_label.setPixmap(pixmap.scaled(
            self.parent.layout_manager.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation
        ))


    def draw_colony(self, colony_coords):
        """Draw colony center points as small circles on the current pixmap."""
        pixmap = self.parent.layout_manager.image_label.pixmap()
        if not pixmap:
            image = self.parent.original_image.copy()
            pixmap = QPixmap.fromImage(self.cv2_to_qimage(image))

        orig_h, orig_w = self.parent.original_image.shape[:2]
        pixmap_h, pixmap_w = pixmap.height(), pixmap.width()

        aspect_ratio_orig = orig_w / orig_h
        aspect_ratio_pixmap = pixmap_w / pixmap_h
        if aspect_ratio_orig > aspect_ratio_pixmap:
            scale_factor = pixmap_w / orig_w
            offset_y = (pixmap_h - orig_h * scale_factor) / 2
            offset_x = 0
        else:
            scale_factor = pixmap_h / orig_h
            offset_x = (pixmap_w - orig_w * scale_factor) / 2
            offset_y = 0

        painter = QPainter(pixmap)
        pen = QPen(Qt.GlobalColor.blue, 2)
        painter.setPen(pen)
        painter.setBrush(QBrush(Qt.GlobalColor.blue, Qt.BrushStyle.SolidPattern))
        point_diameter = 6
        for cx, cy, _ in colony_coords:
            scaled_x = int(cx * scale_factor + offset_x)
            scaled_y = int(cy * scale_factor + offset_y)
            painter.drawEllipse(scaled_x - point_diameter // 2, 
                               scaled_y - point_diameter // 2, 
                               point_diameter, point_diameter)
        
        painter.end()
        self.parent.layout_manager.image_label.setPixmap(pixmap.scaled(
            self.parent.layout_manager.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation
        ))

    def get_current_pixmap(self):
        """Return the current pixmap from the image label."""
        return self.parent.layout_manager.image_label.pixmap()

    def map_to_pixmap_coords(self, coords_list):
        """Map coordinates from original image space to pixmap space."""
        pixmap = self.parent.layout_manager.image_label.pixmap()
        if not pixmap or self.parent.original_image is None:
            return coords_list
        
        orig_h, orig_w = self.parent.original_image.shape[:2]
        pixmap_h, pixmap_w = pixmap.height(), pixmap.width()

        aspect_ratio_orig = orig_w / orig_h
        aspect_ratio_pixmap = pixmap_w / pixmap_h
        if aspect_ratio_orig > aspect_ratio_pixmap:
            scale_factor = pixmap_w / orig_w
            offset_y = (pixmap_h - orig_h * scale_factor) / 2
            offset_x = 0
        else:
            scale_factor = pixmap_h / orig_h
            offset_x = (pixmap_w - orig_w * scale_factor) / 2
            offset_y = 0

        pixmap_coords = []
        for x1, y1, x2, y2 in coords_list:
            pixmap_x1 = int(x1 * scale_factor + offset_x)
            pixmap_y1 = int(y1 * scale_factor + offset_y)
            pixmap_x2 = int(x2 * scale_factor + offset_x)
            pixmap_y2 = int(y2 * scale_factor + offset_y)
            pixmap_coords.append((pixmap_x1, pixmap_y1, pixmap_x2, pixmap_y2))
        return pixmap_coords

    def map_pixmap_to_original_coords(self, pixmap_coords):
        """Map coordinates from pixmap space to original image space."""
        pixmap = self.parent.layout_manager.image_label.pixmap()
        if not pixmap or self.parent.original_image is None:
            return pixmap_coords
        
        orig_h, orig_w = self.parent.original_image.shape[:2]
        pixmap_h, pixmap_w = pixmap.height(), pixmap.width()

        aspect_ratio_orig = orig_w / orig_h
        aspect_ratio_pixmap = pixmap_w / pixmap_h
        if aspect_ratio_orig > aspect_ratio_pixmap:
            scale_factor = pixmap_w / orig_w
            offset_y = (pixmap_h - orig_h * scale_factor) / 2
            offset_x = 0
        else:
            scale_factor = pixmap_h / orig_h
            offset_x = (pixmap_w - orig_w * scale_factor) / 2
            offset_y = 0

        if isinstance(pixmap_coords, tuple) and len(pixmap_coords) == 4:
            x1, y1, x2, y2 = pixmap_coords
            orig_x1 = int((x1 - offset_x) / scale_factor)
            orig_y1 = int((y1 - offset_y) / scale_factor)
            orig_x2 = int((x2 - offset_x) / scale_factor)
            orig_y2 = int((y2 - offset_y) / scale_factor)
            return (orig_x1, orig_y1, orig_x2, orig_y2)
        else:
            x, y = pixmap_coords
            orig_x = int((x - offset_x) / scale_factor)
            orig_y = int((y - offset_y) / scale_factor)
            return (orig_x, orig_y)

    def map_pixmap_delta_to_original(self, dx, dy):
        """Map delta (movement) from pixmap space to original image space."""
        pixmap = self.parent.layout_manager.image_label.pixmap()
        if not pixmap or self.parent.original_image is None:
            return dx, dy
        
        orig_h, orig_w = self.parent.original_image.shape[:2]
        pixmap_h, pixmap_w = pixmap.height(), pixmap.width()

        aspect_ratio_orig = orig_w / orig_h
        aspect_ratio_pixmap = pixmap_w / pixmap_h
        if aspect_ratio_orig > aspect_ratio_pixmap:
            scale_factor = pixmap_w / orig_w
        else:
            scale_factor = pixmap_h / orig_h

        orig_dx = int(dx / scale_factor)
        orig_dy = int(dy / scale_factor)
        return orig_dx, orig_dy