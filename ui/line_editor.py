from PyQt6.QtWidgets import QMessageBox
from PyQt6.QtCore import Qt, QPoint

class LineEditor:
    def __init__(self, parent):
        """Initialize the LineEditor with a reference to the parent widget."""
        self.parent = parent
        self.modifying_lines = False
        self.adding_line = False
        self.selected_line_idx = -1
        self.drag_start = None
        self.drag_corner = None  # 0: top-left, 1: bottom-right, None: move whole box
        self.new_line_start = None

    def reset_line_states(self):
        """Reset all line modification states and clear coordinates."""
        self.modifying_lines = False
        self.adding_line = False
        self.selected_line_idx = -1
        self.drag_start = None
        self.drag_corner = None
        self.new_line_start = None
        self.parent.view_lines_coords = []
        self.parent.colony_coords = []
        self.parent.lines_coords = []  # Reset lines_coords too
        self.parent.image_utils.draw_lines(self.parent.view_lines_coords)

    def modify_lines(self):
        """Enable line modification mode and activate relevant buttons."""
        if not self.parent.lines_coords:
            QMessageBox.warning(self.parent, "Warning", "Please detect lines first")
            return
        self.modifying_lines = True
        self.adding_line = False
        self.parent.layout_manager.clear_spinboxes()
        self.parent.layout_manager.set_button_states(modifying=True)
        self.parent.image_utils.draw_lines(self.parent.view_lines_coords)
        QMessageBox.information(self.parent, "Modify Lines", 
                                "Click and drag corners to resize or inside to move lines. "
                                "Right-click inside a bbox to remove it. "
                                "Use 'Add Line' to add new lines. Click 'Confirm Lines' when done.")

    def add_line(self):
        """Add a predefined rectangle to view_lines_coords in modification mode, if fewer than 4 lines exist."""
        # Check if modification mode is enabled
        if not self.modifying_lines:
            QMessageBox.warning(self.parent, "Warning", "Please click 'Modify Lines' first")
            return
        
        # Check if an image is loaded
        if self.parent.original_image is None:
            QMessageBox.warning(self.parent, "Warning", "Please load an image first")
            return
        
        # Enforce the limit of 4 lines
        if len(self.parent.view_lines_coords) >= 4:
            QMessageBox.warning(self.parent, "Cannot Add Line", 
                                "Adding more lines is not suitable. Maximum limit of 4 reached.")
            return
        
        # Determine ymin and ymax from existing lines or use defaults
        if self.parent.view_lines_coords:
            y_coords = [y for coord in self.parent.view_lines_coords for y in (coord[1], coord[3])]
            ymin = min(y_coords)
            ymax = max(y_coords)
        else:
            orig_h = self.parent.original_image.shape[0]
            ymin = 0
            ymax = orig_h // 2  # Default to half the image height
        
        # Add the new rectangle
        new_rect = (0, ymin, 20, ymax)
        self.parent.view_lines_coords.append(new_rect)
        
        # Update the display
        self.parent.image_utils.draw_lines(self.parent.view_lines_coords)

    def confirm_lines(self):
        """Confirm and finalize line modifications or additions."""
        if not (self.modifying_lines or self.adding_line):
            return
        
        self.modifying_lines = False
        self.adding_line = False
        self.selected_line_idx = -1
        self.drag_start = None
        self.drag_corner = None
        self.new_line_start = None
        self.parent.view_lines_coords = [coord for coord in self.parent.view_lines_coords]
        self.parent.lines_coords = [self.map_to_cropped_coords(coord) for coord in self.parent.view_lines_coords]
        print(self.parent.lines_coords)
        self.parent.layout_manager.update_spinboxes(self.parent.lines_coords)
        self.parent.layout_manager.set_button_states(modifying=False, adding=False)
        self.parent.image_utils.draw_lines(self.parent.view_lines_coords)
        QMessageBox.information(self.parent, "Lines Confirmed", "Line modifications confirmed.")

    def mouse_press_event(self, event):
        """Handle mouse press events on the image label."""
        if not (self.modifying_lines or self.adding_line) or self.parent.original_image is None:
            return
        
        pos = event.pos()
        if event.button() == Qt.MouseButton.RightButton and self.modifying_lines and self.drag_start is None:
            self.handle_right_click(pos)
        elif event.button() == Qt.MouseButton.LeftButton:
            if self.modifying_lines and not self.adding_line:
                self.handle_modify_press(pos)
            elif self.adding_line:
                self.handle_add_press(pos)

    def handle_right_click(self, pos):
        """Handle right-click to remove a bbox."""
        label = self.parent.layout_manager.image_label
        pixmap = label.pixmap()
        if not pixmap:
            return

        label_size = label.size()
        pixmap_size = pixmap.size()
        offset_x = (label_size.width() - pixmap_size.width()) / 2
        offset_y = (label_size.height() - pixmap_size.height()) / 2
        pixmap_x = pos.x() - offset_x
        pixmap_y = pos.y() - offset_y

        if 0 <= pixmap_x < pixmap_size.width() and 0 <= pixmap_y < pixmap_size.height():
            pixmap_coords = self.parent.image_utils.map_to_pixmap_coords(self.parent.view_lines_coords)
            for i, (x1, y1, x2, y2) in enumerate(pixmap_coords):
                if x1 < pixmap_x < x2 and y1 < pixmap_y < y2:
                    reply = QMessageBox.question(self.parent, "Remove Bbox", "Do you want to remove this bbox?",
                                                 QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                                 QMessageBox.StandardButton.No)
                    if reply == QMessageBox.StandardButton.Yes:
                        del self.parent.view_lines_coords[i]
                        self.parent.image_utils.draw_lines(self.parent.view_lines_coords)
                    break

    def handle_modify_press(self, pos):
        """Handle mouse press for modifying lines (resize or move) in pixmap space."""
        label = self.parent.layout_manager.image_label
        pixmap = label.pixmap()
        if not pixmap:
            return

        label_size = label.size()
        pixmap_size = pixmap.size()
        offset_x = (label_size.width() - pixmap_size.width()) / 2
        offset_y = (label_size.height() - pixmap_size.height()) / 2
        pixmap_x = pos.x() - offset_x
        pixmap_y = pos.y() - offset_y

        if 0 <= pixmap_x < pixmap_size.width() and 0 <= pixmap_y < pixmap_size.height():
            pixmap_coords = self.parent.image_utils.map_to_pixmap_coords(self.parent.view_lines_coords)
            for i, (x1, y1, x2, y2) in enumerate(pixmap_coords):
                # Check corners for resizing with increased threshold
                if abs(pixmap_x - x1) < 10 and abs(pixmap_y - y1) < 10:  # Top-left corner
                    self.selected_line_idx = i
                    self.drag_start = QPoint(int(pixmap_x), int(pixmap_y))
                    self.drag_corner = 0
                    break
                elif abs(pixmap_x - x2) < 10 and abs(pixmap_y - y2) < 10:  # Bottom-right corner
                    self.selected_line_idx = i
                    self.drag_start = QPoint(int(pixmap_x), int(pixmap_y))
                    self.drag_corner = 1
                    break
                elif x1 < pixmap_x < x2 and y1 < pixmap_y < y2:
                    self.selected_line_idx = i
                    self.drag_start = QPoint(int(pixmap_x), int(pixmap_y))
                    self.drag_corner = None  # Move whole box
                    break

    def handle_add_press(self, pos):
        """Handle mouse press for adding a new line in pixmap space (not used with new add_line)."""
        pass  # Kept for compatibility but not used

    def mouse_move_event(self, event):
        """Handle mouse move events for dragging lines in pixmap space."""
        if not self.modifying_lines or self.selected_line_idx == -1 or not self.drag_start or self.adding_line:
            return
        
        pos = event.pos()
        label = self.parent.layout_manager.image_label
        pixmap = label.pixmap()
        if not pixmap:
            return

        label_size = label.size()
        pixmap_size = pixmap.size()
        offset_x = (label_size.width() - pixmap_size.width()) / 2
        offset_y = (label_size.height() - pixmap_size.height()) / 2
        pixmap_x = pos.x() - offset_x
        pixmap_y = pos.y() - offset_y

        if 0 <= pixmap_x < pixmap_size.width() and 0 <= pixmap_y < pixmap_size.height():
            x1, y1, x2, y2 = self.parent.view_lines_coords[self.selected_line_idx]
            
            if self.drag_corner is None:  # Move whole box
                dx = pixmap_x - self.drag_start.x()
                dy = pixmap_y - self.drag_start.y()
                dx_orig, dy_orig = self.parent.image_utils.map_pixmap_delta_to_original(dx, dy)
                self.parent.view_lines_coords[self.selected_line_idx] = (x1 + dx_orig, y1 + dy_orig, x2 + dx_orig, y2 + dy_orig)
                self.drag_start = QPoint(int(pixmap_x), int(pixmap_y))
            elif self.drag_corner == 0:  # Top-left
                mapped_coords = self.parent.image_utils.map_pixmap_to_original_coords((pixmap_x, pixmap_y))
                new_x1, new_y1 = mapped_coords[:2] if len(mapped_coords) >= 2 else (0, 0)
                self.parent.view_lines_coords[self.selected_line_idx] = (new_x1, new_y1, x2, y2)
            else:  # Bottom-right
                mapped_coords = self.parent.image_utils.map_pixmap_to_original_coords((pixmap_x, pixmap_y))
                new_x2, new_y2 = mapped_coords if len(mapped_coords) == 2 else (x2, y2)
                self.parent.view_lines_coords[self.selected_line_idx] = (x1, y1, new_x2, new_y2)
            
            x1, y1, x2, y2 = self.parent.view_lines_coords[self.selected_line_idx]
            self.parent.view_lines_coords[self.selected_line_idx] = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
            self.parent.image_utils.draw_lines(self.parent.view_lines_coords)

    def mouse_release_event(self, event):
        """Handle mouse release events to finalize modifications."""
        if self.modifying_lines and self.selected_line_idx != -1 and not self.adding_line:
            # Update the entire lines_coords list to avoid index errors
            self.parent.lines_coords = [self.map_to_cropped_coords(coord) for coord in self.parent.view_lines_coords]
            self.drag_start = None
            self.selected_line_idx = -1  # Reset selection
            self.parent.image_utils.draw_lines(self.parent.view_lines_coords)

    def map_to_cropped_coords(self, view_coords):
        """Map coordinates from original image space to cropped image space."""
        x1, y1, x2, y2 = view_coords
        orig_h, orig_w = self.parent.original_image.shape[:2]
        center = (orig_w // 2, orig_h // 2)
        crop_start_x = max(0, center[0] - self.parent.cropped_radius)
        crop_start_y = max(0, center[1] - self.parent.cropped_radius)
        return (x1 - crop_start_x, y1 - crop_start_y, x2 - crop_start_x, y2 - crop_start_y)