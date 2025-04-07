from PyQt6.QtWidgets import QWidget, QMessageBox
from PyQt6.QtGui import QGuiApplication
import cv2
import os

from .layout_manager import LayoutManager
from .image_utils import ImageUtils
from .data_handler import DataHandler
from .colony_detector import ColonyDetector
from .line_editor import LineEditor

class ColonyCounterApp(QWidget):
    def __init__(self):
        super().__init__()
        self.image_paths = []
        self.current_index = -1
        self.original_image = None
        self.cropped_image = None
        self.binary_image = None
        self.cropped_radius = 0
        self.lines_coords = []
        self.view_lines_coords = []
        self.colony_coords = []
        self.spinbox_groups = []
        self.default_params = {"Lambda": 38, "Spacing": 0.5, "Min Radius": 0}

        screen = QGuiApplication.primaryScreen()
        self.screen_geometry = screen.availableGeometry()
        self.screen_width = self.screen_geometry.width()
        self.screen_height = self.screen_geometry.height()

        self.layout_manager = LayoutManager(self)
        self.image_utils = ImageUtils(self)
        self.data_handler = DataHandler(self)
        self.colony_detector = ColonyDetector(self)
        self.line_editor = LineEditor(self)
        self.initUI()

    def initUI(self):
        """Initialize the main UI layout."""
        self.setWindowTitle("Colony Counter")
        self.showMaximized()
        self.setLayout(self.layout_manager.create_main_layout())
        self.setGeometry(self.screen_geometry)

        self.layout_manager.image_label.setMouseTracking(True)
        self.layout_manager.image_label.mousePressEvent = self.line_editor.mouse_press_event
        self.layout_manager.image_label.mouseMoveEvent = self.line_editor.mouse_move_event
        self.layout_manager.image_label.mouseReleaseEvent = self.line_editor.mouse_release_event

    def load_images(self):
        """Load image files from a file dialog."""
        paths, _ = self.layout_manager.file_dialog.getOpenFileNames(
            self, "Select Images", "", "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if not paths:
            return
        
        self.image_paths = paths
        self.layout_manager.list_widget.clear()
        self.layout_manager.list_widget.addItems([os.path.basename(p) for p in paths])
        self.current_index = 0
        self.show_image()

    def show_selected_image(self, item):
        """Display the image selected from the list."""
        self.current_index = self.layout_manager.list_widget.row(item)
        self.show_image()

    def show_image(self):
        """Show the current image in the label."""
        if not (0 <= self.current_index < len(self.image_paths)):
            return
        
        self.original_image = cv2.imread(self.image_paths[self.current_index])
        if self.original_image is None:
            QMessageBox.warning(self, "Error", "Failed to load image")
            return
        
        self.image_utils.display_image(self.original_image)
        self.layout_manager.clear_spinboxes()
        self.layout_manager.update_navigation_buttons()
        self.line_editor.reset_line_states()

    def show_previous_image(self):
        """Navigate to the previous image."""
        if self.current_index > 0:
            self.current_index -= 1
            self.show_image()
            self.layout_manager.list_widget.setCurrentRow(self.current_index)

    def show_next_image(self):
        """Navigate to the next image."""
        if self.current_index < len(self.image_paths) - 1:
            self.current_index += 1
            self.show_image()
            self.layout_manager.list_widget.setCurrentRow(self.current_index)

    def detect_lines(self):
        """Detect colony lines and draw them on the image."""
        self.colony_detector.detect_lines()

    def count_colony(self):
        """Count colonies in detected lines and update the results table."""
        self.colony_detector.count_colony()

    def save_image(self):
        """Save the processed image with lines and colony points to a file."""
        if self.original_image is None:
            QMessageBox.warning(self, "Warning", "Please load an image first")
            return
        
        if not self.lines_coords:
            QMessageBox.warning(self, "Warning", "Please detect lines before saving")
            return
        
        if not self.colony_coords:
            QMessageBox.warning(self, "Warning", "Please count colony before saving")
            return

        path, _ = self.layout_manager.file_dialog.getSaveFileName(
            self, "Save Processed Image", "", "PNG (*.png);;JPEG (*.jpg *.jpeg)"
        )
        if not path:
            return

        pixmap = self.image_utils.get_current_pixmap()
        if pixmap and pixmap.save(path):
            QMessageBox.information(self, "Success", f"Image saved to {path}")
        else:
            QMessageBox.warning(self, "Error", "Failed to save image")

    def modify_lines(self):
        """Enable line modification mode and activate Add/Confirm buttons."""
        self.line_editor.modify_lines()

    def add_line(self):
        """Enable line addition mode within modification mode."""
        self.line_editor.add_line()

    def confirm_lines(self):
        """Confirm and finalize line modifications or additions."""
        self.line_editor.confirm_lines()