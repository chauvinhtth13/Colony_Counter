from PyQt6.QtWidgets import (
    QHBoxLayout, QVBoxLayout, QPushButton, QListWidget, QFileDialog,
    QSpinBox, QDoubleSpinBox, QGroupBox, QSizePolicy, QLabel, QTableWidget, QTableWidgetItem,
    QGroupBox, QHeaderView
)
from PyQt6.QtCore import Qt

class LayoutManager:
    def __init__(self, parent):
        self.parent = parent
        self.list_widget = QListWidget()
        self.btn_prev = None
        self.btn_next = None
        self.image_label = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        self.bottom_layout = QHBoxLayout()
        self.table_result = None
        self.file_dialog = QFileDialog()
        self.spinbox_groups = []
        self.left_layout = None

    def create_main_layout(self):
        """Create the main application layout."""
        main_layout = QHBoxLayout()
        main_layout.addLayout(self.create_left_panel(), stretch=0)
        main_layout.addLayout(self.create_center_panel(), stretch=2)
        main_layout.addLayout(self.create_right_panel(), stretch=1)
        return main_layout

    def create_left_panel(self):
        """Create the left panel with buttons and image list."""
        self.left_layout = QVBoxLayout()

        button_configs = [
            ("Select Images", self.parent.load_images, "Load image files"),
            ("Previous Image", self.parent.show_previous_image, "Go to previous image"),
            ("Next Image", self.parent.show_next_image, "Go to next image"),
            ("Detect Lines", self.parent.detect_lines, "Detect colony lines"),
            ("Modify Lines", self.parent.modify_lines, "Adjust existing lines"),
            ("Add Line", self.parent.add_line, "Add a new line"),
            ("Confirm Lines", self.parent.confirm_lines, "Finalize line changes"),
            ("Count Colonies", self.parent.count_colony, "Count colonies in lines"),
            ("Save Image", self.parent.save_image, "Save processed image")
        ]

        self.list_widget.itemClicked.connect(self.parent.show_selected_image)
        nav_layout = QHBoxLayout()

        for text, callback, tooltip in button_configs:
            btn = self.create_button(text, callback, tooltip)
            if text in ["Previous Image", "Next Image"]:
                nav_layout.addWidget(btn)
            else:
                self.left_layout.addWidget(btn)
                if text == "Select Images":
                    self.left_layout.addWidget(self.list_widget)
                    self.left_layout.addLayout(nav_layout)

        self.btn_prev = nav_layout.itemAt(0).widget() if nav_layout.itemAt(0) else self.create_button("Previous Image")
        self.btn_next = nav_layout.itemAt(1).widget() if nav_layout.itemAt(1) else self.create_button("Next Image")
        return self.left_layout

    def create_center_panel(self):
        """Create the center panel with image display and spinbox area."""
        center_layout = QVBoxLayout()
        
        self.image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.image_label.setMinimumSize(
            int(self.parent.screen_width * 0.6), int(self.parent.screen_height * 0.8)
        )
        center_layout.addWidget(self.image_label, stretch=2)
        center_layout.addLayout(self.bottom_layout, stretch=1)
        return center_layout

    def create_right_panel(self):
        """Create the right panel with results table and save button."""
        right_layout = QVBoxLayout()
        
        self.table_result = QTableWidget(0, 5)
        self.table_result.setHorizontalHeaderLabels(["File Name", "Line 1", "Line 2", "Line 3", "Line 4"])
        self.table_result.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        right_layout.addWidget(self.table_result)

        right_layout.addWidget(
            self.create_button("Save Results", self.parent.data_handler.save_to_xlsx, "Save results to Excel")
        )
        return right_layout

    def create_button(self, text, callback=None, tooltip=""):
        """Create a button with optional callback and tooltip."""
        btn = QPushButton(text)
        btn.setMinimumHeight(40)
        if callback:
            btn.clicked.connect(callback)
        else:
            btn.setEnabled(False)
        btn.setToolTip(tooltip)
        return btn

    def create_spinbox_group(self, line_index):
        """Create spinbox group with visible default values."""
        group = QGroupBox(f"Line {line_index + 1}")
        layout = QVBoxLayout()

        configs = [
            ("λ:", 0, 100, self.parent.default_params["Lambda"], 1, False, 
             f"Lambda (default: {self.parent.default_params['Lambda']})"),
            ("Sp:", 0.0, 1.0, self.parent.default_params["Spacing"], 0.01, True, 
             f"Spacing (default: {self.parent.default_params['Spacing']})"),
            ("R:", 0, 100, self.parent.default_params["Min Radius"], 0, False, 
             f"Min Radius (default: {self.parent.default_params['Min Radius']})")
        ]

        for label, min_v, max_v, def_v, step, is_double, tip in configs:
            layout.addLayout(self.create_spinbox(label, min_v, max_v, def_v, step, is_double, tip))
        group.setLayout(layout)
        return group

    def create_spinbox(self, label_text, min_val, max_val, default, step, use_double=True, tooltip=""):
        """Create spinbox showing default value."""
        layout = QHBoxLayout()
        
        label = QLabel(label_text)
        label.setFixedWidth(30)
        layout.addWidget(label)

        spinbox = QDoubleSpinBox() if use_double else QSpinBox()
        if use_double:
            spinbox.setDecimals(2)
        spinbox.setRange(min_val, max_val)
        spinbox.setValue(default)
        spinbox.setSingleStep(step)
        spinbox.setToolTip(tooltip)
        spinbox.setMinimumWidth(80)
        layout.addWidget(spinbox)
        return layout

    def get_all_spinbox_values(self):
        """Retrieve spinbox values, falling back to defaults if needed."""
        values = []
        for group in self.parent.spinbox_groups:
            group_vals = []
            try:
                layout = group.layout()
                for i in range(min(3, layout.count())):
                    h_layout = layout.itemAt(i).layout()
                    if h_layout and h_layout.count() > 1:
                        group_vals.append(h_layout.itemAt(1).widget().value())
            except AttributeError:
                continue
            if len(group_vals) == 3:
                values.append(group_vals)
        
        if not values:
            values = [[self.parent.default_params["Lambda"], 
                       self.parent.default_params["Spacing"], 
                       self.parent.default_params["Min Radius"]]]
        return values

    def clear_spinboxes(self):
        """Clear existing spinbox groups from the bottom layout."""
        while self.bottom_layout.count():
            self.bottom_layout.takeAt(0).widget().deleteLater()
        self.parent.spinbox_groups.clear()

    def update_spinboxes(self, lines_coords):
        """Update spinbox groups based on detected lines."""
        self.clear_spinboxes()
        for i in range(len(lines_coords)):
            group = self.create_spinbox_group(i)
            self.parent.spinbox_groups.append(group)
            self.bottom_layout.addWidget(group)

    def update_navigation_buttons(self):
        """Update the state of navigation buttons."""
        self.btn_prev.setEnabled(self.parent.current_index > 0) # type: ignore
        self.btn_next.setEnabled(self.parent.current_index < len(self.parent.image_paths) - 1) # type: ignore

    def set_button_states(self, modifying=False, adding=False):
        """Set button enabled states based on modification mode."""
        self.btn_prev.setEnabled(not (modifying or adding) and self.parent.current_index > 0)
        self.btn_next.setEnabled(not (modifying or adding) and self.parent.current_index < len(self.parent.image_paths) - 1)
        if self.left_layout is None:
            return
        for i in range(self.left_layout.count()):
            item = self.left_layout.itemAt(i)
            if item.widget() and isinstance(item.widget(), QPushButton):
                text = item.widget().text()
                if text == "Modify Lines":
                    item.widget().setEnabled(not modifying)
                elif text in ["Add Line", "Confirm Lines"]:
                    item.widget().setEnabled(modifying and not adding)  # Active during Modify Lines mode
                elif text not in ["Previous Image", "Next Image"]:
                    item.widget().setEnabled(not (modifying or adding))