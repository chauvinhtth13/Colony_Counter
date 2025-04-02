from PyQt6.QtWidgets import QTableWidgetItem, QMessageBox
import polars as pl
import os

class DataHandler:
    def __init__(self, parent):
        self.parent = parent

    def update_table(self, image_path, counts):
        """Update the results table with colony counts."""
        filename = os.path.basename(image_path)
        table = self.parent.layout_manager.table_result
        
        for row in range(table.rowCount()):
            if table.item(row, 0) and table.item(row, 0).text() == filename:
                table.removeRow(row)
                break  # Exit after removing the row, as there should only be one match

        # Add new row with updated counts
        row = table.rowCount()
        table.insertRow(row)
        table.setItem(row, 0, QTableWidgetItem(filename))
        for col, count in enumerate(counts[:4], 1):  # Limit to 4 lines as per table headers
            table.setItem(row, col, QTableWidgetItem(str(count)))

    def save_to_xlsx(self):
        """Save table results to an Excel file."""
        table = self.parent.layout_manager.table_result
        if not table.rowCount():
            QMessageBox.warning(self.parent, "Warning", "No results to save")
            return
        
        path, _ = self.parent.layout_manager.file_dialog.getSaveFileName(
            self.parent, "Save Results", "", "Excel (*.xlsx)"
        )
        if path:
            pl.DataFrame(
                [[table.item(r, c).text() if table.item(r, c) else ""
                  for c in range(5)] for r in range(table.rowCount())],
                  orient="row",
                schema=["File Name", "Line 1", "Line 2", "Line 3", "Line 4"]
            ).write_excel(path)
            QMessageBox.information(self.parent, "Success", "Results saved successfully")