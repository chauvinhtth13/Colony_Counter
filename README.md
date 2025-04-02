# Colony Counter

**Colony Counter** is a desktop application designed to assist researchers in counting bacterial or cellular colonies in images. It leverages a PyQt6-based graphical user interface (GUI) and OpenCV for image processing, with additional support from libraries like NumPy, SciPy, scikit-learn, Polars, and XlsxWriter for advanced data handling, optimization, and export capabilities. The application enables automated line detection, manual bounding box (bbox) editing, colony counting, and result exporting.

---

## Features

-   **Image Loading**: Load one or multiple images using `sys` and `os` for file handling.
-   **Line Detection**: Automatically detect colony lines with OpenCV (`opencv-python-headless`).
-   **Manual Editing**: Resize, move, or remove bboxes:
    -   Left-click and drag corners (top-left or bottom-right) to resize using `math` for coordinate calculations.
    -   Left-click and drag inside to move.
    -   Right-click inside to remove.
-   **Colony Counting**: Count colonies using `sklearn.mixture.GaussianMixture` for clustering and `sklearn.metrics.silhouette_score` for validation.
-   **Data Export**: Export results to Excel with `xlsxwriter` and manage data with `polars`.
-   **Image Navigation**: Switch between images with "Next" and "Previous" buttons.
-   **Optimization**: Utilize `pyscipopt` and `scipy` for advanced processing (assumed functionality).

---

## Prerequisites

-   **Python 3.8+**: Required for running the source code.
-   **Windows**: Deployed as a standalone executable for Windows systems.

---

## Installation

### For Users

1.  Download `colony_counter.exe` from the [Releases](#) page (link to be added by repository owner).
2.  Double-click the executable to launch—no additional setup required.

### For Developers

To run or modify the source code:

1.  **Clone the Repository**:
    ```bash
    git clone [https://github.com/yourusername/colony_counter.git](https://github.com/yourusername/colony_counter.git) # Replace with your actual repo URL
    cd colony_counter
    ```
2.  **Set Up a Virtual Environment**:
    ```bash
    # Create the environment
    python -m venv venv

    # Activate the environment
    # On Windows:
    venv\Scripts\activate
    # On macOS/Linux:
    # source venv/bin/activate
    ```
3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run the Application**:
    ```bash
    python run.py
    ```

---

## Requirements

The `requirements.txt` file lists all necessary third-party Python packages:

```plaintext
opencv-python-headless==4.10.0.84
PyQt6==6.7.1
numpy==2.0.2
scipy==1.14.1
pyscipopt==5.1.0
scikit-learn==1.5.2
polars==1.7.1
xlsxwriter==3.2.0