# Colony Counter Application

**Colony Counter** is a desktop tool designed to automate and assist researchers in quantifying bacterial or cell colonies within digital images. It features a graphical interface developed with PyQt6 and utilizes OpenCV for core image analysis tasks. The application integrates several powerful libraries, including NumPy, SciPy, scikit-learn, Polars, and XlsxWriter, to offer sophisticated data processing, analysis, optimization features, and the ability to export findings. Key functions include automatic detection of colony lines, tools for manually adjusting detection boundaries (bounding boxes), colony enumeration, and exporting the results.

## Key Capabilities

* **Image Input**: Load single or multiple image files via standard file dialogs (using `sys` and `os`).
* **Automated Line Detection**: Employs OpenCV (`opencv-python-headless`) to automatically identify colony streaks.
* **Manual Bounding Box Adjustments**: Provides tools to refine detections:
    * Resize boxes by clicking and dragging the top-left or bottom-right corners (uses `math` for coordinate updates).
    * Move boxes by clicking and dragging anywhere inside the box area.
    * Delete a box by right-clicking within its boundaries.
* **Colony Enumeration**: Counts colonies by applying clustering techniques (`sklearn.mixture.GaussianMixture`) and evaluates cluster quality (`sklearn.metrics.silhouette_score`).
* **Data Output**: Exports the counting results into Excel spreadsheets using `xlsxwriter`, leveraging `polars` for efficient data structuring.
* **Multi-Image Workflow**: Easily navigate between loaded images using "Next" and "Previous" controls.
* **Advanced Processing**: Incorporates `pyscipopt` and `scipy` for advanced analysis, such as optimizing the separation of overlapping colonies.

## System Requirements

* **Python Version**: Needs Python 3.8 or newer to execute from source.
* **Operating System**: Primarily distributed as a ready-to-run executable for Windows environments.

## Getting Started

### For End Users

1.  Obtain the `colony_counter.exe` file from the project's [Releases](#) section (link should be provided by the repository maintainer).
2.  Simply double-click the `.exe` file to run the application. No installation is necessary.

### For Developers (Running from Source)

If you want to run the application from its source code or contribute to its development:

1.  **Get the Code**: Clone the repository from GitHub.
    ```bash
    # Clone the repository
    # Make sure to use the correct repository URL
    git clone [https://github.com/yourusername/colony_counter.git](https://github.com/yourusername/colony_counter.git)
    # Navigate into the project directory
    cd colony_counter
    ```
2.  **Prepare the Environment**: It's recommended to use a virtual environment.
    ```bash
    # Create a virtual environment named 'venv'
    python -m venv venv

    # Activate it:
    # Windows:
    # Activate the virtual environment (example for Windows)
    venv\Scripts\activate
    # macOS/Linux:
    # source venv/bin/activate
    ```
3.  **Install Required Libraries**:
    ```bash
    # Install all required packages from the requirements file
    pip install -r requirements.txt
    ```
4.  **Launch the App**:
    ```bash
    # Run the main application file
    python main.py
    ```

## Dependencies

All external Python libraries needed for this project are listed in the `requirements.txt` file. This includes specific versions for:

```plaintext
opencv-python-headless==4.10.0.84
PyQt6==6.7.1
numpy==2.0.2
scipy==1.14.1
pyscipopt==5.1.0
scikit-learn==1.5.2
polars==1.7.1
xlsxwriter==3.2.0
