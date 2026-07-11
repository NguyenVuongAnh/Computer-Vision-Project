# Computer Vision Project

This project is a Streamlit application for computer vision tasks.

## Project Structure

```
.
├── .gitignore
├── app.py
├── backend
│   ├── inference.py
│   ├── preprocessing.py
│   └── utils.py
├── Dataset
│   ├── Classification
│   └── Segmentation
├── frontend
│   ├── components
│   └── style.py
├── model
│   ├── __init__.py
│   ├── CMAC.py
│   ├── components.py
│   ├── ultis
│   └── weights
├── outputs
│   ├── predictions.csv
│   └── result_images
├── README.md
├── READMEvn.md
├── requirements.txt
└── Report
    ├── Bệnh võng mạc tiểu đường.docx
    └── Phát hiện bệnh tiểu đường trên ảnh võng mạc mắt dựa trên miền không gian.docx
```

## Folder Explanations

- **`app.py`**: The main entry point for the Streamlit application.
- **`backend/`**: Contains the core logic for data processing and model inference.
  - `inference.py`: Handles running the model to generate predictions.
  - `preprocessing.py`: Contains functions for preparing input data.
  - `utils.py`: Utility scripts for backend operations.
- **`Dataset/`**: Contains the datasets used for training and evaluation. It's divided into `Classification` and `Segmentation` tasks.
- **`frontend/`**: Manages the user interface components of the Streamlit app.
  - `style.py`: Defines the visual style of the application.
  - `components/`: Holds reusable UI components.
- **`model/`**: Contains the neural network models and related utilities.
  - `__init__.py`: Initializes the `model` folder as a Python package, allowing for easier module imports. It imports `CMAC` and `components`.
  - `CMAC.py`: Defines the core `CMAC` model architecture.
  - `components.py`: Contains building blocks or sub-modules for the models.
  - `ultis/`: Holds utility functions specific to the models, like custom metrics or configuration settings.
  - `weights/`: Stores the pre-trained model weights.
- **`outputs/`**: Default directory for saving the application's output.
  - `predictions.csv`: Stores model predictions in a CSV file.
  - `result_images/`: Stores output images after processing.
- **`Report/`**: Contains project documentation and reports in Vietnamese.
- **`requirements.txt`**: Lists all the Python packages required to run the project.
- **`README.md` / `READMEvn.md`**: Project documentation in English and Vietnamese.
