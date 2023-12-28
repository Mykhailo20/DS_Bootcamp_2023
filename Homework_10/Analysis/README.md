# ANPR with YOLOv8, OpenCV and EasyOCR
Algorithm for localization of license plates on images of cars, which uses:
1. YOLOv8 model (the first step of license plate localization);
2. image processing functions of the OpenCV library (the second potential step of license plate localization and its alignment using affine transformations);
3. easyOCR (checking the presence of text on selected image areas).

## :open_file_folder: Project Files Description
This project includes 2 Jupyter Notebook files and 1 directory as follows:

**Jupyter Notebook files**:
1. **LP_Alignment.ipynb** - Contains the license plate image alignment function code and its test results.
2. **Select_Data.ipynb** - Contains helping functions and the code of the implemented algorithm.

**Directories**:
1. **my_yolo_model** - Contains weights, plots, and other results of training the YOLOv8n model on the filtered dataset.

More detailed information about the algorithm can be viewed in the reports for homework [№9](https://github.com/Mykhailo20/DS_Bootcamp_2023/tree/main/Homework_9/Report) and [№10](https://github.com/Mykhailo20/DS_Bootcamp_2023/tree/main/Homework_10/Report).