# Physical Activity Recognition
A Feed Forward Neural Network model trained on a self-collected dataset to recognize physical activity types based on accelerometer and gyroscope data.
## :open_file_folder: Project Files Description
This project includes 3 Jupyter Notebook files and 2 directories as follows:

**Jupyter Notebook files**:
1. **Physical_Activity_Analysis.ipynb** - Contains code for processing training and test datasets, in particular, combining the datasets collected for each physical activity into the train dataset and checking the correctness of data labeling in Label Studio.
2. **Features.ipynb** - Contains code for performing the following Data Science pipeline steps on training and test data:
   - Data Cleaning and Preprocessing;
   - Exploring measurement period and frequency stability;
   - Data Filtering;
   - Exploratory Data Analysis;
   - Data Transformation;
   - Feature Engineering.
3. **NN_model_training.ipynb** - Contains code for training a neural network, evaluating its performance, and using Feature Selection techniques.

**Directories**:
1. **data** - Contains data (both raw and processed) of the smartphone's accelerometer and gyroscope for 5 investigated types of physical activity.
2. **models** - Contains the .h5 file of the trained neural network model and its training history file.

## :bar_chart: Project pipeline
The implementation of the project involved the following Data Science stages:
1. **Data Collection** - Collecting readings of my smartphone's accelerometer and gyroscope.
2. **Data Cleaning and Preprocessing** - Checking for null values ​​and errors in the 'time' column.
3. **Data Labeling** - Labeling training and test datasets in Label Studio.
4. **Frequency stability analysis** - Checking the accelerometer and gyroscope data sampling frequency.
5. **Data Filtering** - Using a median filter with a window size of 10.
6. **Exploratory Data Analysis** - Analysis of distribution of target classes, study of distributions of accelerometer and gyroscope signals, as well as conducting correlation analysis.
7. **Data Transformation** - Division of the original dataframe into windows.
8. **Feature Engineering** - Calculation of values ​​of statistical parameters for each window.
9. **Model Training And Evaluation** - Model training, selection of its optimal architecture and evaluation on test data.
10. **Feature Selection** - Using PFI, MDI and RFE algorithms to reduce the number of model input parameters.

For more detailed information about each step of the DS pipeline, you can use the [report](https://github.com/Mykhailo20/DS_Bootcamp_2023/tree/main/Homework_6/Report).

