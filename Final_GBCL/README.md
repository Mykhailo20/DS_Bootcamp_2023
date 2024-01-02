<p align='center'>
   <img src='images\assets\Online_Garbage_Classifier_Banner.png' alt='banner'>
</p>
<p align='center'>
   <i>
      A Streamlit application with a user-friendly interface that uses a trained CNN model based on ResNet152 to classify a user-uploaded garbage image
   </i>
</p>

## :open_file_folder: Project Files Description
This project includes 3 Python files and 3 directories as follows:

**Python files**:
1. **nn_model.py** - Contains functions for configuring TensorFlow, loading and using a neural network model.
2. **preprocess_image.py** - Contains functions for preparing original images for use in the CNN model, which is based on ResNet152.
3. **main.py** - Contains code for the Streamlit application functionality.

**Directories**:
1. **images**: 
   * **classes_images** - Contains images of classic representatives of each class of garbage.
   * **assets** - Contains the images used in the README file.

2. **modules** - Contains files nn_model.py and preprocess_image.py for working with CNN models.
3. **styles** - Contains a styles.css file with additional settings for the CSS styles of the application's main web page.

For more detailed information about CNN model training and the project, you can view the relevant [directory](https://github.com/Mykhailo20/DS_Bootcamp_2023/tree/main/Final_Project/Analysis) and [report](https://github.com/Mykhailo20/DS_Bootcamp_2023/tree/main/Final_Project/Report).

## :inbox_tray: Installation
1. Download and Extract:
   * Download the [zip file](https://drive.google.com/drive/u/0/folders/1tnMkm3FkGGosZ4rT0YzcQPmARRe07Oz9) of the project folder.
   * Extract the contents to your desired location.
2. Create a Virtual Environment: 
   * Open a terminal at the project level.
   * Run the following commands to create and activate a virtual environment:
   
      ```python
      python -m venv venv

      .\venv\Scripts\activate    # For Windows

      source venv/bin/activate   # For Unix/Linux
      ```
3. Install the necessary libraries:
   
   ```
   pip install -r requirements.txt
   ```

## :rocket: Usage
Run the command:
```
streamlit run main.py
```

## :computer: Functionality Overview
1. Classification of user-uploaded garbage image:

<p align='center'>
   <img src='images\assets\classification.gif' alt='classification' width='900'>
</p>

2. Viewing the classic representatives of each garbage class:

<p align='center'>
   <img src='images\assets\view_representatives.gif' alt='representatives' width='900'>
</p>