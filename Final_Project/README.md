# Online Garbage Classifier CNN
CNN models built on the basis of ResNet152 and trained using built-in TensorFlow and self-implemented augmentation techniques to classify garbage images into one of 6 classes.

## :open_file_folder: Project Files Description
This project includes 5 Jupyter Notebook files and 2 directories as follows:
**Jupyter Notebook files**:
1. **Augmentation_Techniques.ipynb** and **Augmentation_Techniques_Colab.ipynb** - Contains functions and code for implementing augmentation techniques (the **Augmentation_Techniques_Colab.ipynb** file contains a larger set of augmentation techniques - since the training was conducted in Google Colab).
2. **1_TF_Classification_DNN.ipynb** and **1_TF_Classification_DNN_Colab.ipynb** - Contains functions and code for performing image augmentation, training neural network models, evaluating their effectiveness, and storing results.
3. **Test_Models.ipynb** - Contains the results of testing trained models on original and self-collected test datasets.

**Directories**:
1. **data** - Contains image files for backgrounds (used for augmentation) and self-compiled test mini-dataset (my_test_data).
2. **models** - Contains .h5 files of trained CNN models (unfortunately, due to file size limitations that can be uploaded to GitHub, the models with the best metric values ​​are not available on GitHub, but they can be downloaded from [here](https://drive.google.com/drive/u/0/folders/10jRQjfWCOBEqEPYYb6Jq-nqIo1KZVB_0)).

For more detailed information about applied augmentation techniques and training of CNN models, you can view the [report](https://github.com/Mykhailo20/DS_Bootcamp_2023/tree/main/Final_Project/Report).