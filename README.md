R. Fukuda, Y. Yokoyanagi, C. Prathom, and Y. Okada, "Towards Personal Identification with Multi-Angle Ear Images: A Feasibility Study", 2025.

In this study, we constructed a convolutional neural network (CNN) model for identification based on ear images captured from multiple angles. Additionally, we used Grad-CAM to visualize feature points contributing to identification. This repository provides materials utilized in this study, including the original model and coding (.py files). The brief details are described as follows.

///Folders///
Three folders (train_data, val_data, test_data) were prepared to store images for the experiment. Each of the three folders contains five subfolders (0–4), which were prepared for 5-fold cross-validation.
Within each subfolder, there are ten additional folders (s0–s9), which are intended to contain images for each subject's ear. However, since ear images are personal data, we cannot provide the actual images here. Instead, dummy ear images are placed in these subfolders. When running the programs, please follow the steps outlined in the "Experiment Steps" section below and replace the dummy images with your own ear images in the corresponding subfolders.

///Experiment Steps///
1.	Ear trimming
Trimming ear contours using ImageJ (URL: https://imagej.net/ij/)
2.	Extract ear
Move the ear to the center of the image (ear_detect.py)
3.	Resize
Resize to 96×96 (resize.py)
4.	Change image background
Transform black background to white noise (white_noise.py)
5.	Model training
Train five CNN models using ear images (train_5fold.py)
6.	Model evaluation and visualization
Evaluate the model and visualize feature points with Grad-CAM (Test_and_GradCAM_5fold.py)

///Original Model///
original_model.h5 is the original model, which was trained based on 10 subjects' ear images captured from multiple angles.
