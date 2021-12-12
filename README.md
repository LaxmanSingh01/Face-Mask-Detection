# Face-Mask-Detection
![test](https://user-images.githubusercontent.com/84785447/135806806-7b9dc817-8ed3-49db-995e-4afc347f0a2f.gif)

## About Project

This project uses a Deep Neural Network, more specifically a Convolutional Neural Network, to differentiate between images of people with and without masks. The CNN manages to get an accuracy of 98.2% on the training set and 94.3% on the test set. Then the stored weights of this CNN are used to classify as mask or no mask, in real time, using OpenCV. With the webcam capturing the video, the frames are preprocessed and and fed to the model to accomplish this task. The model works efficiently with no apparent lag time between wearing/removing mask and display of prediction.

## Dataset

The data used can be downloaded through this link or can be downloaded from this repository as well (folders 'test' and 'train'). There are 1100 training images and 276 test images divided into two catgories, with and without mask.

## To run the app follow the below command.
streamlit run app.py
