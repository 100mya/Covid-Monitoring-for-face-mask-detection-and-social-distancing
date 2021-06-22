# Covid-Monitoring-for-face-mask-detection-and-social-distancing

Covid Monitoring project consists of a FACE MASK DETECTOR and SOCIAL DISTANCING DETECTOR both integrated in a single project with graphical user interface.


## Face Mask Detection

The main goal of the project is to overcome the problem of people not wearing masks in various public places or gatherings. It aims at detecting the faces with mask and without mask using various tools and technologies.

It is implemented as a machine learning model in the form of a code written in python language. 

It uses a dataset of various faces with mask and without mask. The dataset is created by taking images of faces and artificially adding a mask to faces by obtaining facial landmarks and applying appropriate algorithms.

The project uses computer vision techniques to read the pixel data from images. This data is converted in the form of features (the pixel data) and labels (a binary value for classification). The machine learning technique requires a set of data for training the model and a small part of it for testing. In this project we have used random forest classifier to classify the images with mask and without mask.

In order to train a custom face mask detector, we need to break our project into two distinct phases, each with its own respective sub-steps:

1.	Training: Here we’ll focus on loading our face mask detection dataset from disk, training a model (using scikit-learn library) on this dataset, and then serializing the face mask detector to disk

2.	Deployment: Once the face mask detector is trained, we can then move on to loading the mask detector, performing face detection, and then classifying each face as with mask or without mask

After the training of data, the model is tested on testing set of data and evaluated on various parameters including accuracy, precision, recall and f1 score.
We deploy the model as such to predict the output from an image of a face with or without mask as well as in real-time video stream using computer vision.
Often, we have to capture live stream with camera. OpenCV provides a very simple interface to this. 

To capture a video, you need to create a VideoCapture object. Its argument can be either the device index or the name of a video file. Device index is just the number to specify which camera.

The video is a collection of frames, each frame treated like images.

The trained model predicts and classifies each frame as 'with mask' and 'without mask'. The output is printed on each frame, all together making it visible as predicted text output in live stream.


## Social Distancing Detection

We have used OpenCV, computer vision, and deep learning to implement social distancing detectors.

The steps to build a social distancing detector include:

1.	Apply object detection to detect all people (and only people) in a video stream. 
2.	Compute the pairwise distances between all detected people
3.	Based on these distances, check to see if any two people are less than N pixels apart

The simplest approach to build an Object Detection model is through a Sliding Window approach. As the name suggests, an image is divided into regions of a particular size and then every region is classified into the respective classes.

In this project we aim at using YOLO. YOLO stands for You Only Look Once. It’s a fast-operating object detection system that can recognize various object types in a single frame more precisely than other detection systems.

After detecting people in the frame, we follow the steps mentioned below:

1.	Calculate Euclidean distance between two points
2.	Convert centre coordinates into rectangle coordinates
3.	Filter the person class from the detections and get a bounding box centroid for each person detected
4.	Check which person bounding boxes are close to each other
5.	Display risk analytics and risk indicators

For the most accurate results, we can calibrate our camera through intrinsic/extrinsic parameters so that we can map pixels to measurable units.
An easier alternative (but less accurate) method would be to apply triangle similarity calibration but, in this project, we have calculated Euclidean distance between each combination of every two people detected.

Both of these methods can be used to map pixels to measurable units.

For the sake of simplicity, our OpenCV social distancing detector implementation will rely on pixel distances.

The project is integrated with another project and Graphical use interface has been implemented to make it easier to use for common people with no coding knowledge or no understanding of the technology used to develop the entire project.

For creating GUI we have used Tkinter library that is used with python to create user friendly graphical interfaces with ease.
