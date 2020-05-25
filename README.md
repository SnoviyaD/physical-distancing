# Physical-Distancing
This project was built during Covid19 lockdown. It uses Tensorflow and OpenCV Python to check if physical distancing measures are followed by the citizens or not.

### This project is inspired from the works of Landing AI and Airpix.

![banner](output.gif)

## Technologies 
1. OpenCV 3.0 (or above)
2. Tensorflow 1.5 (or above)
3. Python 3.5 (or above)
###### (Note: If your machine supports GPU then using Tensorflow GPU version is recommended for better results)

## Getting Started

1. Make sure you have installed all the [mentioned](#Technologies) technologies on your machine.
2. Create a folder for this project.
3. Download [PhysicalDistancing.py](https://github.com/SnoviyaD/physical-distancing/blob/master/PhysicalDistancing.py) and save it into the project directory.
4. Download COCO trained-model from [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) and extract it. For this project I have used "faster_rcnn_inception_v2_coco". You can change the model depending on your computing resources. 
5. Update the paths to model and and input video in the code. Use 0 to start real-time streaming through webcam.
6. Move to the project directory and run the python file. The output will be displayed. Press Esc to stop.
###### (Note: If you are using Tensorflow 1.x (where x>=5 )then change to "import tensorflow as tf" in the code)

## Testing Environment

1. Laptop specifications: Intel(R) Core(TM)i5-7200U (upto 2.7GHz), 8 GB Memory
2. Operating System     : Ubuntu 16.04
3. Python version       : Python 3.5.2
4. Tensorflow version   : 2.2.0 
5. OpenCV version       : 4.2.0     
6. Input Video          : [TownCentreXVID.avi](http://www.robots.ox.ac.uk/~lav/Research/Projects/2009bbenfold_headpose/project.html)

## References

1. [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) is a great start to detect objects of different categories alongwith their scores.
2. [DetectionAPI](https://gist.github.com/madhawav/1546a4b99c8313f06c0b2d7d7b4a09e2),authored by [@Madhawa Vidanapathirana](https://gist.github.com/madhawav) is an open source project that helps us detect humans from the input and is used for the start of the project
3. The video used for testing is from [Coarse Gaze Estimation in Visual Surveillance Project](http://www.robots.ox.ac.uk/~lav/Research/Projects/2009bbenfold_headpose/project.html).
