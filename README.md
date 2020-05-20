## Image Stitching

Image stitching is a basic concept in the field of Image Processing that is used in order to form 'Panaroma' images. If you are not fully aware of how image stitching works then go ahead and read the next section. Otherwise, you can directly jump to the 'Requirements' and 'Implementation' section of this repository.

### Introduction:
##### Feature descriptor:
1. SIFT
2. SURF
3. ORB

##### Feature matching:

##### Get top n best features:

##### RANSAC to get homography matrix:

##### Warp image

### Requirements:
Firstly, this project is based on Python and I highly recommend to create a Python virtual enviroment. This python virtual enviroment is an isolated container the helps you ....
( https://heartbeat.fritz.ai/creating-python-virtual-environments-with-conda-why-and-how-180ebd02d1db ).
There are multiple ways of creating a virtual enviroment, two of them being:
1. __virtualenv__ or __venv__ (builtin Python 3)
2. __conda env__ (using Anaconda)

For this project I am using anaconda, but it's completely upto you and what you are comfortable using.

* Python >= 3.6
* OpenCV = 3.4.2
* yaml = 5.3.1
* numpy = 1.16.4
* matplotlib = 3.1.1


### Implementation:
I have tried to make this repository as simple as possible.
There are only two things you have to keep in mind while running this repository.

1. __config.yml__ file:

  This yaml file contains the following fields:
  *  *DATASET*:
    - *INPUT_DIR*: Path to the folder where all the images are stored.
    - *INPUT_IMG_LIST*: List of Images.
    - *OUTPUT_DIR*: Path to the folder where all the results should be stored.
  * *FEATURES*:
    - *FEATURE_DESCRIPTORS*: Default is set (Other choices are provided in comments)
    - *FEATURE_MATCHING*: Default is set (Other choices are provided in comments)
    - *FEATURE_MATCHING_THRESHOLD*: Default is set (Other choices are provided in comments)

  One can simply change the parameters in the config file to try the effect of the different techniques. 

2.  Command to run the program ``` python -m run --c [path to config.yml]  ```

    I have kept the path to config.yml as an argument so that the user can have multiple config files corresponding to different projects (with different images and varied feature attributes)
