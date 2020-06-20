## Image Stitching

Image stitching is a basic concept in the field of Image Processing that is used in order to form 'Panaroma'. If you are not fully aware of how image stitching works then go ahead and read the next section. Otherwise, you can directly jump to the 'Requirements' and 'Implementation' section of this repository.

### Introduction: (Work In Progress)
Let's first dive into the key points used in Image stitching:

##### Feature descriptor:

Feature descriptors are dense representations that best describe the contents of a given image. Following are some feature descriptors used widely. (The description of each method is out of scope of this repository, please check out the associated link (if any) for more information)

1. Scale Invarient Feature Transform (SIFT):
2. Speeded Up Robust Features (SURF):
3. Oriented Fast and Robust BRIEF (ORB):

##### Feature matching:

While creating panorama by stitching two images together, we require some ammount of overlap between these two images. Feature matching, as the name suggests, is used to match features between these two images in order to find the overlap.
1. Brute Force : Matches feature set in one image with feature set in second image using the following distance:
    1. L2
    2. Hamming

##### Get top n best features:
Choose those set of features  with highest match.

##### Obtain Homography Matrix:
(incomplete)
Random sample consensus or RANSAC is a method used for fitting models to data.

##### Warp Image:
(incomplete)

### Requirements:
Firstly, this project is based on Python and I highly recommend to create a Python virtual enviroment.
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

  * *DATASET*:
    - *INPUT_DIR*: Path to the folder where all the images are stored. (default=test)
    - *INPUT_IMG_LIST*: List of Images.
    - *OUTPUT_DIR*: Path to the folder where all the results should be stored. (default=results)

  * *FEATURES*:
    - *FEATURE_DESCRIPTORS*: Default is set (Other choices are provided in comments)
    - *FEATURE_MATCHING*: Default is set (Other choices are provided in comments)
    - *FEATURE_MATCHING_THRESHOLD*: Default is set (Other choices are provided in comments)

  One can simply change the parameters in the config file to try the effect of the different techniques.

2.  Command to run the program ``` python -m run --c [path to config.yml]  ```

    I have kept the path to config.yml as an argument so that the user can have multiple config files corresponding to different projects (with different images and varied feature attributes)
