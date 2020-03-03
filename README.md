# ImageBgRemoval

## Overview

This project is to detect the several objects including person in an image, convert white background to transparent and 
save the obtained image into png file format. To improve the accuracy of transformation, the Pytorch framework is used, 
which needs usage of GPU. In detail, the deeplab framework is used in this project.

## Structure

- data

    There are two directories, one of which contains the images to convert background and other of which shows the 
    result images, contains the png file.

- source
    
    The main source code to process image transformation.
    
- utils

    The tools concerned with image processing, models and file management.

- main

    The main execution file.

- requirements
    
    All the dependencies to execute project.

- settings

    Several options in it.

## Installation

- Environment

    Ubuntu 18.04, Python 3.6, GPU

- Dependency installation
    
    ```
        pip3 install -r requirements.txt
    ```

- Preparation for deeplab model
    
    * Please download the model from https://drive.google.com/file/d/1vzLRbMua_hxYXpBH8WWTozThHX4DTvIX/view?usp=sharing 
    and copy them to /utils/model.
    * After downloading deep learning model at 
    http://download.tensorflow.org/models/deeplabv3_pascal_train_aug_2018_01_04.tar.gz, please copy it to 
    utils/model directory.
    
## Execution

- Please copy images to transform in data/input directory.

- Please go ahead to this project directory and run the following command in terminal
    
    ```
        python3 main.py
    ```

Then this project transforms all the images in input directory and exports the result with png format.
