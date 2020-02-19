# Project Description

This project is to detect a several objects including person in an image, convert white background to transparent and save the image obtained through such a process into png file format.

## Project Structure

- data

    There are two directories, one of which contains the images to process and other of which shows the result images, contains the png file.

- source
    
    The main source code to process image transformation is contained.
    
- utils

    Tools concerned with image processing and file management are contained.

- main

    This is the main execution file.

- requirements
    
    All the libraries to execute project are inserted.

- settings

    Several settings are conducted in this file

## Project Install

- Environment

    Ubuntu 18.04
    
- Python 3.6 environment

- Library install
    
    ```
     pip3 install -r requirement.txt
    ```

## Project Execution

- Please copy images to transform in data/input directory.

- After downloading deep learning model at http://download.tensorflow.org/models/deeplabv3_pascal_train_aug_2018_01_04.tar.gz, please copy it to utils/model directory. 

- Please go ahead to this project directory and execute following command in terminal
    
    ```
    python3 main.py
    ```

Then this project transforms all the images in input directory and exports the result with png format.
