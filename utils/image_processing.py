import numpy as np
import cv2
from matplotlib import pyplot as plt


def unique_count_app(frame, mask):

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv_main_values = []
    for i in range(1, 3):

        hist = cv2.calcHist([hsv_frame], [i], mask, [256], [0, 256])
        main_value = np.argmax(hist)
        hsv_main_values.append(main_value)

    return hsv_main_values


if __name__ == '__main__':

    pass
    # unique_count_app(frame_path="")
