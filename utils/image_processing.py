import numpy as np
import cv2


def unique_count_app(frame_path):

    cv2_frame = cv2.imread(frame_path)

    colors, count = np.unique(cv2_frame.reshape(-1, cv2_frame.shape[-1]), axis=0, return_counts=True)

    return colors[count.argmax()]


if __name__ == '__main__':

    unique_count_app(frame_path="")
