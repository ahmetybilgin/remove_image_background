import os
import cv2
import ntpath

from settings import OUTPUT_DIR
from source.dominant_color_removal import remove_dominant_color


def detect_image_outline(frame_path, dm_color):

    origin_file_name = ntpath.basename(frame_path)
    png_file_name = origin_file_name.replace(origin_file_name[origin_file_name.rfind(".") + 1:], "png")
    file_path = os.path.join(OUTPUT_DIR, png_file_name)

    frame = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
    width = frame.shape[1]
    height = frame.shape[0]

    frame_rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, frame_thresh = cv2.threshold(frame_gray, 250, 255, cv2.THRESH_BINARY_INV)
    # cv2.imshow("thresh image", frame_thresh)
    # cv2.waitKey()

    contour, _ = cv2.findContours(frame_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_contour = sorted(contour, key=cv2.contourArea, reverse=True)

    for y in range(height):

        for x in range(width):

            point = (x, y)
            point_test = True
            for cnt in sorted_contour:
                dist_point_cnt = cv2.pointPolygonTest(cnt, point, False)

                if dist_point_cnt > 0:
                    point_test = False
                    break
            pixel = frame_rgba[y, x]
            if point_test:

                pixel[3] = 0

            if pixel[3] != 0 and pixel[0] == dm_color[0] and pixel[1] == dm_color[1] and pixel[2] == dm_color[2]:
                pixel[3] = 0

    # bg_color_removed_frame = remove_dominant_color(cv_frame=frame_rgba, dm_color=dm_color, frame_width=width,
    #                                                frame_height=height)
    cv2.imwrite(file_path, frame_rgba)

    return file_path


if __name__ == '__main__':

    detect_image_outline(frame_path="", dm_color=[])
