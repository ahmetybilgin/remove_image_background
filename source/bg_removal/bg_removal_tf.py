import os
import numpy as np
import cv2

from PIL import Image
from settings import OUTPUT_DIR, INPUT_DIR
from source.dominant_color_removal import remove_dominant_color


def draw_segment(base_img, mat_img, filename_d):
    """Postprocessing. Saves complete image."""

    frame_path = INPUT_DIR + '/' + filename_d
    frame = cv2.imread(frame_path)
    # Get image size
    width, height = base_img.size
    # Create empty numpy array
    dummy_img = np.zeros([height, width, 4], dtype=np.uint8)
    # Create alpha layer from model output
    for x in range(width):
        for y in range(height):
            color = mat_img[y, x]
            (r, g, b) = base_img.getpixel((x, y))
            if color == 0:
                dummy_img[y, x, 3] = 0
            else:
                dummy_img[y, x] = [r, g, b, 255]

    # Restore image object from numpy array
    img = Image.fromarray(dummy_img)
    # cv2_frame = np.array(img)
    # Convert RGB to BGR
    # before_r, before_g, before_b, before_a = cv2.split(cv2_frame)
    # bgra_frame = cv2.merge([before_b, before_g, before_r, before_a])
    # bgr_frame = cv2.cvtColor(bgra_frame, cv2.COLOR_BGRA2BGR)
    # cv2.imshow("bgr frame", bgr_frame)
    # cv2.waitKey()
    # bgr_gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("gray frame", bgr_gray)
    # cv2.waitKey()
    # _, bgr_thresh = cv2.threshold(bgr_gray, 160, 255, cv2.THRESH_BINARY)
    # cv2.imshow("bgra frame", bgr_thresh)
    # cv2.waitKey()
    # cv2_frame = cv2_frame[:, :, ::-1, :].copy()
    # bg_color_removed_frame = remove_dominant_color(cv_frame=bgra_frame, dm_color=dm_color, frame_width=width,
    #                                                frame_height=height)
    # after_b, after_g, after_r, after_a = cv2.split(bg_color_removed_frame)
    # rgba_frame = cv2.merge([after_r, after_g, after_b, after_a])
    # im_pil = Image.fromarray(rgba_frame)

    # Remove file extension
    filename_d = os.path.splitext(filename_d)[0]
    # Save image
    save_file_path = OUTPUT_DIR + '/' + filename_d + '.png'
    img.save(save_file_path)
    # im_pil.save(save_file_path)

    return save_file_path


if __name__ == '__main__':

    draw_segment(base_img=Image.open(""), mat_img=Image.open(""), filename_d="")
