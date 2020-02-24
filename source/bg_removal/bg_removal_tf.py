import os
import numpy as np
import cv2

from settings import OUTPUT_DIR, INPUT_DIR, HSV_THRESH, SUB_IMG_COUNTS, SUB_IMG_MARGIN
from utils.image_processing import unique_count_app


def draw_segment(base_img, mat_img, filename_d):
    """Postprocessing. Saves complete image."""

    # Get image size
    ######################################################################################3
    base_img = cv2.imread("/media/mensa/Data/Task/ImageBgRemoval/HTB1ePz7s21TBuNjy0Fjq6yjyXXaU.jpg")
    mat_img = cv2.imread("/media/mensa/Data/Task/ImageBgRemoval/HTB1ePz7s21TBuNjy0Fjq6yjyXXaU.jpg_mask.png", 0)
    # base_img = cv2.imread("/media/mensa/Data/Task/ImageBgRemoval/attachment_96091205.png")
    # mat_img = cv2.imread("/media/mensa/Data/Task/ImageBgRemoval/attachment_96091205_mask.png", 0)
    ######################################################################################3

    frame_height = base_img.shape[0]
    frame_width = base_img.shape[1]
    bg_removed_img = np.zeros((frame_height, frame_width, 3), np.uint8)

    mat_img_not = cv2.bitwise_not(mat_img)

    # Convert RGB to BGR
    # base_img_hsv = cv2.cvtColor(base_img, cv2.COLOR_BGR2HSV)
    s_v_value = unique_count_app(frame=base_img, mask=mat_img_not)
    lower_range = adjust_hsv_range(hsv_list=np.array(s_v_value) - HSV_THRESH)
    upper_range = adjust_hsv_range(hsv_list=np.array(s_v_value) + HSV_THRESH)
    lower_range = np.array([0, lower_range[0], lower_range[1]])
    upper_range = np.array([255, upper_range[0], upper_range[1]])

    for i in range(SUB_IMG_COUNTS):

        top = int(i * frame_height / SUB_IMG_COUNTS) - 2 * SUB_IMG_MARGIN
        bottom = int((i + 1) * frame_height / SUB_IMG_COUNTS)
        if top < 0:
            top = 0

        sub_img = base_img[top:bottom, :]
        new_margin_img = np.zeros((bottom - top + 2 * SUB_IMG_MARGIN, frame_width, 3), np.uint8)
        new_base_margin_img = new_margin_img.copy()
        new_base_margin_img[:, :] = (0, s_v_value[0], s_v_value[1])
        new_margin_img_bgr = cv2.cvtColor(new_base_margin_img, cv2.COLOR_HSV2BGR)
        new_margin_img_bgr[SUB_IMG_MARGIN:SUB_IMG_MARGIN + bottom - top, :] = sub_img
        new_margin_img_hsv = cv2.cvtColor(new_margin_img_bgr, cv2.COLOR_BGR2HSV)

        sub_mat_img = mat_img[top:bottom, :]
        new_mat_margin_img = np.zeros((bottom - top + 2 * SUB_IMG_MARGIN, frame_width), np.uint8)
        new_mat_margin_img[SUB_IMG_MARGIN:SUB_IMG_MARGIN + bottom - top, :] = sub_mat_img
        # cv2.imshow("new mat margin image", new_mat_margin_img)
        # cv2.waitKey()
        kernel = np.ones((2, 2), np.uint8)
        new_mat_margin_erosion = cv2.erode(new_mat_margin_img, kernel, iterations=1)

        # mat_img_not = cv2.bitwise_not(sub_mat_img)
        bg_mask = cv2.inRange(src=new_margin_img_hsv, lowerb=lower_range, upperb=upper_range)
        cv2.imshow("back ground mask", bg_mask)
        cv2.waitKey(1)

        # bg_mask_not = cv2.bitwise_not(bg_mask)
        # bg_mask_not_dilation = cv2.dilate(bg_mask_not, kernel, iterations=5)
        # cv2.imshow("back ground dilation", bg_mask_not_dilation)
        # bg_mask_not_erosion = cv2.erode(bg_mask_not_dilation, kernel, iterations=7)

        # Convert RGB to BGR

        contour, _ = cv2.findContours(new_mat_margin_erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sorted_contour = sorted(contour, key=cv2.contourArea, reverse=True)
        rect = cv2.boundingRect(sorted_contour[0])
        # rect = [rect[0] - SUB_IMG_MARGIN, rect[1], rect[2] + 2 * SUB_IMG_MARGIN, rect[3]]
        # rect = [rect[0], rect[1] + SUB_IMG_MARGIN, rect[2], rect[3] - 2 * SUB_IMG_MARGIN]

        # rect_img = cv2.rectangle(new_margin_img_bgr.copy(), (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 0, 255), 5)
        # cv2.imshow("rect", rect_img)
        # cv2.imshow("new_mat_margin_img", new_mat_margin_img)
        # cv2.waitKey(0)

        mask = np.zeros(new_margin_img_bgr.shape[:2], np.uint8)
        # mask[bg_mask_not_erosion == 0] = 0
        # mask[bg_mask_not_erosion == 255] = 1

        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        # mask = np.zeros(sub_img.shape[:2], np.uint8)
        # mask, _, _ = cv2.grabCut(new_margin_img_bgr, mask, None, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)
        mask, _, _ = cv2.grabCut(new_margin_img_bgr, mask, rect, bgd_model, fgd_model, 5,
                                 cv2.GC_INIT_WITH_RECT)
        new_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

        new_sub_img = new_margin_img_bgr * new_mask[:, :, np.newaxis]
        # bg_removed_img[top:bottom, :] = new_sub_img
        cv2.imshow("sub bg removed image", new_sub_img)
        cv2.waitKey()

    cv2.imshow("total bg removed image", bg_removed_img)
    cv2.waitKey()
    # h, w = base_img.shape[:2]

    # base_img = base_img[h//2:]
    # mat_img = mat_img[h//2:]
    #
    # Create empty numpy array
    mat_img_not = cv2.bitwise_not(mat_img)

    # Convert RGB to BGR
    base_img_hsv = cv2.cvtColor(base_img, cv2.COLOR_BGR2HSV)
    s_v_value = unique_count_app(frame=base_img, mask=mat_img_not)
    lower_range = adjust_hsv_range(hsv_list=np.array(s_v_value) - HSV_THRESH)
    upper_range = adjust_hsv_range(hsv_list=np.array(s_v_value) + HSV_THRESH)
    lower_range = np.array([0, lower_range[0], lower_range[1]])
    upper_range = np.array([255, upper_range[0], upper_range[1]])

    bg_mask = cv2.inRange(src=base_img_hsv, lowerb=lower_range, upperb=upper_range)
    cv2.imshow("back ground mask", bg_mask)
    cv2.waitKey(1)

    bg_mask_not = cv2.bitwise_not(bg_mask)
    kernel = np.ones((7, 7), np.uint8)
    bg_mask_not_dilation = cv2.dilate(bg_mask_not, kernel, iterations=5)
    # cv2.imshow("back ground dilation", bg_mask_not_dilation)
    bg_mask_not_erosion = cv2.erode(bg_mask_not_dilation, kernel, iterations=7)
    cv2.imshow("back ground erosion", bg_mask_not_erosion)
    cv2.waitKey(1)

    contour, _ = cv2.findContours(bg_mask_not_erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_contour = sorted(contour, key=cv2.contourArea, reverse=True)
    rect = cv2.boundingRect(sorted_contour[0])
    x, y, w, h = rect
    m = 20
    rect = [x, y + m, w, h - m * 2]

    # cv2.rectangle(base_img, (rect[0], rect[1]), (rect[2] + rect[0], rect[1] + rect[3]), (0, 0, 255), 5)
    # rect = [1, 1, base_img.shape[1], base_img.shape[0]]

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    mask = np.zeros(base_img.shape[:2], np.uint8)
    mask[bg_mask_not_erosion == 0] = 0
    mask[bg_mask_not_erosion == 255] = 1

    # op_img = cv2.cvtColor(bg_mask_not_erosion, cv2.COLOR_GRAY2BGR)
    # op_img = cv2.blur(op_img, (11, 11)) / 255.0
    # blur_bgr_frame = (base_img * op_img).astype(np.uint8)

    # mask, bgdModel, fgdModel = cv2.grabCut(base_img, mask, None, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)
    mask, bgdModel, fgdModel = cv2.grabCut(base_img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    grab_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    # grab_frame = blur_bgr_frame * grab_mask[:, :, np.newaxis]
    grab_frame = base_img * grab_mask[:, :, np.newaxis]

    res_img = cv2.merge(cv2.split(grab_frame)[:3] + [grab_mask * 255])
    cv2.imshow("base", cv2.resize(base_img, None, fx=0.5, fy=0.5))
    cv2.imshow("background", cv2.resize(grab_mask * 255, None, fx=0.5, fy=0.5))
    cv2.imshow("mask", cv2.resize((mask * 120).astype(np.uint8), None, fx=0.5, fy=0.5))
    cv2.imshow("foreground", cv2.resize(grab_frame, None, fx=0.5, fy=0.5))
    cv2.imwrite("res_img.png", res_img)
    cv2.waitKey(0)

    # _, bgr_thresh = cv2.threshold(bgr_gMensaray, 160, 255, cv2.THRESH_BINARY)
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
    # img.save(save_file_path)
    # im_pil.save(save_file_path)

    return save_file_path


def adjust_hsv_range(hsv_list):
    for i, value in enumerate(hsv_list):
        value = max(0, min(value, 255))
        hsv_list[i] = value
    return hsv_list


if __name__ == '__main__':

    # draw_segment(base_img=Image.open(""), mat_img=Image.open(""), filename_d="")
    draw_segment(base_img="", mat_img="", filename_d="")
