import numpy as np
import random
import time
import cv2
import os

from settings import MASK_RCNN_DIR


def detect_person_mask():

    labelsPath = os.path.join(MASK_RCNN_DIR, "object_detection_classes_coco.txt")
    LABELS = open(labelsPath).read().strip().split("n")

    # weightsPath = os.path.join(MASK_RCNN_DIR, "frozen_inference_graph.pb")
    # configPath = os.path.join(MASK_RCNN_DIR, "mask_rcnn_inception_v2_coco_2018_01_28.pbtxt")

    weightsPath = "/media/mensa/Data/Task/ImageBgRemoval/utils/mask_rcnn_resnet101_atrous_coco_2018_01_28/frozen_inference_graph.pb"
    configPath = "/media/mensa/Data/Task/ImageBgRemoval/utils/graph.pbtxt"

    colorsPath = os.path.join(MASK_RCNN_DIR, "colors.txt")
    COLORS = open(colorsPath).read().strip().split("n")
    COLORS = np.array([0, 0, 255])
    COLORS = np.array(COLORS, dtype="uint8")

    net = cv2.dnn.readNetFromTensorflow(weightsPath, configPath)

    (H, W) = image.shape[:2]

    blob = cv2.dnn.blobFromImage(image, swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    (boxes, masks) = net.forward(["detection_out_final", "detection_masks"])
    end = time.time()

    for i in range(0, boxes.shape[2]):
        # extract the class ID of the detection along with the confidence
        # (i.e., probability) associated with the prediction
        classID = int(boxes[0, 0, i, 1])
        confidence = boxes[0, 0, i, 2]
        # filter out weak predictions by ensuring the detected probability
        # is greater than the minimum probability
        if confidence > 0.1:
            # clone our original image so we can draw on it
            clone = image.copy()
            # scale the bounding box coordinates back relative to the
            # size of the image and then compute the width and the height
            # of the bounding box
            box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
            (startX, startY, endX, endY) = box.astype("int")
            boxW = endX - startX
            boxH = endY - startY
            # extract the pixel-wise segmentation for the object, resize
            # the mask such that it's the same dimensions of the bounding
            # box, and then finally threshold to create a *binary* mask
            mask = masks[i, classID]

            cv2.imshow("mask", mask)
            cv2.waitKey(0)

            mask = cv2.resize(mask, (boxW, boxH), interpolation=cv2.INTER_NEAREST)
            mask = (mask > 0.1)

            cv2.imshow("mask", mask)
            cv2.waitKey(0)

            # extract the ROI of the image
            roi = clone[startY:endY, startX:endX]
            # check to see if are going to visualize how to extract the
            # masked region itself

            # convert the mask from a boolean to an integer mask with
            # to values: 0 or 255, then apply the mask
            visMask = (mask * 255).astype("uint8")
            instance = cv2.bitwise_and(roi, roi, mask=visMask)
            # show the extracted ROI, the mask, along with the
            # segmented instance

            roi = roi[mask]
            # randomly select a color that will be used to visualize this
            # particular instance segmentation then create a transparent
            # overlay by blending the randomly selected color with the ROI
            color = random.choice(COLORS)
            blended = ((0.4 * color) + (0.6 * roi)).astype("uint8")
            # store the blended ROI in the original image
            print(classID)
            clone[startY:endY, startX:endX][mask] = blended
            cv2.imshow("mask image", clone)
            cv2.waitKey(0)
            remove_bg_cv2(x1=startX, y1=startY, x2=W, y2=H)


def remove_bg_cv2(x1, y1, x2, y2):

    mask = np.zeros(image.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    rect = (1, 1, x2-1, y2-1)
    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img = image * mask2[:, :, np.newaxis]

    cv2.imshow("mask cv image", img)
    cv2.waitKey()


if __name__ == '__main__':

    frame_path = "/media/mensa/Data/Task/ImageBgRemoval/data/input/HTB1ePz7s21TBuNjy0Fjq6yjyXXaU.jpg"
    image = cv2.imread(frame_path)
    detect_person_mask()
