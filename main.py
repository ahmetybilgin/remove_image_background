import os
import tqdm
import numpy as np

from io import BytesIO
from PIL import Image
from utils.image_processing import unique_count_app
from source.object_detection import DeepLabModel
from source.bg_removal.bg_removal_tf import draw_segment
from source.bg_removal.bg_removal_cv import detect_image_outline
from settings import INPUT_DIR, OUTPUT_DIR


def run_visualization(filepath, filename_r):
    """Inferences DeepLab model and visualizes result."""

    try:
        jpeg_str = open(filepath, "rb").read()
        original_im = Image.open(BytesIO(jpeg_str))
    except IOError:
        print('Cannot retrieve image. Please check file: ' + filepath)
        return

    dominant_color = unique_count_app(frame_path=filepath)
    resized_im, seg_map = model.run(image=original_im)
    zero_ratio = np.count_nonzero(seg_map) / seg_map.size

    if zero_ratio < 0.1:

        save_file_path = detect_image_outline(frame_path=filepath, dm_color=dominant_color)
    else:

        save_file_path = draw_segment(base_img=resized_im, mat_img=seg_map, filename_d=filename_r)

    return save_file_path


if __name__ == '__main__':

    model = DeepLabModel()

    if INPUT_DIR is None or OUTPUT_DIR is None:
        print("Bad parameters. Please specify input dir path and output dir path")
        exit(1)

    files = os.listdir(INPUT_DIR)
    for file in tqdm.tqdm(files, ascii=True, desc='Remove Background', unit='|image|'):
        filename = file
        file = INPUT_DIR + '/' + file
        file_path = run_visualization(filepath=file, filename_r=filename)

        print("Save {}".format(file_path))
