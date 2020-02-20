import os

from utils.folder_file_manager import make_directory_if_not_exists


CUR_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = make_directory_if_not_exists(os.path.join(CUR_DIR, 'data', 'output'))
INPUT_DIR = make_directory_if_not_exists(os.path.join(CUR_DIR, 'data', 'input'))
MASK_RCNN_DIR = make_directory_if_not_exists(os.path.join(CUR_DIR, 'utils', 'model', 'mask_rcnn_model'))

MODEL_DIR = make_directory_if_not_exists(os.path.join(CUR_DIR, 'utils', 'model'))
MODEL_PATH = os.path.join(MODEL_DIR, 'bg_removal_graph.pb')

MIN_RATIO = 0.1
