import os
dir_path = os.path.dirname(os.path.realpath(__file__))

dir_path = dir_path.replace('protocol', 'data/')

BASE_PATH = dir_path

SPLIT_PATH = 'IJB-A_11_sets/'

IMAGES_PATH = 'ijba_aligned_all/'

SPLIT_PATHS = ['','split1/', 'split2/', 'split3/', 'split4/', 'split5/', 'split6/', 'split7/', 'split8/', 'split9/', 'split10/']

TRAIN_CSV_PREFIX = 'train_'
VERIFY_CSV_PREFIX = 'verify_comparisons_'
VERIFY_METADATA_CSV_PREFIX = 'verify_metadata_'




