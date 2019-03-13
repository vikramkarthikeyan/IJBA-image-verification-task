import pandas as pd
import os
from . import config

def get_training_template_map(protocol):
    template_info = {}

    for i, row in protocol.iterrows():
        template_id = row['TEMPLATE_ID']

        file_location = config.BASE_PATH + config.IMAGES_PATH + str(row['SUBJECT_ID']) + '/' + row['FILE']
        
        if template_id not in template_info and os.path.exists(file_location):
            template_info[template_id] = {}
            template_info[template_id]['locations'] = []
            template_info[template_id]['subject'] = row['SUBJECT_ID']

        if os.path.exists(file_location):
            template_info[template_id]['locations'].append(file_location)
    
    return template_info

def convert_to_template_pairs(protocol, metadata):
    pairs = []

    for i, row in protocol.iterrows():
        if row[0] in metadata and row[1] in metadata:
            pairs.append((row[0], row[1]))

    return pairs

def get_training_templates(split):
    # Get training template info for corresponding split
    TRAINING_INFO_PATH = config.BASE_PATH + config.SPLIT_PATH + config.SPLIT_PATHS[split]

    training_protocol = pd.read_csv(TRAINING_INFO_PATH + config.TRAIN_CSV_PREFIX + str(split)+ '.csv')

    templates = get_training_template_map(training_protocol)
    return templates

def get_validation_pairs(split, metadata):
    # Get validation pair info for corresponding split
    VALIDATION_INFO_PATH = config.BASE_PATH + config.SPLIT_PATH + config.SPLIT_PATHS[split]

    validation_protocol = pd.read_csv(VALIDATION_INFO_PATH + config.VERIFY_CSV_PREFIX + str(split)+ '.csv', header=None)

    validation_pairs = convert_to_template_pairs(validation_protocol, metadata)

    return validation_pairs

def get_validation_metadata(split):
    # Get validation metadata info for corresponding split 
    VALIDATION_METADATA_INFO_PATH = config.BASE_PATH + config.SPLIT_PATH + config.SPLIT_PATHS[split]

    validation_metadata = pd.read_csv(VALIDATION_METADATA_INFO_PATH + config.VERIFY_METADATA_CSV_PREFIX + str(split)+ '.csv')

    metadata = get_training_template_map(validation_metadata)

    return metadata