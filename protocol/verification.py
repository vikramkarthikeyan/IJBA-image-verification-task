import pandas as pd
import config

def get_training_template_map(protocol):
    template_info = {}

    for i, row in protocol.iterrows():
        template_id = row['TEMPLATE_ID']
        
        if template_id not in template_info:
            template_info[template_id] = {}
            template_info[template_id]['locations'] = []
            template_info[template_id]['subject'] = row['SUBJECT_ID']

        file_location = config.BASE_PATH + config.IMAGES_PATH + str(row['SUBJECT_ID']) + '/' + row['FILE']

        template_info[template_id]['locations'].append(file_location)
    
    return template_info

def convert_to_template_pairs(protocol):
    pairs = []

    for i, row in protocol.iterrows():
        pairs.append((row[0], row[1]))

    return pairs

def get_training_templates(split):
    # Get training template info for corresponding split
    TRAINING_INFO_PATH = config.BASE_PATH + config.SPLIT_PATH + config.SPLIT_PATHS[split]

    training_protocol = pd.read_csv(TRAINING_INFO_PATH + config.TRAIN_CSV_PREFIX + str(split)+ '.csv')

    templates = get_training_template_map(training_protocol)
    return templates

def get_validation_pairs(split):
    # Get validation pair info for corresponding split
    VALIDATION_INFO_PATH = config.BASE_PATH + config.SPLIT_PATH + config.SPLIT_PATHS[split]

    validation_protocol = pd.read_csv(VALIDATION_INFO_PATH + config.VERIFY_CSV_PREFIX + str(split)+ '.csv', header=None)

    validation_pairs = convert_to_template_pairs(validation_protocol)

    return validation_pairs
