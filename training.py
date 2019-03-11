import pandas as pd
import argparse

from protocol import verification
from tqdm import tqdm
from protocol import IJBADataset
from protocol import Trainer

parser = argparse.ArgumentParser()

parser.add_argument("--split", help="The split number in the dataset",
                    type=int)

args = parser.parse_args()


def generate_training_samples(template_directories):
    rows = []
    subjects = set()
    for template in template_directories:
        images = template_directories[template]['locations']
        subject = template_directories[template]['subject']
        for image in images:
            rows.append([subject, image])
            subjects.add(subject)
    
    rows = pd.DataFrame(rows, columns=['subject', 'image_location'])

    return rows, subjects


if __name__ == "__main__":
    split_number = args.split

    # Get training set info from the respective csv file
    template_directories = verification.get_training_templates(split_number)
    samples, subjects = generate_training_samples(template_directories)

    # Create IJBA dataset object
    training_set = IJBADataset.IJBADataset(samples)

    # Initialize Trainer and Data Loader for training
    trainer = Trainer.Trainer(training_set, subjects)

