import pandas as pd
import argparse

from protocol import verification
from tqdm import tqdm
from protocol import IJBADataset

parser = argparse.ArgumentParser()

parser.add_argument("--split", help="The split number in the dataset",
                    type=int)

args = parser.parse_args()


def generate_training_samples(template_directories):
    rows = []
    for template in template_directories:
        images = template_directories[template]['locations']
        subject = template_directories[template]['subject']
        for image in images:
            rows.append([subject, image])
    
    rows = pd.DataFrame(rows, columns=['subject', 'image_location'])

    return rows


if __name__ == "__main__":
    split_number = args.split

    template_directories = verification.get_training_templates(split_number)

    samples = generate_training_samples(template_directories)

    # Create IJBA dataset object
    training_set = IJBADataset.IJBADataset(samples)

