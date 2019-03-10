import pandas as pd

from protocol import verification
from tqdm import tqdm
from protocol import IJBADataset
from skimage import io, transform



def generate_training_samples(template_directories):
    rows = []
    for template in template_directories:
        images = template_directories[template]['locations']
        subject = template_directories[template]['subject']
        for image in images:
            rows.append([subject, image])
    
    rows = pd.DataFrame(rows, columns=['subject', 'image_location'])

    return rows


for split_number in range(1,11):

    template_directories = verification.get_training_templates(split_number)

    samples = generate_training_samples(template_directories)

    training_set = IJBADataset.IJBADataset(samples)

    print(len(training_set))

# for i in range(len(training_set)):
#     print training_set[i]
