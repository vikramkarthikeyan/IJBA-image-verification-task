import pandas as pd

from protocol import verification
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform

class IJBADataset(Dataset):

    def __init__(self, data_frame, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.split_frame = data_frame
        self.transform = transform

    def __len__(self):
        return len(self.split_frame)

    def __getitem__(self, idx):
        image = io.imread(self.split_frame.iloc[idx]['image_location'])

        sample = {'image': image, 'subject': self.split_frame.iloc[idx]['subject']}

        # if self.transform:
        #     sample = self.transform(sample)

        return sample


def generate_training_samples(template_directories):
    rows = []
    for template in template_directories:
        images = template_directories[template]['locations']
        subject = template_directories[template]['subject']
        for image in images:
            rows.append([subject, image])
    
    rows = pd.DataFrame(rows, columns=['subject', 'image_location'])

    return rows



template_directories = verification.get_training_templates(1)

samples = generate_training_samples(template_directories)

training_set = IJBADataset(samples)

for i in range(len(training_set)):
    print training_set[i]
