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
        try:
            image = io.imread(self.split_frame.iloc[idx]['image_location'])
        except:
            print("Image not found..so returning empty entry")
            return {}
        

        sample = {'image': image, 'subject': self.split_frame.iloc[idx]['subject']}

        # if self.transform:
        #     sample = self.transform(sample)

        return sample
