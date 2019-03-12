from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
from PIL import Image
import torchvision.transforms as transforms
import traceback
import os
import torch

# https://www.cs.virginia.edu/~vicente/recognition/notebooks/image_processing_lab.html
class IJBAVerification(Dataset):

    def __init__(self, data_frame, metadata):

        self.split_frame = data_frame
        self.metadata = metadata
        self.pil2tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.split_frame)

    def __getitem__(self, idx):
        try:
            template_1 = self.split_frame[idx][0]
            template_2 = self.split_frame[idx][1]

            subject_1 = self.metadata[template_1]['subject']
            subject_2 = self.metadata[template_2]['subject']

            # Open and consolidate all images in single template in one array
            images_1 = []
            images_2 = []

            for image in self.metadata[template_1]['locations']:
                with open(image, 'rb') as f:
                    temp = Image.open(f)
                    temp = temp.convert('RGB')
                    images_1.append(self.pil2tensor(temp))
            
            for image in self.metadata[template_2]['locations']:
                with open(image, 'rb') as f:
                    temp = Image.open(f)
                    temp = temp.convert('RGB')
                    images_2.append(self.pil2tensor(temp))

        except Exception:
            print(traceback.format_exc())
            return ([],[], subject_1, subject_2)
    
        images_1 = torch.stack(images_1)
        images_2 = torch.stack(images_2)

        return (images_1, images_2, subject_1, subject_2, template_1, template_2)
