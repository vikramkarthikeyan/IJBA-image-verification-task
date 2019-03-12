import pandas as pd
import argparse
import torch.nn as nn
import model
import torch
import os

from protocol import verification
from tqdm import tqdm
from protocol import IJBADataset
from protocol import Trainer
from torchsummary import summary


parser = argparse.ArgumentParser()

parser.add_argument("--split", help="The split number in the dataset",
                    type=int)

args = parser.parse_args()

# Hyperparameters
LR = 0.01
SGD_MOMENTUM = 0.9
WEIGHT_DECAY = 0.00001
EPOCHS = 50

def generate_training_samples(template_directories):
    rows = []
    subjects = set()
    for template in template_directories:
        images = template_directories[template]['locations']
        subject = template_directories[template]['subject']
        for image in images:
            if os.path.exists(image):
                rows.append([subject, image])
                subjects.add(subject)
    
    rows = pd.DataFrame(rows, columns=['subject', 'image_location'])

    return rows, subjects


if __name__ == "__main__":
    split_number = args.split

    print("\nChecking if a GPU is available...")
    use_gpu = torch.cuda.is_available()
    # Initialize new model
    if self.use_gpu:
    model = model.cuda()
        print ("Using GPU")
    else:
        print ("Using CPU as GPU is unavailable")    

    # Get training set info from the respective csv file
    template_directories = verification.get_training_templates(split_number)
    samples, subjects = generate_training_samples(template_directories)

    # Create IJBA dataset object
    training_set = IJBADataset.IJBADataset(samples)

    # Initialize model
    model = model.Base_CNN(num_classes=len(subjects))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=SGD_MOMENTUM, weight_decay=WEIGHT_DECAY)

    # Initialize Trainer and Data Loader for training
    trainer = Trainer.Trainer(training_set, subjects)

    # Train the model
    summary(model, (3, 202, 203))
    trainer.train(model, criterion, optimizer, EPOCHS, use_gpu)


