import pandas as pd
import argparse
import torch.nn as nn
import model
import torch
import os

from protocol import verification
from tqdm import tqdm
from protocol import IJBADataset
from protocol import IJBAVerification
from protocol import Trainer
from torchsummary import summary
from protocol import EarlyStopping

parser = argparse.ArgumentParser()

parser.add_argument("--split", help="The split number in the dataset",
                    type=int)

args = parser.parse_args()

# Hyperparameters
LR = 0.01
SGD_MOMENTUM = 0.9
EPOCHS = 15

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

def generate_subject_class_mapping(subjects):
    subject_class_map = {}
    class_subject_map = {}
    class_number = 0
    for subject in subjects:
        subject_class_map[subject] = class_number
        class_subject_map[class_number] = subject
        class_number += 1

    return subject_class_map, class_subject_map


if __name__ == "__main__":
    split_number = args.split

    # Get training set info from the respective csv file
    template_directories = verification.get_training_templates(split_number)
    samples, subjects = generate_training_samples(template_directories)

    # Generate subject class mapping for training data
    subject_class_map, class_subject_map = generate_subject_class_mapping(subjects)

    # Get Verification pairs
    metadata = verification.get_validation_metadata(split_number)
    pairs = verification.get_validation_pairs(split_number, metadata)

    # Create IJBA dataset object for training
    training_set = IJBADataset.IJBADataset(samples)

    # Create IJBA dataset object for validation
    validation_set = IJBAVerification.IJBAVerification(pairs, metadata)

    # Initialize model
    # model = model.Base_CNN(num_classes=len(subjects))
    model = model.MyModel(num_classes=len(subjects))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=SGD_MOMENTUM)

    print("\nChecking if a GPU is available...")
    use_gpu = torch.cuda.is_available()
    # Initialize new model
    if use_gpu:
        model = model.cuda()
        print ("Using GPU")
    else:
        print ("Using CPU as GPU is unavailable")  

    # Initialize Trainer and Data Loader for training
    trainer = Trainer.Trainer(training_set, validation_set, subjects)

    # Train the model
    summary(model, (3, 202, 203))

    highest_accuracy = 0
    highest_accuracy_5 = 0

    print("\nInitiating training...\n")

    model_name = './models/split_' + str(split_number) + '_checkpoint.pth.tar'
    best_model_name = './models/best_split_' + str(split_number) + '_model.pth.tar'

    for epoch in range(0, EPOCHS):
    
        # Train for one Epoch
        trainer.train(model, criterion, optimizer, epoch, use_gpu, subject_class_map, class_subject_map)

        if epoch%5 == 0:
            # Evaluate on the verification set every 5 epochs
            similarity_scores, actual_scores = trainer.validate(model, epoch, use_gpu)

        # Checkpointing the model after every epoch
        trainer.save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'best_accuracy': 0,
                        'optimizer' : optimizer.state_dict(),
        }, model_name)

        # If this epoch's model proves to be the best till now, save it as best model
        # if accuracy == max(accuracy, self.highest_accuracy):
        #     self.highest_accuracy = accuracy
        #     self.highest_accuracy_5 = accuracy_5
        #     self.trainer.save_checkpoint({
        #             'epoch': epoch + 1,
        #             'state_dict': self.model.state_dict(),
        #             'best_accuracy': self.highest_accuracy,
        #             'optimizer' : self.optimizer.state_dict()
        #     }, best_model_name)



