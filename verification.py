import pandas as pd
import argparse
import torch.nn as nn
import model
import torch
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from protocol import verification
from tqdm import tqdm
from protocol import IJBADataset
from protocol import IJBAVerification
from protocol import Trainer
from torchsummary import summary
from protocol import EarlyStopping
from resnet50 import resnet50


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
            if os.path.exists(image):
                rows.append([subject, image])
                subjects.add(subject)
    
    rows = pd.DataFrame(rows, columns=['subject', 'image_location'])

    return rows, subjects

def plot_ROC(fpr, tpr, roc_auc, split_number):
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange',
            lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for split:' +str(split_number))
    plt.legend(loc="lower right")
    plt.savefig('./ROC/split_' + str(split_number) + '.png')

def save_results(pairs, actual_scores, cosine_scores, split_number):
    templates_1 = []
    templates_2 = []
    for (template_1, template_2) in pairs:
        templates_1.append(template_1)
        templates_2.append(template_2)
    
    d = {'template_1': templates_1, 'template_2': templates_2, 'actual': actual_scores, 'cosine': cosine_scores}
    
    df = pd.DataFrame(data=d)

    df.to_csv('./ROC/split_' + str(split_number) + '.csv')

# For ROC: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html
if __name__ == "__main__":
    split_number = args.split

    # Get training set info from the respective csv file
    template_directories = verification.get_training_templates(split_number)
    samples, subjects = generate_training_samples(template_directories)

    # Get Verification pairs
    metadata = verification.get_validation_metadata(split_number)
    pairs = verification.get_validation_pairs(split_number, metadata)

    # Create IJBA dataset object for training
    training_set = IJBADataset.IJBADataset(samples)

    # Create IJBA dataset object for verification
    verification_set = IJBAVerification.IJBAVerification(pairs, metadata)

    # Initialize model
    # model = model.MyModel(num_classes=len(subjects))
    model = resnet50(load=True, num_classes=len(subjects))

    # num_ftrs = model.fc.in_features

    # Re-Initialize the output layer to the number of subjects in split
    # model.fc = nn.Sequential(
    #         nn.Linear(num_ftrs, len(subjects)),
    #         nn.Softmax(1)
    #     )
    

    print("\nChecking if a GPU is available...")
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model = model.cuda()
        print ("Using GPU")
    else:
        print ("Using CPU as GPU is unavailable")   

    # Get saved model
    model_name = './models/split_' + str(split_number) + '_checkpoint.pth.tar'
    if use_gpu:
        checkpoint = torch.load(model_name)
    else:
        checkpoint = torch.load(model_name, map_location='cpu')
    
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    print("\nLoaded saved model for split:", str(split_number))

    # Initialize Trainer and Data Loader for training
    trainer = Trainer.Trainer(training_set, verification_set, subjects)

    # Run the verification protocol
    similarity_scores, actual_scores = trainer.validate(model, 1, use_gpu)

    # Compute ROC, TPR AND FPR
    fpr, tpr, thresholds = metrics.roc_curve(actual_scores, similarity_scores, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)

    # Plot the ROC curve
    plot_ROC(fpr, tpr, roc_auc, split_number)

    # Save the results as csv
    save_results(pairs, actual_scores, similarity_scores, split_number)

    


    