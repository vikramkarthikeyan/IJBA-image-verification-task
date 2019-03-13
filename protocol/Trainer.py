import matplotlib.pyplot as plt
import time
import numpy as np
import shutil
import os
import argparse
import cv2
import torch
import torch.nn as nn

from torchvision import datasets, transforms
from torchvision import models
from torch.autograd import Variable
from torchsummary import summary
from . import EarlyStopping
from . import AverageMeter
from sklearn.metrics.pairwise import cosine_similarity



# Followed PyTorch's ImageNet documentation
# https://github.com/pytorch/examples/blob/master/imagenet/main.py
class Trainer:

    def __init__(self, training_data, validation_data, classes, training_batch_size=128, validation_batch_size=1): 

        # Create training dataloader
        self.train_loader = torch.utils.data.DataLoader(training_data, batch_size=training_batch_size, shuffle=True,
                                                             num_workers=5)

        # Create validation dataloader
        self.validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=validation_batch_size, shuffle=False,
                                                             num_workers=5)

        self.classes = classes
        self.num_classes = len(classes)
        self.early_stopper = EarlyStopping.EarlyStopper()
    
    def convert_subjects_to_classes(self, target, subject_class_map):
        result = []
        for t in target.numpy():
            result.append(subject_class_map[t])
        return Variable(torch.from_numpy(np.array(result)), requires_grad=False)

    
    def train(self, model, criterion, optimizer, epoch, usegpu, subject_class_map, class_subject_map):
        batch_time = AverageMeter.AverageMeter()
        losses = AverageMeter.AverageMeter()
        top1 = AverageMeter.AverageMeter()
        top5 = AverageMeter.AverageMeter()

        # switch to train mode
        model.train()

        start = time.time()

        torch.cuda.empty_cache()

        for i, (data, target) in enumerate(self.train_loader):

            data, target = Variable(data), Variable(target, requires_grad=False)

            target = self.convert_subjects_to_classes(target, subject_class_map)

            if usegpu:
                data = data.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

            # Compute Model output
            output = model(data)

            # Compute Loss
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = self.accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), data.size(0))
            top1.update(acc1[0], data.size(0))
            top5.update(acc5[0], data.size(0))

            # Clear(zero) Gradients for theta
            optimizer.zero_grad()

            # Perform BackProp wrt theta
            loss.backward()

            # Update theta
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - start)
            start = time.time()

            print('\rTraining - Epoch [{:04d}] Batch [{:04d}/{:04d}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                        epoch, i, len(self.train_loader), batch_time=batch_time,
                        loss=losses), end="")
        
        print("\nTraining Accuracy: Acc@1: {top1.avg:.3f}%, Acc@5: {top5.avg:.3f}%".format(top1=top1, top5=top5))

    def normalize(self, features):
        norm = np.linalg.norm(features)
        if norm == 0: 
            return features
        return features/norm

    def validate(self, model, epoch, usegpu):
        batch_time = AverageMeter.AverageMeter()
        losses = AverageMeter.AverageMeter()
        accuracy = AverageMeter.AverageMeter()
        top1 = AverageMeter.AverageMeter()
        top5 = AverageMeter.AverageMeter()

        # switch to evaluate mode
        model.eval()

        validation_loss = 0
        correct_predictions = 0
        validation_size = len(self.validation_loader.dataset)

        print("Running Verification Protocol")

        similarity_scores = []

        with torch.no_grad():
            end = time.time()
            for i, (template_1, template_2, subject_1, subject_2, template_n1, template_n2) in enumerate(self.validation_loader):

                correct_predictions_epoch = 0

                template_1 = Variable(template_1[0])
                template_2 = Variable(template_2[0])

                if usegpu:
                    template_1 = template_1.cuda(non_blocking=True)
                    template_2 = template_2.cuda(non_blocking=True)

                # Compute outputs of two templates
                output_1 = model.features(template_1)
                output_2 = model.features(template_2)

                # Compute average of all the feature vectors into a single feature vector
                output_1 = np.average(output_1.cpu().numpy(), axis=0).flatten().reshape(1, -1)
                output_2 = np.average(output_2.cpu().numpy(), axis=0).flatten().reshape(1, -1)

                # Normalize features before computing similarity
                output_1 = self.normalize(output_1)
                output_2 = self.normalize(output_2)

                # Compute Cosine Similarity
                similarity = cosine_similarity(output_1, output_2)

                similarity_scores.append(similarity[0][0])

                print("\rTemplate 1:{:04d}, Template 2:{:04d}, Subject 1:{:04d}, Subject 2:{:04d} - Similarity: {}".format(template_n1[0], template_n2[0], subject_1[0], subject_2[0], similarity[0][0]),end="")

        return similarity_scores


    def save_checkpoint(self, state, filename='./models/checkpoint.pth.tar'):
        torch.save(state, filename)
    
    # Used - https://github.com/pytorch/examples/blob/master/imagenet/main.py
    def accuracy(self, output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res
