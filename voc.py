import os
import csv
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm import tqdm
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt 
import cv2
from data_aug import *
class VOCDataset(Dataset):

    def __init__(self,mode, files_dir, annotations_dir ,transform, files, S=7, B=2, C=20):
        super(VOCDataset).__init__()
        self.mode = mode
        self.files_dir = files_dir
        self.annotations_dir = annotations_dir 
        self.transform = transform
        self.files = files
        self.label_dictionary = {'person':0, 'bird':1, 'cat':2, 'cow':3, 'dog':4, 'horse':5, 'sheep':6, 'aeroplane':7,  'bicycle':8, 'boat':9, 'bus':10, 'car':11, 'motorbike':12,  'train':13, 'bottle':14, 'chair':15, 'diningtable':16, 'pottedplant':17, 'sofa':18, 'tvmonitor':19}
        self.S = S
        self.B = B
        self.C = C

    def __len__(self):
        return len(self.files)

    def __getitem__(self,idx):
        filexml = str(self.files[idx]) + ".xml"
        
        bboxes = []
        for annotation in self.annotations_dir:
            if os.path.exists(str(os.path.join(annotation, filexml))):
                tree = ET.parse(os.path.join(annotation, filexml))
        root = tree.getroot()
        filename = root.find('filename').text
        #print(filename)

        for member in root.findall('object'):
            
            name = member.find('name').text
            label = self.label_dictionary[name]
        
            xmin = float(member.find('bndbox').find('xmin').text)
            xmax = float(member.find('bndbox').find('xmax').text)
            img_width = float(root.find('size').find('width').text)
        
            ymin = float(member.find('bndbox').find('ymin').text)
            ymax = float(member.find('bndbox').find('ymax').text)
            img_height = float(root.find('size').find('height').text)
            
            centerx = ((xmax + xmin) / 2) / img_width
            centery = ((ymax + ymin) / 2) / img_height
            boxwidth = (xmax - xmin) / img_width
            boxheight = (ymax - ymin) / img_height
            
            #boxes.append([label, centerx, centery, boxwidth, boxheight])
            bboxes.append([xmin, ymin, xmax, ymax,label,img_width,img_height])

        #boxes = torch.tensor(boxes)
        #image = Image.open(os.path.join(self.files_dir, filename))
        #image = image.convert("RGB")
        transferbboxes = []

        image = None
        for dir in self.files_dir:
            if os.path.exists(os.path.join(dir, filename)):
                image = cv2.imread(os.path.join(dir, filename))
        bboxes = np.array(bboxes)
        if self.mode == "train":
            image, bboxes = RandomHorizontalFlip(0.5)(image.copy(), bboxes.copy())
            #image, bboxes = RandomScale(0.3, diff = True)(image.copy(), bboxes.copy())
            #image, bboxes = RandomTranslate(0.3, diff = True)(image.copy(), bboxes.copy())
            image, bboxes = RandomHSV(100, 100, 100)(image.copy(), bboxes.copy())
            image, bboxes = Resize(448)(image.copy(), bboxes.copy())
            for index,box in enumerate(bboxes):
                xmin, ymin, xmax, ymax,label,img_width,img_height = bboxes[index]
                xmin = min(447,max(xmin,0))
                ymin = min(447,max(ymin,0))
                xmax = min(447,max(xmax,0))
                ymax = min(447,max(ymax,0))
                centerx = ((xmax + xmin) / 2) / 448
                centery = ((ymax + ymin) / 2) / 448
                boxwidth = (xmax - xmin) / 448
                boxheight = (ymax - ymin) / 448
                transferbboxes.append([label, centerx, centery, boxwidth, boxheight])
        else:
            image, bboxes = Resize(448)(image.copy(), bboxes.copy())
            for index,box in enumerate(bboxes):
                xmin, ymin, xmax, ymax,label,img_width,img_height = bboxes[index]
                centerx = ((xmax + xmin) / 2) / 448
                centery = ((ymax + ymin) / 2) / 448
                boxwidth = (xmax - xmin) / 448
                boxheight = (ymax - ymin) / 448
                transferbboxes.append([label, centerx, centery, boxwidth, boxheight])


        if self.transform:
            # image = self.transform(image)
            image, transferbboxes = self.transform(image), transferbboxes
            #print(image.shape)
        #print(boxes)
        # Convert To Cells
        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))
        for box in transferbboxes:
            class_label, x, y, width, height = box
            class_label = int(class_label)

            # i,j represents the cell row and cell column
            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i

            """
            Calculating the width and height of cell of bounding box,
            relative to the cell is done by the following, with
            width as the example:
            
            width_pixels = (width*self.image_width)
            cell_pixels = (self.image_width)
            
            Then to find the width relative to the cell is simply:
            width_pixels/cell_pixels, simplification leads to the
            formulas below.
            """
            width_cell, height_cell = (
                width * self.S,
                height * self.S,
            )

            # If no object already found for specific cell i,j
            # Note: This means we restrict to ONE object
            # per cell!
#             print(i, j)
            if label_matrix[i, j, self.C] == 0:
                # Set that there exists an object
                label_matrix[i, j, self.C] = 1

                # Box coordinates
                box_coordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )

                label_matrix[i, j, 21:25] = box_coordinates

                # Set one hot encoding for class_label
                label_matrix[i, j, class_label] = 1

        #print(np. where(label_matrix == 1))
        return image, label_matrix

