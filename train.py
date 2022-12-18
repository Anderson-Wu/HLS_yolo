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

from yolov1 import *
from voc import *
from utils import *

device = "cuda" if torch.cuda.is_available() else "cpu"

myseed = 93830  # set a random seed for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

learning_rate = 2e-5
batch_size = 32 # 64 in original paper but resource exhausted error otherwise.
weight_decay = 0.0005
epochs = 300
num_workers = 0

train_dir = ['./VOCdevkit/VOC2007/JPEGImages','./VOCdevkit/VOC2012/JPEGImages']
val_dir = ['./VOCdevkit/VOC2007/JPEGImages']
train_txt = ['./VOCdevkit/VOC2007/ImageSets/Main/trainval.txt','./VOCdevkit/VOC2012/ImageSets/Main/trainval.txt']
val_txt = ['./VOCdevkit/VOC2007/ImageSets/Main/val.txt']
annotations_dir = ['./VOCdevkit/VOC2007/Annotations','./VOCdevkit/VOC2012/Annotations']

PIN_MEMORY = True
LOAD_MODEL = False
LOAD_MODEL_FILE = "yolov1.pth"

def val_fn(val_loader, model, loss_fn):
    
    loop = tqdm(val_loader, leave=True)
    mean_loss = []
    
    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        
        loop.set_postfix(loss = loss.item())
        
    print(f"val Mean loss was {sum(mean_loss) / len(mean_loss)}")

def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []
    
    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loop.set_postfix(loss = sum(mean_loss) / len(mean_loss))
        
    print(f"train Mean loss was {sum(mean_loss) / len(mean_loss)}")








def main():

    #launch(sys.argv[1:])



    model = YOLOv1(S=7, B=2, C=20).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model.to(device)
    #print(model)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=0.1, patience=3, mode='max', verbose=True)
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[round(epochs * x) for x in (0.8, 0.9)], gamma=0.1)

    loss_fn = YoloLoss()

    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)

    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    transform_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    train_file = []
    for textfile in train_txt:
        f = open(textfile)
        for line in f:
            train_file.append(line.replace('\n', ''))
        f.close()

    val_file = []
    for textfile in val_txt:
        f = open(textfile)
        for line in f:
            val_file.append(line.replace('\n', ''))
        f.close()


    

    train_dataset = VOCDataset(
        mode="train",
        files_dir=train_dir,
        annotations_dir=annotations_dir,
        transform=transform_train,
        files=train_file
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )

    val_dataset = VOCDataset(
        mode="val",
        files_dir=val_dir,
        annotations_dir=annotations_dir,
        transform=transform_val, 
        files=val_file
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=PIN_MEMORY,
        shuffle=False,
        drop_last=False,
    )

    best_map = 0
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1})")
        model.train()
        train_fn(train_loader, model, optimizer, loss_fn)
        

            
        
        model.eval()
        val_fn(val_loader, model, loss_fn)
        
        pred_boxes, target_boxes = get_bboxes(
            val_loader, model, iou_threshold=0.5, threshold=0.4, is_val=True
        )

        mean_avg_prec = mean_average_precision(
            pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
        )
        print(f"Val mAP: {mean_avg_prec:.5f}")

        if mean_avg_prec > best_map:
            print(f"save model")
            best_map = mean_avg_prec
            torch.save(model.state_dict(), f"best_map.pt")
        
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    save_checkpoint(checkpoint, filename=LOAD_MODEL_FILE)

def inference():
    model = YOLOv1(S=7, B=2, C=20).to(device)
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    loss_fn = YoloLoss()

    transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor()
    ])

    
    load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)

    f = open(val_txt)
    test_file = []
    for line in f:
        test_file.append(line.replace('\n', ''))
    f.close()

    test_dataset = VOCDataset(
        files_dir=val_dir,
        annotations_dir=annotations_dir,
        transform=transform, 
        files=test_file
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=PIN_MEMORY,
        shuffle=False,
        drop_last=False,
    )
    
    model.eval()
    train_fn(test_loader, model, optimizer, loss_fn)
    
    pred_boxes, target_boxes = get_bboxes(
        test_loader, model, iou_threshold=0.5, threshold=0.4
    )

    mean_avg_prec = mean_average_precision(
        pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
    )
    print(f"Test mAP: {mean_avg_prec:.5f}")

if __name__ == "__main__":
    main()
    #infernce()