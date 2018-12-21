import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
from collections import OrderedDict
import json
import PIL
from PIL import Image
import argparse

arch = {"vgg16":25088}

def load_data(where  = "./flowers" ):

    data_dir = where
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    train_transforms = transforms.Compose([transforms.RandomRotation(50),
                                               transforms.RandomResizedCrop(224),
                                               transforms.RandomHorizontalFlip(),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406],
                                                                    [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406],
                                                                   [0.229, 0.224, 0.225])])

    validation_transforms = transforms.Compose([transforms.Resize(256),
                                                    transforms.CenterCrop(224),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                                         [0.229, 0.224, 0.225])])
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    validation_data = datasets.ImageFolder(valid_dir, transform=validation_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validationloader = torch.utils.data.DataLoader(validation_data, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)

    return trainloader,validationloader,testloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_model(structure='vgg16',dropout=0.5,lr = 0.003,power=gpu):



    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(OrderedDict([
                                ('fc1',nn.Linear(25088, 250)),
                                ('relu1',nn.ReLU()),
                                ('Dropout',nn.Dropout(0.5)),
                                ('fc2',nn.Linear(250, 150)),
                                ('relu2',nn.ReLU()),
                                ('Dropout',nn.Dropout(0.2)),
                                ('fc3',nn.Linear(150, 102)),
                                ('output',nn.LogSoftmax(dim=1))]))

    model.to(device)
    criterion = nn.NLLLoss()
# Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)

    return model, criterion, optimizer



def train_network(model, criterion, optimizer, epochs = 12,loader=trainloader, print_every=5, power='gpu'):

    steps = 0
    running_loss = 0
    for epoch in range(epochs):
        for inputs, labels in loader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        test_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Test loss: {test_loss/len(testloader):.3f}.. "
                      f"Test accuracy: {accuracy/len(testloader):.3f}")
                running_loss = 0
                model.train()


def save_checkpoint(path='checkpoint.pth',structure ='vgg16',dropout=0.5,lr=0.003,epochs=12):
    '''
    Arguments: The saving path and the hyperparameters of the network
    Returns: Nothing
    This function saves the model at a specified by the user path
    '''
    model.class_to_idx = train_data.class_to_idx
    model.cpu
    torch.save({'structure' :'vgg16',
            'state_dict':model.state_dict(),
            'class_to_idx':model.class_to_idx},
            path)



def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    structure = checkpoint['structure']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])



def process_image(image_path):
    '''
    Arguments: The image's path
    Returns: The image as a tensor
    This function opens the image usign the PIL package, applies the  necessery transformations and returns the image as a tensor ready to be fed to the network
    '''

    for i in image_path:
        path = str(i)
    img = Image.open(i) # Here we open the image

    make_img_good = transforms.Compose([ # Here as we did with the traini ng data we will define a set of
        # transfomations that we will apply to the PIL image
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    tensor_image = make_img_good(img)

    return tensor_image


def predict(image_path, model, topk=5,power='gpu'):
    '''
    Arguments: The path to the image, the model, the number of prefictions and whether cuda will be used or not
    Returns: The "topk" most probable choices that the network predicts
    '''

    model.to(device)

    img_torch = process_image(image_path)
    img_torch = img_torch.unsqueeze_(0)
    img_torch = img_torch.float()

    if power == 'gpu':
        with torch.no_grad():
            output = model.forward(img_torch.cuda())
    else:
        with torch.no_grad():
            output=model.forward(img_torch)

    probability = F.softmax(output.data,dim=1)

    return probability.topk(topk)
