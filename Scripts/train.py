import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from model import Network

train_on_gpu = True if torch.cuda.is_available() else False

dhcd_model = Network()

if train_on_gpu:
  print("Training on GPU...")
  dhcd_model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(dhcd_model.parameters(), lr=0.001, momentum=0.9)

n_epochs = 55
train_losses = []
valid_losses = []
valid_loss_min = np.inf

for e in range(n_epochs):
    train_loss = 0
    valid_loss = 0
    
    dhcd_model.train()
    
    for img, label in train_loader:
        
        if train_on_gpu:
            img = img.cuda()
            label = label.cuda()
        
        optimizer.zero_grad()
        
        predicted_label = dhcd_model(img)
        loss = criterion(predicted_label, label)
        loss.backward()
        optimizer.step()
        train_loss = train_loss + loss.item()
    
    dhcd_model.eval()
    
    for img, label in valid_loader:
        
        if train_on_gpu:
            img = img.cuda()
            label = label.cuda()
        predicted_label = dhcd_model(img)
        loss = criterion(predicted_label, label)
        
        valid_loss = valid_loss + loss.item()
    
    train_loss = train_loss/len(train_loader)
    train_losses.append(train_loss)
    valid_loss = valid_loss/len(valid_loader)
    valid_losses.append(valid_loss)
    
    print("Epoch: {} Train Loss: {} Valid Loss: {}".format(e+1, train_loss, valid_loss))
    
    if valid_loss < valid_loss_min:
        print("Validation Loss Decreased From {} to {}".format(valid_loss_min, valid_loss))
        valid_loss_min = valid_loss
        torch.save(dhcd_model.state_dict(), "dhcd_model_8_March_2020.pth")
        print("Saving Best Model")
    
        
