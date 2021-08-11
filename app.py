import numpy as np
import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from model import Network

train_on_gpu = torch.cuda.is_available()

def pil_loader(pth):
    with open(pth, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def TestPath():
    data_dir = 'C:\\Users\\91865\\Desktop\\Deva\\DevanagariHandwrittenCharacterDataset'
    train_dir = os.path.join(data_dir, 'Train/')
    test_dir = os.path.join(data_dir, 'Test/')

    test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, ))
        ])

        # train_data = datasets.ImageFolder(train_dir, transform=train_transform)
    test_data = datasets.ImageFolder(test_dir, transform=test_transform)
    return test_data


st.write('hushvr')

uploaded_file = st.file_uploader("Upload Files",type=['png','jpg','jpeg'])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.write(uploaded_file)
    st.image(img,width=250)
#     file_details = {"FileName":uploaded_file.name,"FileType":uploaded_file.type,"FileSize":uploaded_file.size}
#     st.write(file_details)


test_data = TestPath()
# st.write(test_data.classes)

dhcd_model = Network()
dhcd_model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(dhcd_model.parameters(), lr=0.001, momentum=0.9)

dhcd_model.load_state_dict(torch.load("Models\model_98.pt"))

tfms =  transforms.Compose([
        transforms.Scale((32,32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5))])

img_tensor = tfms(img).to('cuda').unsqueeze(0)
st.write(img_tensor[0].shape)

if train_on_gpu:
    img_tensor = img_tensor.cuda()

# get sample outputs
output = dhcd_model(img_tensor)
# convert output probabilities to predicted class
_, preds_tensor = torch.max(output, 1)
preds = np.squeeze(preds_tensor.numpy()) if not train_on_gpu else np.squeeze(preds_tensor.cpu().numpy())

st.write(test_data.classes[preds_tensor[0]])