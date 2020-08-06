import numpy as np
import pandas as pd
import cv2
import os
import PIL
import random

import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
import torchvision.transforms as transforms

from efficientnet_pytorch import EfficientNet

def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(y_pred)

    results = (y_pred_tag.argmax(1) == y_test.argmax(1)).cpu().numpy()
    correct_results_sum = results.sum()
    correct_results_sum = correct_results_sum.sum()
    acc = correct_results_sum / y_test.shape[0]
    acc = acc * 100
    return acc


class DataGenerator(Dataset):
    def __init__(self, df, images_path,  _type='train', transform=None):
        self.df = df
        self.images_path = images_path
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        current = self.df.loc[idx]

        #gender = current["gender"]
        gender = np.zeros(5, dtype=np.float32)
        gender[current["gender"]] = 1

        #usage = current["usage"]
        usage = np.zeros(8, dtype=np.float32)
        usage[current["usage"]] = 1

        #masterCategory = current["masterCategory"]
        masterCategory = np.zeros(7, dtype=np.float32)
        masterCategory[current["masterCategory"]] = 1

        image_path = os.path.join(self.images_path, str(current["id"]) + ".jpg")

        f = open(image_path, "rb")
        chunk = f.read()
        chunk_arr = np.frombuffer(chunk, dtype=np.uint8)
        image = cv2.imdecode(chunk_arr, cv2.IMREAD_COLOR)
        f.close()

        #image = cv2.resize(image, (int(512), int(512)))

        #cv2.imshow("", image)
        #cv2.waitKey()
        image = image[:, :, ::-1].copy()

        #image = image.transpose(2, 0, 1)
        #image = image.astype(np.float)

        if self.transform:
            image = self.transform(image)

        if False:
            shared = image.cpu().numpy()
            shared = shared.transpose(1, 2, 0)
            shared = shared[:, :, ::-1].copy()
            #[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            shared[:, :, 0] = (shared[:, :, 0] * 0.225) + 0.406
            shared[:, :, 1] = (shared[:, :, 1] * 0.224) + 0.456
            shared[:, :, 2] = (shared[:, :, 2] * 0.229) + 0.485

            shared *= 255
            shared = shared.astype(np.uint8)
            cv2.imshow("", shared)
            cv2.waitKey()
        return image, gender, usage, masterCategory

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--pretrained', default='resnet34_epoch_3_batchs_saved_weights.pth', action='store_true',
                    help='use pre-trained model')

args = parser.parse_args()

categorical_map = {
    "gender": {
        'Boys': 0,
        'Men': 1,
        'Unisex': 2,
        'Women': 3,
        'Girls': 4
    },
    "usage": {
        'Casual': 0,
        'Ethnic': 1,
        'Formal': 2,
        'Home': 3,
        'Party': 4,
        'Smart Casual': 5,
        'Sports': 6,
        'Travel': 7
    },
    "masterCategory": {
        'Accessories': 0,
        'Apparel': 1,
        'Footwear': 2,
        'Free Items': 3,
        'Home': 4,
        'Personal Care': 5,
        'Sporting Goods': 6,
    }
}

dataset_path = "fashion-dataset"
images_path = os.path.join(dataset_path, "images")
df = pd.read_csv(dataset_path + "/styles.csv", sep=",", error_bad_lines=False)

SEED = 2020
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

IMAGE_SIZE = 224
batch_size = 16
epochs = 100

# Clear data set
for key in categorical_map:
    df = df.dropna(subset=[key])
df.reset_index(drop=True, inplace=True)

for key in categorical_map:
    for category in categorical_map[key]:
        df.loc[df[key] == category, key] = categorical_map[key][category]

    df[key] = df[key].astype(np.int)

#df = df.loc[:1000]


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = EfficientNet.from_name('efficientnet-b4', in_channels=3).to(device)

summary(model, (3, IMAGE_SIZE, IMAGE_SIZE))

#if args.pretrained != '':
 #   model.load_state_dict(torch.load(args.pretrained))

#optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.001)

optimizer = torch.optim.SGD(model.parameters(), 0.08,
                                momentum=0.9,
                                weight_decay=1e-4)

examples_size = len(df)
val_size = int(examples_size * 0.2)
indexes_stay = df.index.values
valid_inds = np.random.choice(indexes_stay, val_size, replace=False)
train_inds = np.setdiff1d(indexes_stay, valid_inds)

train = df.iloc[train_inds]
train.reset_index(drop=True, inplace=True)

valid = df.iloc[valid_inds]
valid.reset_index(drop=True, inplace=True)

transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMAGE_SIZE + 20, IMAGE_SIZE + 20), interpolation=PIL.Image.BICUBIC),
    transforms.RandomAffine(
            degrees=(-20, 20),
            scale=(0.8889, 1.0),
            shear=(-18, 18),
            fillcolor=(255, 255, 255)
    ),

    transforms.CenterCrop(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    #transforms.RandomVerticalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

transform_val = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

mse_criterion = nn.MSELoss()
softmax_criterion = nn.Softmax(dim=1)
cross_criterion = nn.CrossEntropyLoss()

for epoch in range(epochs):
    train_image = DataGenerator(train, images_path, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_image, batch_size=batch_size, shuffle=True)

    val_image = DataGenerator(valid, images_path, _type="evaluate", transform=transform_val)
    val_loader = torch.utils.data.DataLoader(val_image, batch_size=batch_size, shuffle=False)

    print("------------------------------------------------")
    print('Epochs {}/{} '.format(epoch + 1, epochs))

    for mode in ["train", "val"]:
        if mode == "train":
            print("Train")
            loader = train_loader
            model.train()
            torch.set_grad_enabled(True)

        else:
            loader = val_loader
            print("Validate")
            model.eval()
            torch.set_grad_enabled(False)

        gender_acc = 0
        usage_acc = 0
        category_acc = 0
        running_loss = 0

        for idx, (inputs, gender, usage, masterCategory) in enumerate(loader):

            inputs = inputs.to(device)

            gender = gender.to(device)
            usage = usage.to(device)
            masterCategory = masterCategory.to(device)
            #masterCategory = masterCategory.to(device).long()

            optimizer.zero_grad()
            outputs1, outputs2, outputs3 = model(inputs.float())
            loss1 = mse_criterion(outputs1, gender)
            loss2 = mse_criterion(outputs2, usage)
            loss3 = mse_criterion(outputs3, masterCategory)

            #loss1 = cross_criterion(outputs1, torch.max(gender, 1)[1])
            #loss2 = cross_criterion(outputs2, torch.max(usage, 1)[1])
            #loss3 = cross_criterion(outputs3, torch.max(masterCategory, 1)[1])
            #loss1 = cross_criterion(outputs1, gender)
            #loss2 = cross_criterion(outputs2, usage)
            #loss3 = cross_criterion(outputs3, masterCategory)


            running_loss += loss1 + loss2 + loss3

            gender_acc += binary_acc(outputs1, gender)
            usage_acc += binary_acc(outputs2, usage)
            category_acc += binary_acc(outputs3, masterCategory)

            #gender_acc += (outputs1.argmax(1) == gender).float().mean()
            #usage_acc += (outputs2.argmax(1) == usage).float().mean()
            #category_acc += (outputs2.argmax(1) == masterCategory).float().mean()


            if mode == 'train':
                #(loss1 + loss2 + loss3).backward()
                (loss3).backward()
                optimizer.step()

            if idx % 5 == 0:
                print("-------------------------------------")
                print(f'epoch: {epoch} idx: {idx}')
                print(f"gender accuracity {gender_acc / (idx + 1)} category loss {loss1} ")
                print(f"usage accuracity {usage_acc / (idx + 1)} category loss {loss2} ")
                print(f"category accuracity {category_acc / (idx + 1)} category loss {loss3} " )
                print("loss: ", loss1 + loss2 + loss3)

        print("-------------------------------------")
        print('gender accuracity  : {:.2f}%'.format(gender_acc / (len(loader))))
        print('usage accuracity : {:.2f}%'.format(usage_acc / (len(loader))))
        print('category accuracity : {:.2f}%'.format(category_acc / (len(loader))))
        print('loss : {:.4f}'.format(running_loss / len(loader)))

        if mode == "train":
            torch.save(model.state_dict(), f'resnet34_epoch_{epoch}_batchs_saved_weights.pth')

