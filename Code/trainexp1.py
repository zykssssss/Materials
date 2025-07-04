import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from data_loader import *
from model import *
from tqdm import *
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
import logging
import sys
import pandas as pd


logging.basicConfig(filename='output.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

logging.getLogger('').addHandler(console_handler)

torch.manual_seed(2024)

# transform = transforms.Compose([
#     transforms.Resize(15),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
csv_path_to_train_data = './label_csv/label_train.CSV'
csv_path_to_val_data = './label_csv/label_val.CSV'

train_dataset = CustomDataset(csv_path_to_train_data, transform=None)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

val_dataset = CustomDataset(csv_path_to_val_data, transform=None)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=True)

# resnet = models.resnet50(pretrained=True)

# num_classes = len(train_dataset.classes)
# resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)

resnet = ResNet50(in_channels=4, num_classes=8)

print(resnet)

criterion = nn.CrossEntropyLoss()
def step_lr(ep):
    if ep < 30:
        lr = 0.001
    elif ep < 60:
        lr = 0.0005
    elif ep < 90:
        lr = 0.0001
    elif ep < 120:
        lr = 0.00001
    else:
        lr = 0.000005
    return lr

def print_confusion_matrix(targets, predictions):
    cm = confusion_matrix(targets, predictions)
    logging.info("Confusion Matrix:")
    logging.info(cm)
    
    accuracy = accuracy_score(targets, predictions)
    recall = recall_score(targets, predictions, average='macro')
    precision = precision_score(targets, predictions, average='macro')
    f1 = f1_score(targets, predictions, average='macro')

    logging.info("Accuracy: %f" % accuracy)
    logging.info("Recall: %f" % recall)
    logging.info("Precision: %f" % precision)
    logging.info("F1 Score: %f" % f1)


def get_constraint_variable(predicted, image_names):    
    df = pd.read_csv('./data.csv')
    labels_dict = {0: '2000-2300', 1: '1000-1900', 2: '2000-2700', 3: '750-1300', 
               4: '2650-3000', 5: '3000-3400', 6: '3400-3777', 7: '3400-3777'}
    value = 0
    for i, class_id in enumerate(predicted.tolist()):
        altitudeRange = labels_dict[class_id]
        min_alt, max_alt = altitudeRange.split('-')
        min_alt = int(min_alt)
        max_alt = int(max_alt)
        altitude = int(df.loc[df['ImageName'] == image_names[i], 'DEM'].values[0])
        if altitude > min_alt-100 and altitude < max_alt +100 :
            value += 0
        elif altitude < min_alt-100:
            value += abs(altitude - min_alt)
        else:
            value += abs(altitude - max_alt)
    #print(float(value)/1e5)
    constraint_variable = torch.tensor(float(value)/1e5, requires_grad=True)
    return constraint_variable


def get_label(name):
    df = pd.read_csv('./data.csv')

    image_name = name
    dem_value = df.loc[df['ImageName'] == image_name, 'DEM'].values[0]
    altitude = int(dem_value)
    labels_dict = {0: '2000-2300', 1: '1000-1900', 2: '2000-2700', 3: '750-1300', 
               4: '2650-3000', 5: '3000-3400', 6: '3400-3777', 7: '3400-3777'}
    for class_id in labels_dict.keys():
        altitudeRange = labels_dict[class_id]
        min_alt, max_alt = altitudeRange.split('-')
        min_alt = int(min_alt)
        max_alt = int(max_alt)
        if altitude > min_alt-100 and altitude < max_alt +100 :
            return class_id
        #elif altitude < min_alt-100:
        #    value += abs(altitude - min_alt)
        #else:
        #    value += abs(altitude - max_alt)
    return 8
    #return dem_value            
        

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet.to(device)

# 训练模型
num_epochs = 150
train_losses = []
train_acc = []
val_acc = []

for epoch in range(num_epochs):
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    learning_rate = step_lr(epoch)
    optimizer = torch.optim.Adam(resnet.parameters(), lr=learning_rate)

    resnet.train()
    
    for image_names, images, labels in tqdm(train_loader):
        images = images.to(torch.float32)
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = resnet(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        _, predicted = torch.max(outputs.data, 1)
        train_loss += loss.item()
        
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
        
    train_accuracy = 100.0 * train_correct / train_total
    train_loss /= len(train_loader)
    

    train_losses.append(train_loss)
    train_acc.append(train_accuracy/100)
    
    

    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    resnet.eval()
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for image_names, images, labels in val_loader:
            images = images.to(torch.float32)
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = resnet(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)

            
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            all_predictions.extend(predicted.tolist())
            all_targets.extend(labels.tolist())
    
    
    val_accuracy = 100.0 * val_correct / val_total
    val_loss /= len(val_loader)
    
    val_acc.append(val_accuracy/100)
    
    logging.info(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
    #logging.info((epoch+1) % 10 == 0)
    if (epoch + 1) % 10 == 0:
        print_confusion_matrix(all_targets, all_predictions)
    if (epoch+1) % 10 == 0:
        print("Save model success!")
        torch.save(resnet, './models/model_'+str(epoch+1)+'.pth')
    
    if (epoch + 1) % 10 == 0:
        # 绘制训练曲线
        plt.plot(range(1, epoch+2), train_losses, label='Train Loss')
        plt.plot(range(1, epoch+2), train_acc, label='Train Accuarcy')
        plt.plot(range(1, epoch+2), val_acc, label='Validation Accuarcy')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.title('Training Curve')
        plt.legend()
        plt.savefig('./train_result/training_curve_'+str(epoch+1)+'.png')
        plt.close()