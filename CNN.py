import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torchvision.models import resnet18 , ResNet18_Weights 
torch.serialization.add_safe_globals([torch.optim.SGD])
#print(torch.cuda.is_available())
#print(torch.cuda.device_count())
#print(torch.version.cuda)
device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print("using device:", device)
traning_dataset_path="Data/train dataset"
mean=[0.4883,0.4553,0.4170]
std=[0.2261,0.2214,0.2217]
traning_transforms=transforms.Compose([
    (transforms.Resize((244,244))),
    transforms.ToTensor(),
    transforms.Normalize(torch.Tensor(mean),torch.Tensor(std)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    ])

train_dataset=torchvision.datasets.ImageFolder(root=traning_dataset_path,transform=traning_transforms)
train_loader=torch.utils.data.DataLoader(dataset=train_dataset,batch_size=32, shuffle=True)

test_dataset_path="Data/test_dataset"
test_dataset_transforms=transforms.Compose([
    transforms.Resize((244,244)),
    transforms.ToTensor(),
    transforms.Normalize(torch.Tensor(mean),torch.Tensor(std)), 
 ])
test_dataset=torchvision.datasets.ImageFolder(root=test_dataset_path,transform=test_dataset_transforms)
test_loader=torch.utils.data.DataLoader(test_dataset,batch_size=32,shuffle=False)
def get_mean_and_std(loader):
    mean=0.
    std=0.
    total_images_count=0
    for images,_ in loader:
        images_count_in_a_batch=images.size(0)
        images=images.view(images_count_in_a_batch,images.size(1),-1)
        mean+=images.mean(2).sum(0)
        std+=images.std(2).sum(0)
        total_images_count+=images_count_in_a_batch
    mean /= total_images_count
    std /= total_images_count

    return mean,std
#just for show what transforms happend to images and the augmentation if i used it 
def show_transformed_images(dataset):

    loader=torch.utils.data.DataLoader(dataset,batch_size=6,shuffle=True)
    batch=next(iter(loader))
    images,lables=batch
    grid=torchvision.utils.make_grid(images,nrow=3)
    plt.figure(figsize=(11,11))
    plt.imshow(np.transpose(grid,(1,2,0)))
    plt.show()
    print("lables",lables)
#print("train dataset\n")
#show_transformed_images(train_dataset)
#print("test dataset")
#show_transformed_images(test_dataset)
#this fun   turn on CUDA for using my GPU for traning this model 
def set_device():
     if torch.cuda.is_available():
         dev="cuda:0"
     else:
         dev="cpu"
     return torch.device(dev)


#Here in this function that have all what i need to itrate and learn my model and i used a buildin model called recnet18 
def train_nn(model, train_loader,test_loader,critorion,optimizer,n_epochs):
    best_acc=0
    device=set_device()
    model=model.to(device)
    for epoch in range(n_epochs):
        print("Epoch number % d"%(epoch+1))
        model.train()
        running_loss=0.0
        running_correct=0.0
        total=0
        for data in train_loader:
            images,labels=data
            images=images.to(device)
            labels=labels.to(device)
            total+=labels.size(0)

            optimizer.zero_grad()
            outputs=model(images)
            _, predicted =torch.max(outputs.data,1)
            loss = critorion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()
            running_correct+= (labels==predicted).sum().item()
        epoch_loss =running_loss/len(train_loader)
        epoch_acc=100.00* running_correct/total
        print("     -Training dataset. Got %d out of %d images correctly (%.3f%%).Epoch loss: %.3f"%(running_correct,total,epoch_acc,epoch_loss))
        test_dataset_acc=evaluate_model_on_test_set(model,test_loader)
        if (test_dataset_acc>best_acc):
            best_acc=test_dataset_acc
            save_checkpoint(model,epoch,optimizer,best_acc)
    print("finished")
    return model       



#this fun will have all methods for calculate and evaluate the model  that will run in traning method     
def evaluate_model_on_test_set(model,test_loader):
    model.eval()
    predicted_correctly_on_epoch=0
    device=set_device()
    model=model.to(device)
    total=0
    with torch.no_grad():
        for data in test_loader:
            images,labels=data
            images=images.to(device)
            labels=labels.to(device)
            total+=labels.size(0)
            outputs=model(images)
            _, predicted =torch.max(outputs.data,1)
            predicted_correctly_on_epoch+= (predicted==labels).sum().item()

    epoch_acc=100.0*predicted_correctly_on_epoch/total
    print("     -Testing dataset. Got %d out of %d images correctly (%.3f%%)"%(predicted_correctly_on_epoch ,total,epoch_acc))
    return epoch_acc
    #lr=0.1 to .001


def save_checkpoint(model,epoch,optimizer,best_acc):
    state={
        "epoch":epoch +1,
        "model":model.state_dict(),
        "best accuracy":best_acc,
        "optimizer":optimizer,
    }
    torch.save(state,"model_best_checkpoint.pth.tar")


checkpoint=torch.load("model_best_checkpoint.pth.tar",weights_only=False)
#print(checkpoint["epoch"])
#print(checkpoint["best accuracy"])

recnet18_model=models.resnet18() 
num_ftrs=recnet18_model.fc.in_features
number_of_classes=2
recnet18_model.fc=nn.Linear(num_ftrs,number_of_classes)
recnet18_model.load_state_dict(checkpoint['model'])
torch.save(recnet18_model,'best_model.pth')
device=set_device()
recent_18_model=recnet18_model.to(device)
loss_fn= nn.CrossEntropyLoss()
optimizer=optim.SGD(recnet18_model.parameters(),lr=0.001,momentum=0.9 ,weight_decay=1e-4)
#train_nn(recent18_model,train_loader,test_loader,loss_fn,optimizer,5)