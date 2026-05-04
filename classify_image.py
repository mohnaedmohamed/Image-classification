import torch
import torchvision
import torchvision.transforms as transforms
import PIL.Image as Image
torch.serialization.add_safe_globals([torch.optim.SGD])
classes=[
    'Cat',
    'Dog'
]
model=torch.load('best_model.pth',weights_only=False)
mean=[0.4883,0.4553,0.4170]
std=[0.2261,0.2214,0.2217]
image_transforms= transforms.Compose([  
    (transforms.Resize((244,244))),
    transforms.ToTensor(),
    transforms.Normalize(torch.Tensor(mean),torch.Tensor(std))
    ])
def classify(model,image_transforms,image_path,classes):
    model=model.eval()
    image=Image.open(image_path)
    image=image_transforms(image).float()
    image= image.unsqueeze(0)
    outputs=model(image)
    _, predicted =torch.max(outputs.data,1)
    print(classes[predicted.item()])

classify(model,image_transforms,"images for test/660949373_10242999572434644_2399374915586084416_n.jpg",classes)
