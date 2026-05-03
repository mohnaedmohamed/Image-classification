import torch
import os

#def count_images(path):
 #   total = 0
  #  for cls in os.listdir(path):
   #     total += len(os.listdir(os.path.join(path, cls)))
   # return total

#print("Train:", count_images("Data/train dataset"))
#print("Val:", count_images("Data/validation_dataset"))
#print("Test:", count_images("Data/test_dataset"))


from PIL import Image
import os

def remove_bad_images(folder):
    for root, dirs, files in os.walk(folder):
        for file in files:
            path = os.path.join(root, file)
            try:
                img = Image.open(path)
                img.verify()  # check if corrupted
            except:
                print("Removing:", path)
                os.remove(path)

remove_bad_images("Data")

checkpoit=torch.load("model_best_checkpoint.pth.tar",weights_only=False)
print(checkpoit["epoch"])
print(checkpoit["best accuracy"])
