#Uncomment and Run each section to preprocess the data

#--------------------------------------------------------------------
'''
import pyarrow.parquet as pq
from PIL import Image
import numpy as np
import io

# Load the Parquet file and covert to pandas format
parquet_file = "1.parquet"
table = pq.read_table(parquet_file)
df = table.to_pandas()

# Iterate through the dataset 
# Extract the tif images and corresponding labels/masks
for index, row in df.iterrows():

    # Extract image data and label
    image_bytes = row["tif"]
    image_label_bytes = row["label_tif"]

    # Decode image data and label from bytes
    image_label= Image.open(io.BytesIO(image_label_bytes))

    #Delete sidewalks with a mask of max_pixel_value 0
    image_label_pixel= np.array(image_label)

    # Find the maximum pixel value
    max_pixel_value = np.max(image_label_pixel)
    if max_pixel_value == 0:
        continue
        
    image = Image.open(io.BytesIO(image_bytes))
        
    # Save data to directory tiffile
    image_path = f"image_{i}.tif"
    image.save(f'./tiffile/{image_path}')

    # Save image label to directory tiflabel
    image_label_path= f"image_label{i}.tif"
    image_label.save(f'./tiflabel/{image_label_path}')
'''
#-------------------------------------------------------------------

#-------------------------------------------------------------------
'''
from PIL import Image
import numpy as np 
import matplotlib.pyplot as plt

#read sample image and mask and ouput results
index=30
image= Image.open(f'./tiffile/image_{index}.tif')
image_label= Image.open(f'./tiflabel/image_label{index}.tif')

image= np.array(image)
image_label= np.array(image_label)
       
fig, axes = plt.subplots(1,2)
axes[0].imshow(image)
axes[1].imshow(image_label)

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

plt.show()
'''
#-------------------------------------------------------------------

#-------------------------------------------------------------------
'''
from PIL import Image
import numpy as np
import os

#save the images as a large nparray for processing
images_dir= "./tiffile"
image_data= []

#traverse through the images and save as a nparray
for filename in os.listdir(images_dir):
    image_path= os.path.join(images_dir, filename)
    image= Image.open(image_path)

    #Convert into NumpyArray
    image_array= np.array(image)

    image_data.append(image_array)

large_array_image= np.stack(image_data)
print("Shape of large array:", large_array_image.shape)
np.save("large_array_image.npy", large_array_image)


images_label_dir= "./tiflabel"
image_label_data= []

#traverse through the masks and save as a nparry
for filename in os.listdir(images_label_dir):
    image_label_path= os.path.join(images_label_dir, filename)
    image_label= Image.open(image_label_path)

    #Convert into NumpyArray
    image_label_array= np.array(image_label)

    image_label_data.append(image_label_array)

large_array_label= np.stack(image_label_data)
print("Shape of large array:", large_array_label.shape)
np.save("large_array_label.npy", large_array_label)
'''
#-------------------------------------------------------------------

#-------------------------------------------------------------------
'''
from datasets import Dataset
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random

#load the numpy arrays
large_array_image= np.load("large_array_image.npy")
large_array_label= np.load("large_array_label.npy")

# Convert the NumPy arrays to Pillow images and store them in a dictionary
dataset_dict = {
    "image": [Image.fromarray(img) for img in large_array_image],
    "label": [Image.fromarray(img) for img in large_array_label]
}

# Create the dataset using the datasets.Dataset class
dataset = Dataset.from_dict(dataset_dict)
#Save the dataset
dataset.save_to_disk("./dataset")
'''
#--------------------------------------------------------------------

#--------------------------------------------------------------------
'''
from SAMclass import SAMDataset
from datasets import load_from_disk
from tqdm import tqdm
from statistics import mean
import torch
from torch.nn.functional import threshold, normalize
from transformers import SamProcessor
from torch.optim import Adam as torch_Adam
import monai

#load the dataset
dataset= load_from_disk("./dataset")
processor= SamProcessor.from_pretrained("facebook/sam-vit-base")

#Load the train dataset for training
train_dataset= SAMDataset(dataset=dataset, processor=processor)

#Create a DataLoader instance for the training dataset
from torch.utils.data import DataLoader
train_dataloader= DataLoader(train_dataset, batch_size=2, shuffle=True, drop_last=False)

batch = next(iter(train_dataloader))

#Load the model
from transformers import SamModel
model= SamModel.from_pretrained("facebook/sam-vit-base")

for name, param in model.named_parameters():
    if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
        param.requires_grad_(False)


#Initialize the optimizer and the loss function
optimizer= torch_Adam(model.mask_decoder.parameters(), lr=1e-5, weight_decay=0)
#Try DiceFocalLoss, FocalLoss, DiceCELoss
seg_loss= monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

#Train the data set
num_epochs = 10

#Set device to use GPU if avaliable
device = "cuda" if torch.cuda.is_available() else "cpu"

model.to(device)

torch.cuda.empty_cache()

model.train()
for epoch in range(num_epochs):
    epoch_losses = []
    for batch in tqdm(train_dataloader):
      # forward pass
      outputs = model(pixel_values=batch["pixel_values"].to(device),
                      input_boxes=batch["input_boxes"].to(device),
                      multimask_output=False)

      # compute loss
      predicted_masks = outputs.pred_masks.squeeze(1)
      ground_truth_masks = batch["ground_truth_mask"].float().to(device)
      loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))

      # backward pass (compute gradients of parameters w.r.t. loss)
      optimizer.zero_grad()
      loss.backward()

      # optimize
      optimizer.step()
      epoch_losses.append(loss.item())

      torch.cuda.empty_cache()
    #save batch checkpoint
    torch.save(model.state_dict(), f"./model/mito_model_checkpoint{epoch}.pth")
    
    print(f'EPOCH: {epoch}')
    print(f'Mean loss: {mean(epoch_losses)}')
'''
