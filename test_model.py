import flash
from flash.image import SemanticSegmentation, SemanticSegmentationData
from flash import Trainer
from flash.core.data.io.input import DataKeys
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os, glob

# Load the model
model = SemanticSegmentation.load_from_checkpoint(r"D:\machine_learning_AI_Builders\บท4\Segmentation\Segmentic_Segmentation\lightning_flash\semantic_segmentation_model_model_type_unet_bb_resnet34.pt")

trainer = Trainer(
    gpus=torch.cuda.device_count(),  # Check the number of available GPUs
    precision=16 if torch.cuda.device_count() else 32  # Use precision 16 if GPU is available
)

# Sample images
sample_imgs1 = []
sample_imgs2 = glob.glob(r"data/camvid_tiny/images/*.png")[0:3]
sample_imgs = sample_imgs1 + sample_imgs2

datamodule = SemanticSegmentationData.from_files(
    predict_files=sample_imgs,
    batch_size=3
)
codes = np.loadtxt(os.path.join(r"data/camvid_tiny","codes.txt"),dtype=str)
class_list = list(codes)
# Define class names
#class_list = ['Animal', 'Archway', 'Bicyclist', 'Bridge', 'Building', 'Car', 'CartLuggagePram', 'Child', 'Column_Pole', 'Fence', 'LaneMkgsDriv', 'LaneMkgsNonDriv', 'Misc_Text', 'MotorcycleScooter', 'OtherMoving', 'ParkingBlock', 'Pedestrian', 'Road', 'RoadShoulder', 'Sidewalk', 'SignSymbol', 'Sky', 'SUVPickupTruck', 'TrafficCone', 'TrafficLight', 'Train', 'Tree', 'Truck_Bus', 'Tunnel', 'VegetationMisc', 'Void', 'Wall']

# Define colors for each class (random example colors)
colors = np.random.randint(0, 255, size=(len(class_list), 3))

def decode_segmap(image, num_classes, colors):
    """Decode the segmentation map back to RGB colors."""
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    for l in range(num_classes):
        idx = image == l
        r[idx] = colors[l, 0]
        g[idx] = colors[l, 1]
        b[idx] = colors[l, 2]
    rgb = np.stack([r, g, b], axis=2)
    return rgb

# Create the figure and axes
fig, axarr = plt.subplots(ncols=2, nrows=len(sample_imgs), figsize=(15, 5 * len(sample_imgs)))

# Create a color patch for each class
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], marker='o', color='w', label=class_name, 
                         markerfacecolor=colors[i] / 255, markersize=10) for i, class_name in enumerate(class_list)]

# Add a legend for the classes
fig.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(1.05, 0.5), ncol=1, title="Classes")

running_i = 0
for preds in trainer.predict(model, datamodule=datamodule, output="labels"):
    for pred in preds:
        # Convert pred to a numpy array
        pred_np = np.array(pred)
        decoded_pred = decode_segmap(pred_np, len(class_list), colors)
        
        img = plt.imread(sample_imgs[running_i])
        axarr[running_i, 0].imshow(img)
        axarr[running_i, 1].imshow(decoded_pred)

        # Hide axes for better visualization
        axarr[running_i, 1].get_xaxis().set_visible(False)
        axarr[running_i, 1].get_yaxis().set_visible(False)
        axarr[running_i, 0].get_xaxis().set_visible(False)
        axarr[running_i, 0].get_yaxis().set_visible(False)
        
        running_i += 1

# Save the figure with the legend
plt.tight_layout()
plt.savefig("predictions_.png")  # Save all predictions in a single file
plt.show()
