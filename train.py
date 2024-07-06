import flash
import torch
from flash.core.data.utils import download_data
from flash.image import SemanticSegmentation, SemanticSegmentationData
from pytorch_lightning.loggers import CSVLogger
import os


from icevision.all import *
import numpy as np
codes = np.loadtxt(os.path.join(r"data/camvid_tiny","codes.txt"),dtype=str)
class_map = ClassMap(list(codes))

# 1. Create the DataModule
# The data was generated with the  CARLA self-driving simulator as part of the Kaggle Lyft Udacity Challenge.
# More info here: https://www.kaggle.com/kumaresanmanickavelu/lyft-udacity-challenge
# download_data(
#     "https://github.com/ongchinkiat/LyftPerceptionChallenge/releases/download/v0.1/carla-capture-20180513A.zip",
#     "./data",
# )
print(SemanticSegmentation.available_backbones('unet'))

img_path = os.path.join(r"data/camvid_tiny/images")
label_path = os.path.join(r"data/camvid_tiny/labels")

datamodule = SemanticSegmentationData.from_folders(
    train_folder=img_path,
    train_target_folder=label_path,
    val_split=0.1,
    transform_kwargs={"image_size": (256, 256)},
    num_classes=class_map.num_classes,
    batch_size=4,
)

# 2. Build the task
model = SemanticSegmentation(
    backbone="resnet34",
    head="unet",
    num_classes=datamodule.num_classes,
)
print("Processing training")
trainer = flash.Trainer(
    max_epochs=50,
    logger=CSVLogger(save_dir='logs/'),
    gpus=torch.cuda.device_count(),
    precision=16 if torch.cuda.device_count() else 32,
#     limit_train_batches=0.1,
#     limit_val_batches=0.1,
)
trainer.finetune(model, datamodule=datamodule, strategy="freeze")
trainer.save_checkpoint("semantic_segmentation_model_model_type_unet_bb_resnet34.pt")