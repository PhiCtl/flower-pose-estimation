from my_classes import GilNet, FlowerPoseDataset, MyTransformation, UnNormalize
from my_functions import identity,\
                         draw,\
                         S_Loss_MSE

from my_constants import IMAGENET_MEAN, IMAGENET_STD
import torch
import numpy as np

import cv2
from torchvision import transforms
import imageio

IMAGENET_norm = transforms.Normalize(mean=IMAGENET_MEAN,std=IMAGENET_STD)
unnorm = UnNormalize()

cuda_available = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda_available else "cpu")
print("Evaluating on: {}".format(device))


#=======================================
# Define models

#TODO: find which model is the correct one !!
model_fnames = [
    "model_20201207_203315",
    "model_20201216_224455",
    "model_20201219_125500_30"
]

# Define cofigurations for each model filename
#TODO correct what is wrong in here
models_config = {   "model_20201207_203315": {"model_name": "resnet34",
                              "num_outputs": 16,
                              "out_func": identity,
                              "citerion": S_Loss_MSE,
                              "normalization":(False, None),
                              "rotation":(False,"")},
                    "model_20201216_224455" :{"model_name": "resnet18",
                              "num_outputs": 16,
                              "out_func": identity,
                              "citerion": S_Loss_MSE,
                              "normalization":(True, IMAGENET_norm),
                              "rotation":(False,"")},
                    "model_20201219_125500_30":{"model_name": "resnet18",
                              "num_outputs": 16,
                              "out_func": identity,
                              "citerion": S_Loss_MSE,
                              "normalization":(True, IMAGENET_norm),
                              "rotation":(False,"")}

}

# Map model filenames to the model objects themselves
models = {}

for fname in model_fnames:
    model_config = models_config[fname]
    model_name = model_config["model_name"]
    num_outputs = model_config["num_outputs"]
    out_func = model_config["out_func"]
    model = GilNet(model=model_name,pretrained=False,num_outputs=num_outputs,out_func=out_func)
    model_path = "./models/"+fname+"/model"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.train(False)
    models[fname] = model

transform = MyTransformation(image_transforms=[transforms.ToTensor()], rotation=False)


#=======================================
# Load image for predictions

filepath = 'data/im_test_2.jpeg'
img = imageio.imread(filepath)
image = torch.tensor(img, dtype=torch.float).transpose_(0,2).transpose_(1,2).unsqueeze_(0)

#=======================================
# Compute prediction

for fname in model_fnames:
    model = models[fname]
    image_ = image.detach().clone()
    (apply_norm, norm) = models_config[fname]["normalization"]
    if apply_norm:
        image_ = torch.unsqueeze(norm(image_.squeeze()), 0)
    else:
        image_.mul_(255)

    output_2d = model(image_)
    print(output_2d)
    output = output_2d.t().reshape(8,2)
    img_bbox = draw(img, output, black=True, ID=True)
    cv2.imshow('result', img_bbox) #TODO : convert to bgr convention
    cv2.waitKey(0)
    cv2.destroyAllWindows()



