from my_classes import GilNet, RealFlowersDataset, MyTransformation, UnNormalize
from my_functions import identity,\
                         draw,\
                         S_Loss_MSE
from my_constants import IMAGENET_MEAN, IMAGENET_STD
import torch
import cv2
from torchvision import transforms
import torch.nn.functional as F

IMAGENET_norm = transforms.Normalize(mean=IMAGENET_MEAN,std=IMAGENET_STD)
unnorm = UnNormalize()

cuda_available = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda_available else "cpu")
print("Evaluating on: {}".format(device))


#=======================================
# Define models

model_fnames = [
    "model_20201212_033500"
]

# Define cofigurations for each model filename
models_config = {   "model_20201212_033500": {"model_name": "resnet18",
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

#========================================
# Load cropped flower picture
# TODO: crop picture

#=======================================
# Load image for predictions
dir = 'data/' #'../synthetic-flower-dataset/outputs/natural/'
#filenames = ['nat_VG09_9.obj_002_0.png',
             # 'nat_VG09_9.obj_002_1.png',
             # 'nat_VG09_9.obj_002_2.png',
             # 'nat_VG09_9.obj_002_3.png',
             # 'nat_VG09_9.obj_002_4.png']
filenames = ['im_test.jpeg']
flower_dataset = RealFlowersDataset(file_names=filenames, root_dir = dir, transform=transform)
prediction_loader = torch.utils.data.DataLoader(flower_dataset)
data_loaders = {"prediction": prediction_loader}
#=======================================
# Compute prediction

for idx,data in enumerate(data_loaders["prediction"]):

    image = data['image']
   #TODO handle padding
    d1 = int((600 - image.shape[-1])/2)
    d2 = int((600 - image.shape[-2])/2)
    image = F.pad(image, (d2,d2,d1,d1), "constant",0)
    print(image.shape)
    image = image.to(device)

    for fname in model_fnames:
        model = models[fname]
        image_ = image.detach().clone()
        (apply_norm, norm) = models_config[fname]["normalization"]
        if apply_norm:
            image_ = torch.unsqueeze(norm(image_.squeeze()), 0)
        else:
            image_.mul_(255)

        output_2d = model(image_)
        img = image_.squeeze().transpose(0,2).transpose(0,1).detach().numpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        print(output_2d, image.shape, img.shape)
        output = output_2d.t().reshape(8,2)

        img_bbox = draw(img, output, black=True, ID=True)
        cv2.imshow('result', img_bbox) #TODO : convert to bgr convention
        cv2.waitKey(0)
        cv2.destroyAllWindows()

#=====================================
# Get camera intrinsics and return orientation
#=====================================

