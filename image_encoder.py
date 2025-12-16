import math
import os
import argparse

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import v2

from transformers import AutoModel
from PIL import Image
from tqdm import tqdm


class DinoV3FeatureExtractorBase():
    def __init__(self, features_out):
        super().__init__()
        

class DinoFeatureExtractor(nn.Module):
    def __init__(self, features_out=[x for x in range(1, 12)]):
        super().__init__()
        self.dinov3 = AutoModel.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m")
        for param in self.dinov3.parameters():
            param.requires_grad = False
        self.features_out = features_out

    @torch.no_grad
    def forward(self, X):
        self.dinov3.eval()
        out = self.dinov3(X, output_hidden_states=True)   
        hidden_states = out.hidden_states 
        # Hidden states: [H, [B, CLS + 4 register + n patch tokens, 1024]]: [Tuple, [Tensor]]   
        # n patch tokens = (input_size / 16)^2
        features = []
        for f in self.features_out:
            CLS = hidden_states[f][:, 0, :].mean(1).unsqueeze(0)
            features.append(CLS)
        features = torch.concatenate(features).mean(1)
        return features


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, dateset_dir, input_size=576):
        self.dataset_dir = dateset_dir
        self.image_ids = os.listdir(dateset_dir)
        self.input_size = input_size
        self.transform = v2.Compose([
            v2.ToTensor(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.image_ids)
        
    def __getitem__(self, idx):
        images = []
        image_names = os.listdir(os.path.join(self.dataset_dir, self.image_ids[idx]))
        for image_name in image_names:
            image_dir = os.path.join(self.dataset_dir, self.image_ids[idx], image_name)
            image = Image.open(image_dir).convert("RGB").resize((self.input_size, self.input_size))
            image = self.transform(image)
            images.append(image)
        if len(images) == 0:
            return torch.zeros(1, 3, self.input_size, self.input_size), self.image_ids[idx]
        images = torch.stack(images)
        return images, self.image_ids[idx]



def encode_images(image_dir, device, out_dir="image_features.csv"):
    model = DinoFeatureExtractor().to(device)
    dataset = ImageDataset(image_dir)
    with open(out_dir, "w") as file:
        for i in tqdm(range(len(dataset))):
            images, label = dataset[i]
            images = images.to(device)
            current_features = model(images)
            file.write(f"{label},{','.join(str(x) for x in current_features.cpu().detach().tolist())}\n")      


    
if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--device", type=str, required=True, choices=["cpu", "cuda"])
    args = argparser.parse_args()

    encode_images("images", args.device)