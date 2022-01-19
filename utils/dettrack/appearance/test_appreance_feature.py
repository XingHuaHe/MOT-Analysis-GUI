

import os
import argparse
from matplotlib.pyplot import imshow
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from models.autoencoder import AutoEncoder
from models.QRCodeDataset import QRCodeDataset
from torch.utils.data import DataLoader

def test_encoder(args: argparse.ArgumentParser):
    """
        测试编码器
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size,)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    if args.model_name == "autoencoder":
        model = AutoEncoder(trainable=args.trainable, img_size=args.image_size, batch_size=args.batch_size)
    model.to(device)

    try:
        state_dict = torch.load(args.weights, map_location=device)
        model.load_state_dict(state_dict=state_dict)
    except Exception as e:
        print(e.__traceback__())

    filenames = os.listdir(args.images_path)

    count = 0
    plt.figure(figsize=(20, 4))
    n = 10
    for filename in filenames:
        img_path = os.path.join(args.images_path, filename)

        img0 = Image.open(img_path).convert("RGB")

        img = test_transform(img0)

        feature = model(torch.unsqueeze(img.to(device), 0))

        print(feature)

        break

    
def test_autoencoder(args: argparse.ArgumentParser):
    """
        测试自编码器
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size,)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    if args.model_name == "autoencoder":
        model = AutoEncoder(trainable=args.trainable, img_size=args.image_size, batch_size=args.batch_size)
    model.to(device)

    try:
        state_dict = torch.load(args.weights, map_location=device)
        model.load_state_dict(state_dict=state_dict)
    except Exception as e:
        print(e.__traceback__())

    filenames = os.listdir(args.images_path)
    
    count = 0
    plt.figure(figsize=(20, 4))
    n = 10
    for filename in filenames:
        img_path = os.path.join(args.images_path, filename)

        img0 = Image.open(img_path).convert("RGB")

        img = test_transform(img0)

        output = model(torch.unsqueeze(img.to(device), 0))
        output = torch.squeeze(output).data.cpu().permute(1,2,0)

        min_val = output.min()
        max_val = output.max()

        output = ((output - min_val) / (max_val - min_val) * 255).numpy()
        
        output = np.uint8(output)

        img_re = Image.fromarray(output)

        
        ax = plt.subplot(2,n,count+1)
        plt.imshow(img0)
        plt.title("original")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)


        
        ax = plt.subplot(2,n,count+1+n)
        plt.imshow(img_re)
        plt.title("reconstructed")
        plt.gray()
        
        count += 1

        if count >= 10:
            break
        
    # plt.show()

    os.makedirs(args.outputs, exist_ok=True)
    plt.savefig(os.path.join(args.outputs, "vision.jpg"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="what you want to do")
    parser.add_argument('--weights', type=str, default="./weights/autoencoder/50_ck.pt", help="pre-training model")
    parser.add_argument('--images-path', type=str, default="/home/linsi/VisDrone-Experiences/Datasets/QR-Dataset/train-dataset/images-qrcode", help="images directory")
    parser.add_argument('--batch-size', type=int, default=1, help="Traing batch size")
    parser.add_argument('--model-name', type=str, default="autoencoder", help="selectived model to train")
    parser.add_argument('--image-size', type=int, default=56, help="model input images size")
    parser.add_argument('--save', type=bool, default=True, help="wether to save checkpoint")
    parser.add_argument('--trainable', type=bool, default=False, help="True for traing or False for evel")
    parser.add_argument('--outputs', type=str, default="/home/linsi/VisDrone-Experiences/Appreance-Feature/outputs/test", help="output directory")
    args = parser.parse_args()

    os.makedirs(args.outputs, exist_ok=True)

    if args.trainable:
        test_autoencoder(args)
    else:
        test_encoder(args)