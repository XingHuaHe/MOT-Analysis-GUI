"""
    二维码 Dataset
"""

from PIL import Image
from typing import List
from torch.tensor import Tensor
from torchvision.transforms import transforms
from torch.utils.data import Dataset, dataset

class QRCodeDataset(Dataset):
    def __init__(self, img_path: str, transforms: transforms = None) -> None:
        super().__init__()

        self.img_path = img_path
        self.transforms = transforms

        self.images = self.get_images(self.img_path)


    def __getitem__(self, index) -> Tensor:
        
        img_path = self.images[index]

        img = Image.open(img_path).convert("RGB")

        if self.transforms is not None:
            img = self.transforms(img)
        else:
            img = transforms.ToTensor()(img)

        return img


    def __len__(self) -> None:
        return len(self.images)


    def get_images(self, img_path: str) -> List:
        """获取指定路径下的所有图片"""

        import os
        
        if not os.path.exists(img_path):
            raise f"{img_path} is not exist"

        results = []
        filenames = os.listdir(img_path)
        for filename in filenames:
            results.append(os.path.join(img_path, filename))

        return results
        

if __name__ == "__main__":

    qrCodeDataset = QRCodeDataset("/home/linsi/VisDrone-Experiences/Datasets/QR-Dataset/train-dataset/images-qrcode")

    from torch.utils.data import DataLoader
    qrCodeDataloader = DataLoader(qrCodeDataset, batch_size=1, shuffle=True)

    for imgs in qrCodeDataloader:
        print(imgs) 