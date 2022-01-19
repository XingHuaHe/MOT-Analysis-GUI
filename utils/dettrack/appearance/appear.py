"""
    外观特征提取模型类

"""
import torch
import torchvision.transforms as transforms

from utils.dettrack.appearance.models.autoencoder import AutoEncoder


# from .models.autoencoder import AutoEncoder


class Appearance(object):
    """
        Deep SORT 外观特征提取
    """

    def __init__(self, fimg_size: int,
                 model_name: str,
                 ds_weights: str) -> None:
        super().__init__()

        self.fimg_size = fimg_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

        self.transform = transforms.Compose([
            transforms.Resize((self.fimg_size, self.fimg_size,)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        if model_name == "autoencoder" or model_name == "SKCAutoencoder":
            self.model = AutoEncoder(img_size=self.fimg_size, batch_size=1)
        self.model.to(self.device).eval()

        try:
            state_dict = torch.load(ds_weights, map_location=self.device)
            self.model.load_state_dict(state_dict=state_dict)
        except Exception as e:
            print(e)

    def extract_appearance_feature(self, img):
        """
            获取输入图像的外观特征
            img: 3*56*56
        """
        img = self.transform(img)

        feature = self.model(torch.unsqueeze(img.to(self.device), 0))

        feature = torch.squeeze(feature).data.cpu().numpy()

        return feature
