#アプリ上でアップロードされた画像に対して推論するために必要なネットワークの定義用の bts.py を作成します

# 必要なモジュールのインポート
from torchvision import transforms
import pytorch_lightning as pl
import torch.nn as nn

#学習時に使ったのと同じ学習済みモデルをインポート
from torchvision.models import efficientnet_b0

# 学習済みモデルに合わせた前処理を追加
#テストデータ用の前処理
transform = transforms.Compose([
    transforms.Resize((240, 240)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

#学習データ用の前処理
transform2 = transforms.Compose([
    transforms.Resize((240, 240)),
    transforms.RandomHorizontalFlip(0.35),
    transforms.RandomVerticalFlip(0.35),
    transforms.RandomRotation(degrees=[-180, 180]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

#ネットワークの定義
class Net(pl.LightningModule):

    def __init__(self):
        super().__init__()

        self.feature = efficientnet_b0(pretrained=True)
        self.fc= nn.Linear(1000, 3)
        self.bn = nn.BatchNorm1d(3)

    def forward(self, x):
        h = self.feature(x)
        h = self.fc(h)
        return h


