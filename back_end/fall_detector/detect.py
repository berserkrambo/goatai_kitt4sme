import torch
import numpy as np
from torch import nn
from fipy.ngsi.entity import BoolAttr
from path import Path

class LinearModel(nn.Module):

    def __init__(self, nclasses=5):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(34, 32), nn.LeakyReLU(),
            nn.Linear(32, 16), nn.LeakyReLU(), nn.Dropout(),
            nn.Linear(16, nclasses)
        )

    def forward(self, x):
        return self.net(x)


class FallDetector:
    """
    Wrapper class for LineaModel pose classificator
    """

    def __init__(self, pose_model_path="ai4sdw/fall_detector"):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # pose_model_path = Path(pose_model_path)
        self.fall_model = LinearModel()
        self.fall_model.to(self.device)
        self.fall_model.load_state_dict(torch.load(Path(__file__).abspath().parent/"best.pth", map_location=(self.device))[0])
        self.fall_model.eval()

        self.label_dict = {
            0: "Standing",
            1: "Sitting",
            2: "Lying",
            3: "Bending",
            4: "Crawling"
        }

    def predict(self, poses):
        """
        :param poses: Nx17x2 numpy array pose
        :return: predicted label
        """
        with torch.no_grad():

            batch = []
            pred_label = []

            if len(poses) > 0:
                for pose in poses:
                    x_min, y_min, x_max, y_max = min(pose[:, 0]), min(pose[:, 1]), max(pose[:, 0]), max(pose[:, 1])
                    center = np.asarray([(x_min + x_max) / 2, (y_min + y_max) / 2], dtype=np.float32)

                    pose_copy = np.copy(pose)
                    pose_copy -= center

                    r = min(1 / (x_max - x_min), 1 / (y_max - y_min))
                    pose_copy *= r
                    pose_copy += 0.5

                    batch.append(pose_copy.flatten('F'))

                    x = torch.from_numpy(np.asarray(batch, dtype=np.float32))
                    x = x.to(self.device)
                    y = self.fall_model(x)

                pred_label = np.argmax(y.cpu().numpy(), axis=1)
                pred_label = [self.label_dict[pred] for pred in pred_label]
        return pred_label