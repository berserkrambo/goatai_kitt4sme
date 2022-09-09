from path import Path
from hbu_services.fall_detector.model.model import LinearModel
import torch
import numpy as np

class FallDetector:
    """
    Wrapper class for LineaModel pose classificator
    """
    def __init__(self, device, pose_model_path="hbu_services/fall_detector"):
        self.model_path = Path(pose_model_path)
        self.device = device

        self.fall_model = LinearModel()
        self.fall_model.to(self.device)
        self.fall_model.load_state_dict(torch.load(self.model_path / "model" / "best.pth")[0])
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
        batch = []
        for pose in poses:
            x_min, y_min, x_max, y_max = min(pose[:, 0]), min(pose[:, 1]), max(pose[:, 0]), max(pose[:, 1])
            center = np.asarray([(x_min + x_max) / 2, (y_min + y_max) / 2], dtype=np.float32)

            pose_copy = np.copy(pose)
            pose_copy -= center

            r = min(1 / (x_max - x_min), 1 / (y_max - y_min))
            pose_copy *= r
            pose_copy += 0.5

            batch.append(pose_copy.flatten('F'))

        with torch.no_grad():
            x = torch.from_numpy(np.asarray(batch, dtype=np.float32))
            x = x.to(self.device)
            y = self.fall_model(x)

        pred_label = np.argmax(y.cpu().numpy(), axis=1)

        return pred_label



