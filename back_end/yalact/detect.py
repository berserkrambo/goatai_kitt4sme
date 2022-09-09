
import pickle
import torch

from back_end.yalact.modules.yolact import Yolact

from back_end.yalact.utils.output_utils import nms, after_nms, draw_img
from back_end.yalact.config import get_config
from back_end.yalact.utils.augmentations import val_aug
from path import Path

import numpy as np


class Yalact:

    def __init__(self, device):
        root_path = Path("back_end/yalact")

        # with open(root_path / "args_best_28.8_.pkl", "rb") as argsfile:
        with open(root_path / "args_best_28.8_.pkl", "rb") as argsfile:
            args = pickle.load(argsfile)
        self.cfg = get_config(args, mode='detect')

        self.device = device
        self.model = Yolact(self.cfg)

        self.model.load_state_dict(torch.load(root_path/self.cfg.weight), strict=False)
        self.model.to(device)
        self.model.eval()


    def predict(self, im0):
        with torch.no_grad():

            frame_origin = im0
            img_h, img_w = frame_origin.shape[0:2]
            frame_trans = val_aug(frame_origin, self.cfg.img_size)
            frame_tensor = torch.tensor(frame_trans).float()
            frame_tensor = frame_tensor.to(self.device)
            class_p, box_p, coef_p, proto_p = self.model(frame_tensor.unsqueeze(0))
            ids_p, class_p, box_p, coef_p, proto_p = nms(class_p, box_p, coef_p, proto_p, self.model.anchors, self.cfg)
            ids_p, class_p, boxes_p, masks_p = after_nms(ids_p, class_p, box_p, coef_p, proto_p, img_h, img_w, self.cfg)

            if ids_p is not None:
                ids_p = ids_p.cpu().numpy()
                class_p = class_p.cpu().numpy()
                boxes_p = boxes_p.cpu().numpy()
                masks_p = masks_p.cpu().numpy()

                return ids_p, boxes_p, class_p, masks_p
                # frame_numpy = draw_img(ids_p, class_p, boxes_p, masks_p, frame_origin, self.cfg)

            else:
                return None, np.empty(shape=(0, 4)), None, None

    def drawo_pred(self, ids_p, class_p, boxes_p, masks_p, img, draw_boxes=True, draw_mask=True):
        return draw_img(ids_p, class_p, boxes_p, masks_p, img, draw_boxes=draw_boxes, draw_mask=draw_mask)













