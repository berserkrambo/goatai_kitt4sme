import numpy as np
import cv2
import torch

from back_end.pose_resnet.detect import PoseNet

from back_end.tracker.sort import Sort
from back_end.anonymizator.anonymize import anonymize
from back_end.yalact.detect import Yalact
from back_end.fall_detector.detect import FallDetector

class AI4SDW:
    def __init__(self):

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print(self.device)

        self.box_model = Yalact(device=self.device)
        self.pose_model = PoseNet(device=self.device)

        self.tracker = Sort(max_age=60, min_hits=3)

        self.fall_det = FallDetector()

    def process_next_frame(self, image_bgr):

        img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        ids_p, pred_boxes, class_p, masks_p = self.box_model.predict(image_bgr.copy())
        pose_preds = self.pose_model.predict(img, pred_boxes)
        pose_class = self.fall_det.predict(pose_preds)

        # set shape for tracker
        if len(pred_boxes) >= 1:
            out = []
            for i in range(len(pred_boxes)):
                o_i = []
                o_i.extend(pred_boxes[i].astype(np.float32))  # bbox
                o_i.extend([class_p[i]])  # conf
                o_i.extend(np.expand_dims(masks_p[i], 0))  # mask
                o_i.extend([pose_preds[i].astype(np.int32)])  # pose
                o_i.extend([pose_class[i]])  # pose class
                o_i = np.asarray(o_i, dtype=object)
                out.append(o_i)
            pred_boxes = np.asarray(out, dtype=object)

        # update tracker
        outputs = self.tracker.update(np.copy(pred_boxes))

        # anonymization module
        if len(outputs) > 0:
            image_bgr = anonymize(image_bgr, outputs[:,0].copy(), outputs[:,2].copy(), outputs[:,3].copy())

        return image_bgr, outputs