from path import Path
import torch
import pickle
import numpy as np
import cv2
import torchvision.transforms as transforms
from back_end.pose_resnet.lib.core.inference import get_final_preds
from back_end.pose_resnet.lib.utils.transforms import get_affine_transform
from back_end.pose_resnet.model.res18_192x128_d64x3.pose_resnet import get_pose_net
import math

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


PAIRS = [
    [[0, 1], [1, 3]],   # nose - left_eye / left_eye - left_ear                         71
    [[4, 2], [2, 0]],   # right_ear - right_eye / right_eye - nose                      71
    [[6, 0], [0, 5]],   # right_shoulder -  nose /  nose - left_shoulder                104
    [[0, 5], [6, 5]],   # nose - left_shoulder / right_shoulder - left_shoulder         56
    [[6, 5], [6, 0]],   # right_shoulder - nose / right_shoulder - left_shoulder        48
    [[10, 8], [8, 6]],   # right_wrist - right_elbow / right_elbow - right_shoulder     4
    [[5, 7], [7, 9]],   # l_sho - l_elb / l_elb - l_wrist                               4
    [[14, 12], [12, 11]],   # r_kn - r_hip / r_hip - l_hip                              86
    [[12, 11], [11, 13]],   # r_hip - l_hip / l_hip - L_kn                              82
    [[16, 14], [14, 12]],   # r_ank - r_kn / r_kn - r_hip                               7
    [[11, 13], [13, 15]],   # l_hip - l_kn / l_kn - l_ank                               0
    [[12, 0], [0, 11]],   # r_hip -  nose /  nose - l_hip                               161
]

COCO_KEYPOINT_INDEXES = {
    0: 'nose',
    1: 'left_eye',
    2: 'right_eye',
    3: 'left_ear',
    4: 'right_ear',
    5: 'left_shoulder',
    6: 'right_shoulder',
    7: 'left_elbow',
    8: 'right_elbow',
    9: 'left_wrist',
    10: 'right_wrist',
    11: 'left_hip',
    12: 'right_hip',
    13: 'left_knee',
    14: 'right_knee',
    15: 'left_ankle',
    16: 'right_ankle'
}

SKELETON = [
    [1, 3], [1, 0], [2, 4], [2, 0], [0, 5], [0, 6], [5, 7], [7, 9], [6, 8], [8, 10], [5, 11], [6, 12], [11, 12],
    [11, 13], [13, 15], [12, 14], [14, 16]
]

CocoColors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

NUM_KPTS = 17


class PoseNet:
    def __init__(self, device, pose_model_path="back_end/pose_resnet"):
        super(PoseNet, self).__init__()
        self.model_path = Path(pose_model_path)

        with open(self.model_path / "model/pose_model_config.pkl", "rb") as f:
            self.cfg = pickle.load(f)
        self.cfg.defrost()
        self.cfg.TEST.MODEL_FILE = str((self.model_path / "model/inference-config.yaml"))
        self.cfg.freeze()

        self.device = device

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        self.pose_model = get_pose_net(self.cfg, is_train=False)
        self.pose_model.to(self.device)

        self.pose_model.load_state_dict(torch.load(self.model_path / "model/res18_192x128_d64x3/model_best.pth"),
                                        strict=False)
        self.pose_model.eval()

    def predict(self, img, pred_boxes):
        with torch.no_grad():
            image_pose = []
            centers = []
            scales = []
            pose_preds = []

            # pose estimation
            if len(pred_boxes) >= 1:
                for x1,y1,x2,y2 in pred_boxes:
                    box = [(x1,y1), (x2,y2)]
                    center, scale = self.box_to_center_scale(box, self.cfg.MODEL.IMAGE_SIZE[0],
                                                             self.cfg.MODEL.IMAGE_SIZE[1])
                    centers.append(center)
                    scales.append(scale)
                    image_pose.append(img)

                pose_preds = self.get_pose_estimation_prediction(image_pose, centers, scales)

        return pose_preds

    def draw_pose(self, keypoints, img):
        """draw the keypoints and the skeletons.
        :params keypoints: the shape should be equal to [17,2]
        :params img:
        """
        assert keypoints.shape == (NUM_KPTS, 2)
        for i in range(len(SKELETON)):
            kpt_a, kpt_b = SKELETON[i][0], SKELETON[i][1]
            x_a, y_a = keypoints[kpt_a][0], keypoints[kpt_a][1]
            x_b, y_b = keypoints[kpt_b][0], keypoints[kpt_b][1]
            cv2.circle(img, (int(x_a), int(y_a)), 6, CocoColors[i], -1)
            cv2.circle(img, (int(x_b), int(y_b)), 6, CocoColors[i], -1)
            cv2.line(img, (int(x_a), int(y_a)), (int(x_b), int(y_b)), CocoColors[i], 2)

    def draw_plines(self, keypoints, img):
        assert keypoints.shape == (NUM_KPTS, 2)

        for pj, pair in enumerate(PAIRS):
            edge1 = pair[0]
            edge2 = pair[1]
            edge1 = set(edge1)
            edge2 = set(edge2)
            mid_point = edge1.intersection(edge2).pop()
            a = (edge1 - edge2).pop()
            b = (edge2 - edge1).pop()

            v1 = keypoints[mid_point] - keypoints[a]
            v2 = keypoints[mid_point] - keypoints[b]

            angle = (math.degrees(np.arccos(np.dot(v1, v2)
                                            / (np.linalg.norm(v1) * np.linalg.norm(v2)))))

            cv2.circle(img, (int(keypoints[a][0]), int(keypoints[a][1])), 6, CocoColors[pj], -1)
            cv2.circle(img, (int(keypoints[mid_point][0]), int(keypoints[mid_point][1])), 6, CocoColors[pj], -1)
            cv2.circle(img, (int(keypoints[b][0]), int(keypoints[b][1])), 6, CocoColors[pj], -1)

            cv2.line(img, (int(keypoints[a][0]), int(keypoints[a][1])), (int(keypoints[mid_point][0]), int(keypoints[mid_point][1])), CocoColors[pj], 2)
            cv2.line(img, (int(keypoints[mid_point][0]), int(keypoints[mid_point][1])), (int(keypoints[b][0]), int(keypoints[b][1])), CocoColors[pj], 2)

            print(angle)
            # cv2.imshow("", img)
            # cv2.waitKey()


    # def get_person_detection_boxes(self, model, img, threshold=0.5):
    #     pred = model(img)
    #     pred_classes = [COCO_INSTANCE_CATEGORY_NAMES[i]
    #                     for i in list(pred[0]['labels'].cpu().numpy())]  # Get the Prediction Score
    #     pred_boxes = [[(i[0], i[1]), (i[2], i[3])]
    #                   for i in list(pred[0]['boxes'].detach().cpu().numpy())]  # Bounding boxes
    #     pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    #     if not pred_score or max(pred_score) < threshold:
    #         return []
    #     # Get list of index with score greater than threshold
    #     pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
    #     pred_boxes = pred_boxes[:pred_t + 1]
    #     pred_classes = pred_classes[:pred_t + 1]
    #
    #     person_boxes = []
    #     for idx, box in enumerate(pred_boxes):
    #         if pred_classes[idx] == 'person':
    #             person_boxes.append(box)
    #
    #     return person_boxes

    def get_pose_estimation_prediction(self, images, centers, scales):
        model_inputs = []

        rotation = 0

        for i in range(len(images)):
            # pose estimation transformation
            trans = get_affine_transform(centers[i], scales[i], rotation, self.cfg.MODEL.IMAGE_SIZE)
            mi = cv2.warpAffine(
                images[i],
                trans,
                (int(self.cfg.MODEL.IMAGE_SIZE[0]), int(self.cfg.MODEL.IMAGE_SIZE[1])),
                flags=cv2.INTER_LINEAR)

            model_inputs.append(mi)

            # pose estimation inference
        model_inputs = [self.transform(mi).unsqueeze(0) for mi in model_inputs]
        # switch to evaluate mode

        # compute output heatmap

        output = self.pose_model(torch.concat(model_inputs).to(self.device))

        preds, _ = get_final_preds(
            self.cfg,
            output.clone().cpu().numpy(),
            np.asarray(centers),
            np.asarray(scales))

        return preds

    def box_to_center_scale(self, box, model_image_width, model_image_height):
        """convert a box to center,scale information required for pose transformation
        Parameters
        ----------
        box : list of tuple
            list of length 2 with two tuples of floats representing
            bottom left and top right corner of a box
        model_image_width : int
        model_image_height : int

        Returns
        -------
        (numpy array, numpy array)
            Two numpy arrays, coordinates for the center of the box and the scale of the box
        """
        center = np.zeros((2), dtype=np.float32)

        bottom_left_corner = box[0]
        top_right_corner = box[1]
        box_width = top_right_corner[0] - bottom_left_corner[0]
        box_height = top_right_corner[1] - bottom_left_corner[1]
        bottom_left_x = bottom_left_corner[0]
        bottom_left_y = bottom_left_corner[1]
        center[0] = bottom_left_x + box_width * 0.5
        center[1] = bottom_left_y + box_height * 0.5

        aspect_ratio = model_image_width * 1.0 / model_image_height
        pixel_std = 200

        if box_width > aspect_ratio * box_height:
            box_height = box_width * 1.0 / aspect_ratio
        elif box_width < aspect_ratio * box_height:
            box_width = box_height * aspect_ratio
        scale = np.array(
            [box_width * 1.0 / pixel_std, box_height * 1.0 / pixel_std],
            dtype=np.float32)
        if center[0] != -1:
            scale = scale * 1.25

        return center, scale