import numpy as np
import cv2
import torch

from back_end.pose_resnet.detect import PoseNet

from hbu_services.tracker.sort import Sort
from hbu_services.anonymizator.anonymize import anonymize
from back_end.yalact.detect import Yalact


class AI4SDW:
    def __init__(self):

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print(self.device)

        # w_path = "back_end/yolox/yolox_m.pth"
        # box_model = YoloX(device=device, weightsPath=w_path, num_classes=2, get_top2=False)
        self.box_model = Yalact(device=self.device)
        self.pose_model = PoseNet(device=self.device)

        self.tracker = Sort(max_age=60, min_hits=3)

    def process_next_frame(self, image_bgr):

        img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        ids_p, pred_boxes, class_p, masks_p = self.box_model.predict(image_bgr.copy())
        pose_preds = self.pose_model.predict(img, pred_boxes)
        # pose_class = self.fall_det.predict(pose_preds)

        # set shape for tracker
        if len(pred_boxes) >= 1:
            # image_bgr = box_model.drawo_pred(ids_p, class_p, pred_boxes, masks_p, image_bgr) # todo comment this one
            out = []
            for i in range(len(pred_boxes)):
                o_i = []
                o_i.extend(pred_boxes[i].astype(np.float32))  # bbox
                o_i.extend([class_p[i]])  # conf
                o_i.extend(np.expand_dims(masks_p[i], 0))  # mask
                o_i.extend([pose_preds[i].astype(np.int32)])  # pose
                # o_i.extend([pose_class[i]])  # pose class
                o_i = np.asarray(o_i, dtype=object)
                out.append(o_i)
            pred_boxes = np.asarray(out, dtype=object)

        # if len(pose_preds) >= 1:
        #     for kpt in pose_preds:
        #         pose_model.draw_pose(kpt, image_bgr)  # draw the poses

        # box, track_id, mask, pose
        # outputs_ = self.tracker.update(np.copy(pred_boxes))
        outputs = self.tracker.update(np.copy(pred_boxes))
        # s1, s2 = outputs_.shape
        # outputs = np.empty(shape=(s1, s2 + 1), dtype=object)

        # outputs[:, :-1] = outputs_[:]

        # for obi, obj in enumerate(outputs_):
        #     track_id = obj[1]
        #     x1, y1, x2, y2 = obj[0]
            # bottom_center = np.asarray([(x1 + x2) / 2, y2], dtype=np.int32)
            # crossed = self.hbu_lc.update(bottom_center, track_id)
            # outputs[obi, -1] = crossed

        # anonymization module
        if len(outputs) > 0:
            image_bgr = anonymize(image_bgr, outputs[:,0].copy(), outputs[:,2].copy(), outputs[:,3].copy())

        # fps = 1 / (time.time() - last_time)
        # print(f"\rFPS: {fps:.02f}", end='')

        # if self.save_video:
        #     self.writer.write(image_bgr)
        #
        # if self.show_output:
        #     cv2.imshow('demo', image_bgr)
        #     if cv2.waitKey(1) & 0XFF == ord('q'):
        #         break

        # ret, image_bgr = self.vidcap.get_next_frame()

        # if self.save_video:
        #     self.writer.stop()
        #
        # if self.show_output:
        #     cv2.destroyAllWindows()
        #
        # self.vidcap.stop()
        #
        # if self.save_detections:
        #     to_save = np.asarray(to_save, dtype=object)
        #     dest_file = self.video_path[:-3] + "npz"
        #     np.savez_compressed(dest_file, to_save)


        return image_bgr, outputs