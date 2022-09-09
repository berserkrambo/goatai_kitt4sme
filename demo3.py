import time
import numpy as np
import cv2
import torch
from path import Path

from back_end.yolox.detect import YoloX
from back_end.pose_resnet.detect import PoseNet, SKELETON

from hbu_services.fall_detector.detect import FallDetector
from hbu_services.line_crossing.LineCrossing import LineCrossing
from hbu_services.tracker.sort import Sort

from back_end.yalact.detect import Yalact


def main():
    colors = [[h, int(100 * 2.55), int(100 * 2.55)] for h in range(0, 180, 4)]

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # w_path = "back_end/yolox/yolox_m.pth"
    # box_model = YoloX(device=device, weightsPath=w_path, num_classes=2, get_top2=False)
    box_model = Yalact(device=device)
    pose_model = PoseNet(device=device)

    line = [[0, 359], [1279, 159]]
    hbu_lc = LineCrossing(line=line)  # todo definire linea con UI
    fall_det = FallDetector(device=device)
    tracker = Sort(max_age=1, min_hits=3)

    vidcap = cv2.VideoCapture("out2.mp4")
    ret, image_bgr = vidcap.read()

    h, w = image_bgr.shape[:2]
    fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    writer = cv2.VideoWriter("test.avi", fourcc, vidcap.get(cv2.CAP_PROP_FPS), (w, h))

    while ret:

        last_time = time.time()

        img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        ids_p, pred_boxes, class_p, masks_p = box_model.predict(image_bgr.copy())
        pose_preds = pose_model.predict(img, pred_boxes)
        pose_class = fall_det.predict(pose_preds)

        # set shape for tracker
        if len(pred_boxes) >= 1:
            # image_bgr = box_model.drawo_pred(ids_p, class_p, pred_boxes, masks_p, image_bgr) # todo comment this one

            out = []
            for i in range(len(pred_boxes)):
                o_i = []
                o_i.extend(pred_boxes[i].astype(np.float32))    # bbox
                o_i.extend([class_p[i]])                        # conf
                o_i.extend(np.expand_dims(masks_p[i], 0))       # mask
                o_i.extend([pose_preds[i].astype(np.int32)])                      # pose
                o_i.extend([pose_class[i]])                     # pose class
                o_i = np.asarray(o_i, dtype=object)
                out.append(o_i)
            pred_boxes = np.asarray(out, dtype=object)

        #  todo comment draw fun
        # if len(pose_preds) >= 1:
        #     for kpt in pose_preds:
        #         pose_model.draw_pose(kpt, image_bgr)  # draw the poses

        outputs_ = tracker.update(np.copy(pred_boxes))
        s1, s2 = outputs_.shape
        outputs = np.empty(shape=(s1, s2 + 1), dtype=object)
        outputs[:, :-1] = outputs_[:]
        for obi, obj in enumerate(outputs_):
            track_id = obj[1]
            x1, y1, x2, y2 = obj[0]
            bottom_center = np.asarray([(x1 + x2) / 2, y2], dtype=np.int32)
            crossed = hbu_lc.update(bottom_center, track_id)
            outputs[obi, -1] = crossed

        ######## drawing session #################

        masked_img = image_bgr.copy()
        for dets, track_id, confs, mask, pose, pose_class, speed, crossed in outputs:
            color = colors[track_id % len(colors)]
            color[1] = int(50 * 2.55)
            color2_rgb = cv2.cvtColor(np.asarray([[color]], dtype='uint8'), cv2.COLOR_HSV2BGR)[0][0]

            masked_img = np.where(mask[..., None], color2_rgb, masked_img)
        image_bgr = cv2.addWeighted(image_bgr, 0.6, masked_img, 0.4, 0)

        for dets, track_id, confs, mask, pose, pose_class, speed, crossed in outputs:
            color = colors[track_id % len(colors)]

            x1, y1, x2, y2 = dets.astype(np.int32)
            color1_rgb = cv2.cvtColor(np.asarray([[color]], dtype='uint8'), cv2.COLOR_HSV2BGR)[0][0]

            cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color1_rgb.tolist(), 2)
            cv2.putText(image_bgr, f"{fall_det.label_dict[pose_class]}", (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        [0, 127, 255], 2)
            cv2.putText(image_bgr, f"{track_id}", (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, [50, 50, 50], 2)

            bottom_center = np.asarray([(x1 + x2) / 2, y2], dtype=np.int32)
            cv2.circle(image_bgr, bottom_center, 4, [0,0,255] if crossed else [255,0,0], -1)

            # for jt in pose:
            #     cv2.circle(image_bgr, jt, 1, [70,60,60], -1)
            for ln in SKELETON[4:]:
                cv2.line(image_bgr, pose[ln[0]], pose[ln[1]], [180, 180, 180], 2)

        cv2.line(image_bgr, line[0], line[1], [0, 0, 150], 1)

        ######## drawing session #################

        fps = 1 / (time.time() - last_time)
        # cv2.putText(image_bgr, 'fps: ' + "%.2f" % (fps), (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

        writer.write(image_bgr)
        cv2.imshow('demo', image_bgr)

        if cv2.waitKey(1) & 0XFF == ord('q'):
            break

        ret, image_bgr = vidcap.read()

    cv2.destroyAllWindows()
    vidcap.release()
    writer.release()

if __name__ == '__main__':
    main()

#
