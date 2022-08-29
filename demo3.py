import time
import numpy as np
import cv2
import torch
from path import Path

from back_end.yolox.detect import YoloX
from back_end.pose_resnet.detect import PoseNet

from hbu_services.fall_detector.detect import FallDetector
from hbu_services.line_crossing.LineCrossing import LineCrossing
from hbu_services.tracker.sort import Sort


def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    w_path = "back_end/yolox/yolox_m.pth"
    box_model = YoloX(device=device, weightsPath=w_path, num_classes=2, get_top2=False)
    pose_model = PoseNet(device=device)

    hbu_lc = LineCrossing(line=[[50,50],[150,50]])  # todo definire linea con UI
    fall_det = FallDetector(device=device)
    tracker = Sort(max_age=1, min_hits=3)

    vidcap = cv2.VideoCapture("out.mp4")

    while True:
        ret, image_bgr = vidcap.read()
        if ret:
            last_time = time.time()

            img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            pred_boxes = box_model.predict(img)
            pose_preds = pose_model.predict(img, pred_boxes)

            if len(pose_preds) >= 1:
                for kpt in pose_preds:
                    pose_model.draw_pose(kpt, image_bgr)  # draw the poses

            fps = 1 / (time.time() - last_time)
            cv2.putText(image_bgr, 'fps: ' + "%.2f" % (fps), (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

            cv2.imshow('demo', image_bgr)
            if cv2.waitKey(1) & 0XFF == ord('q'):
                break
        else:
            print('cannot load the video.')
            break

    cv2.destroyAllWindows()
    vidcap.release()


def main2():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    w_path = "back_end/yolox/yolox_m.pth"
    box_model = YoloX(device=device, weightsPath=w_path, num_classes=2, get_top2=False)
    pose_model = PoseNet(device=device)

    root = Path("/goat-nas/Datasets/fall_detection_dataset")
    dirs = root.dirs()
    dirs = [d for d in dirs if d.name == "tmp"]

    # out_subdir = "kpts"
    data = []
    for d in dirs:
        img_folder = (d / "rgb")
        imgs = img_folder.files("*.png")
        data.extend(imgs)

    for ii, img_path in enumerate(data):
        image_bgr = cv2.imread(img_path)

        img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pred_boxes = box_model.predict(img)
        pose_preds = pose_model.predict(img, pred_boxes)

        if len(pose_preds) >= 1:
            for kpt in pose_preds:
                # pose_model.draw_pose(kpt,image_bgr) # draw the poses
                pose_model.draw_plines(kpt,image_bgr) # draw the poses

            for pts in pose_preds[0]:
                cv2.circle(image_bgr,(int(pts[0]), int(pts[1])), 4, [255,0,0], -1)
            cv2.imshow("", image_bgr)
            cv2.waitKey()

            # out_path = img_path.parent.parent / out_subdir
            # out_path.makedirs_p()
            # with open(out_path / img_path.name.replace(".png", ".pkl"), "wb") as f:
            #     pickle.dump(pose_preds, f)

        print(f"\rprogress: {(ii/len(data)*100):.02f},  {img_path.parent.parent.name}", end='')

if __name__ == '__main__':
    main()
    # main2()
#