import time
import numpy as np
import cv2
import torch
from path import Path

import click

from back_end.yolox.detect import YoloX
from back_end.pose_resnet.detect import PoseNet, SKELETON

from hbu_services.fall_detector.detect import FallDetector
from hbu_services.line_crossing.LineCrossing import LineCrossing
from hbu_services.tracker.sort import Sort
from hbu_services.anonymizator.anonymize import anonymize

from back_end.yalact.detect import Yalact

from back_end.video_stream_reader import VStreamReader
from back_end.video_stream_writer import VStreamWriter


class AI4SDW:
    def __init__(self, video_path, save_video, show_output, plot):
        self.video_path = video_path
        self.save_video = save_video
        self.show_output = show_output

        self.plot = plot
        self.colors = [[h, int(100 * 2.55), int(100 * 2.55)] for h in range(0, 180, 4)]
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print(self.device)

        # w_path = "back_end/yolox/yolox_m.pth"
        # box_model = YoloX(device=device, weightsPath=w_path, num_classes=2, get_top2=False)
        self.box_model = Yalact(device=self.device)
        self.pose_model = PoseNet(device=self.device)

        self.line = [[320, 210], [480, 410]]
        self.hbu_lc = LineCrossing(line=self.line)  # todo definire linea con UI
        self.fall_det = FallDetector(device=self.device)
        self.tracker = Sort(max_age=60, min_hits=3)

    def run(self):

        self.vidcap = VStreamReader(self.video_path)

        if self.save_video:
            dest_video = self.video_path.replace(".avi", "_result.avi").replace(".mp4", "_result.avi")
            if Path(dest_video).exists():
                dest_video = dest_video.replace(".avi", "0.avi")
            self.writer = VStreamWriter(dest_video, "xvid", self.vidcap.fps, self.vidcap.frame_shape_wh)

        ret, image_bgr = self.vidcap.get_next_frame()

        while ret:
            last_time = time.time()

            img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

            ids_p, pred_boxes, class_p, masks_p = self.box_model.predict(image_bgr.copy())
            pose_preds = self.pose_model.predict(img, pred_boxes)
            pose_class = self.fall_det.predict(pose_preds)

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
                    o_i.extend([pose_class[i]])  # pose class
                    o_i = np.asarray(o_i, dtype=object)
                    out.append(o_i)
                pred_boxes = np.asarray(out, dtype=object)

            #  todo comment draw fun
            # if len(pose_preds) >= 1:
            #     for kpt in pose_preds:
            #         pose_model.draw_pose(kpt, image_bgr)  # draw the poses

            outputs_ = self.tracker.update(np.copy(pred_boxes))
            s1, s2 = outputs_.shape
            outputs = np.empty(shape=(s1, s2 + 1), dtype=object)
            outputs[:, :-1] = outputs_[:]
            for obi, obj in enumerate(outputs_):
                track_id = obj[1]
                x1, y1, x2, y2 = obj[0]
                bottom_center = np.asarray([(x1 + x2) / 2, y2], dtype=np.int32)
                crossed = self.hbu_lc.update(bottom_center, track_id)
                outputs[obi, -1] = crossed

            # anonymization module
            if len(outputs) > 0:
                image_bgr = anonymize(image_bgr, outputs[:,0].copy(), outputs[:,3].copy(), outputs[:,4].copy())

                ######## drawing session #################

            if "mask" in self.plot:
                masked_img = image_bgr.copy()
                for dets, track_id, confs, mask, pose, pose_class, speed, crossed in outputs:
                    color = self.colors[track_id % len(self.colors)]
                    color[1] = int(50 * 2.55)
                    color2_rgb = cv2.cvtColor(np.asarray([[color]], dtype='uint8'), cv2.COLOR_HSV2BGR)[0][0]

                    masked_img = np.where(mask[..., None], color2_rgb, masked_img)
                image_bgr = cv2.addWeighted(image_bgr, 0.6, masked_img, 0.4, 0)

            for dets, track_id, confs, mask, pose, pose_class, speed, crossed in outputs:
                color = self.colors[track_id % len(self.colors)]
                color1_rgb = cv2.cvtColor(np.asarray([[color]], dtype='uint8'), cv2.COLOR_HSV2BGR)[0][0]

                x1, y1, x2, y2 = dets.astype(np.int32)

                if "box" in self.plot:
                    cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color1_rgb.tolist(), 2)

                if "pose_class" in self.plot:
                    cv2.putText(image_bgr, f"{self.fall_det.label_dict[pose_class]}", (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                [0, 127, 255], 2)

                if "track" in self.plot:
                    cv2.putText(image_bgr, f"{track_id}", (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, [50, 50, 50], 2)
                    bottom_center = np.asarray([(x1 + x2) / 2, y2], dtype=np.int32)
                    cv2.circle(image_bgr, bottom_center, 4, [0, 0, 255] if crossed else [255, 0, 0], -1)

                if "pose" in self.plot:
                    for ln in SKELETON[4:]:
                        cv2.line(image_bgr, pose[ln[0]], pose[ln[1]], [180, 180, 180], 2)

            if "line" in self.plot:
                cv2.line(image_bgr, self.line[0], self.line[1], [0, 0, 150], 1)

                ######## drawing session #################

            fps = 1 / (time.time() - last_time)
            print(f"\rFPS: {fps:.02f}", end='')

            if self.save_video:
                self.writer.write(image_bgr)

            if self.show_output:
                cv2.imshow('demo', image_bgr)
                if cv2.waitKey(1) & 0XFF == ord('q'):
                    break

            ret, image_bgr = self.vidcap.get_next_frame()

        if self.save_video:
            self.writer.stop()

        if self.show_output:
            cv2.destroyAllWindows()

        self.vidcap.stop()


@click.command()
@click.option('--video_path', type=str, default=None)
@click.option('--save_video', is_flag=True)
@click.option('--show_output', is_flag=True)
@click.option('--plot', "-p", multiple=True, default=["box", "mask", "pose", "pose_class", "track", "line"])
@click.option('--no_plot', is_flag=True)
def main(video_path, save_video, show_output, plot, no_plot):
    if no_plot:
        plot = [""]
    ai4sdw = AI4SDW(video_path=video_path, save_video=save_video, show_output=show_output, plot=plot)
    ai4sdw.run()

if __name__ == '__main__':
    main()