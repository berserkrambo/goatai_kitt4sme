import numpy as np
import cv2

from back_end.pose_resnet.detect import SKELETON

colors = [[h, int(100 * 2.55), int(100 * 2.55)] for h in range(0, 180, 4)]

def draw_on_image(image_bgr, plot, outputs):
    if "mask" in plot:
        masked_img = image_bgr.copy()
        for dets, track_id, mask, pose, pose_class in outputs:
            color = colors[track_id % len(colors)]
            color[1] = int(50 * 2.55)
            color2_rgb = cv2.cvtColor(np.asarray([[color]], dtype='uint8'), cv2.COLOR_HSV2BGR)[0][0]

            masked_img = np.where(mask[..., None], color2_rgb, masked_img)
        image_bgr = cv2.addWeighted(image_bgr, 0.6, masked_img, 0.4, 0)

    if "box" in plot:
        for dets, track_id, mask, pose, pose_class in outputs:
            color = colors[track_id % len(colors)]
            color1_rgb = cv2.cvtColor(np.asarray([[color]], dtype='uint8'), cv2.COLOR_HSV2BGR)[0][0]
            x1, y1, x2, y2 = dets.astype(np.int32)

            cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color1_rgb.tolist(), 2)

    if "track" in plot:
        for dets, track_id, mask, pose, pose_class in outputs:
            color = colors[track_id % len(colors)]

            x1, y1, x2, y2 = dets.astype(np.int32)
            cv2.putText(image_bgr, f"{track_id}", (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, [50, 50, 50], 2)
            bottom_center = np.asarray([(x1 + x2) / 2, y2], dtype=np.int32)
            # cv2.circle(image_bgr, bottom_center, 4, [0, 0, 255] if crossed else [255, 0, 0], -1)
            cv2.circle(image_bgr, bottom_center, 4, [0, 0, 255], -1)

    if "pose" in plot:
        for dets, track_id, mask, pose, pose_class in outputs:
            for ln in SKELETON[4:]:
                cv2.line(image_bgr, pose[ln[0]], pose[ln[1]], [180, 180, 180], 2)

    if "pose_c" in plot:
        for dets, track_id, mask, pose, pose_class in outputs:
            x1, y1, x2, y2 = dets.astype(np.int32)
            cv2.putText(image_bgr, f"{pose_class}", (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        [0, 127, 255], 2)
