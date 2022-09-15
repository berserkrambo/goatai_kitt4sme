import numpy as np
import cv2


def anonymize(img, boxes, masks, poses):
    assert len(boxes) == len(masks) == len(poses)

    for i in range(len(boxes)):
        shoulders_max_y = max(poses[i][5][1], poses[i][6][1])
        neck_y = int((poses[i][0][1] + shoulders_max_y) / 2)
        bbox_upper_bound = int(min(boxes[i][1], boxes[i][3]))
        x1, x2 = int(boxes[i][0]), int(boxes[i][2])

        img_crop = img[bbox_upper_bound:neck_y, x1:x2].copy()
        if img_crop.shape[0] <= 3 or img_crop.shape[1] <= 3:
            continue
        ksize = max(3,  int(abs(x2 - x1) / 5))

        if ksize % 2 == 0:
            ksize += 1
        img_crop_blur = cv2.GaussianBlur(img_crop, (ksize, ksize), 0)

        mask_crop = masks[i][bbox_upper_bound:neck_y, x1:x2].copy()
        img[bbox_upper_bound:neck_y, x1:x2] = np.where(mask_crop[..., None], img_crop_blur, img_crop)

    return img
