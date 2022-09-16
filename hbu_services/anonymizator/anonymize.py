import numpy as np
import cv2

def anonymize(img, boxes, masks, poses):
    assert len(boxes) == len(masks) == len(poses)

    h,w = img.shape[:2]
    for i in range(len(boxes)):

        x1, y1, x2, y2 = int(boxes[i][0]), int(boxes[i][1]), int(boxes[i][2]), int(boxes[i][3])

        max_bbox_len = max(abs(y2-y1), abs(x2-x1))
        head_dim_r = max_bbox_len / 12

        min_x = np.min(poses[i][:5,0])
        min_y = min(y1, np.min(poses[i][:5,1]))
        max_x = np.max(poses[i][:5, 0])
        max_y = np.max(poses[i][:5, 1])

        head_coord = [
            min(int(max(0,   min_x - head_dim_r)), w-1),
            min(int(max(0,   min_y - head_dim_r)), h-1),
            max(int(min(w-1, max_x + head_dim_r)), 0),
            max(int(min(h-1, max_y + head_dim_r)), 0)
        ]

        img_crop = img[head_coord[1]:head_coord[3], head_coord[0]:head_coord[2]].copy()
        if img_crop.shape[0] <= 3 or img_crop.shape[1] <= 3:
            continue
        ksize = max(3,  max(head_coord[2] - head_coord[0], head_coord[3] - head_coord[1]))

        if ksize % 2 == 0:
            ksize += 1
        img_crop_blur = cv2.GaussianBlur(img_crop, (ksize, ksize), 0)

        mask_crop = masks[i][head_coord[1]:head_coord[3], head_coord[0]:head_coord[2]].copy()
        img[head_coord[1]:head_coord[3], head_coord[0]:head_coord[2]] = np.where(mask_crop[..., None], img_crop_blur, img_crop)

    return img
