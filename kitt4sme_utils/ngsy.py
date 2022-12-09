from fipy.ngsi.entity import BaseEntity, FloatAttr, TextAttr, ArrayAttr, BoolAttr
from kitt4sme_utils.fiware import orion_client
from typing import Optional

class WorkerEntity(BaseEntity):
    """
    type: string, type of entity
    num_obj: number of objects detected
    e_b_t: list [eta, beta, tau]
    area_capacity: max area capacity
    warning_area: list of N pairs of x,y values representing a polygon [x0,y0,x1,y1,x2,y2,...xn,yn]
    bboxes: list of 3 pairs of x,y values representing the bbox coords and the bottom center point in the real world [x_min, y_min, x_max, y_max, x0, y0]
    poses: list of 17 pairs of x,y values representing the human body pose coords
    service_type: string, AI service request, [LineCrossing, FallDetection, PandemicMonitoring] or 'all' to ask all of them
    """
    type = 'ai4sdw_worker'
    warning_area: Optional[ArrayAttr]
    num_obj: Optional[FloatAttr]
    e_b_t: Optional[ArrayAttr]
    area_capacity: Optional[FloatAttr]
    centers: Optional[ArrayAttr]
    poses: Optional[ArrayAttr]
    src_points: Optional[ArrayAttr]
    dst_points: Optional[ArrayAttr]


class AI4SDW_services(BaseEntity):
    type = 'ai4sdw_service'
    area_crossed: Optional[FloatAttr]
    fall_pred: Optional[FloatAttr]
    risk_leve: Optional[FloatAttr]

def send(cnf, data):
    """
    :param cnf: configuration file from witch get info like about worker id and area
    :param data: list of objects, every one contains dets, track_id, mask, pose
    :return:
    """
    orion = orion_client()

    boxes = []
    poses = []

    for dets, track_id, mask, pose, pose_class in data:
        x1, y1, x2, y2 = dets
        xy_center = [(x1+x2)/2, y2]
        boxes.extend(xy_center)
        poses.extend(pose.reshape(-1).tolist())

    assert boxes.__len__() % 2 == 0 and poses.__len__() % 34 == 0, "somethings wrong with data conversion for kitt4sme platform"

    to_send = WorkerEntity(
                    id=cnf.worker_id,
                    warning_area=ArrayAttr.new(cnf.nowalk_area.reshape(-1).tolist()),
                    num_obj=FloatAttr.new(len(data)),
                    centers=ArrayAttr.new(boxes),
                    poses=ArrayAttr.new(poses),
                    e_b_t=ArrayAttr.new([cnf.eta, cnf.beta, cnf.tau]),
                    area_capacity=FloatAttr.new(cnf.area_capacity),
                    src_points=ArrayAttr.new(cnf.src_points.tolist()),
                    dst_points=ArrayAttr.new(cnf.dst_points.tolist())
                )

    orion.upsert_entities([to_send])