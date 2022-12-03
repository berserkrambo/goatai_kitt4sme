from fipy.ngsi.entity import BaseEntity, FloatAttr, TextAttr, BoolAttr, ArrayAttr


class WorkerEntity(BaseEntity):
    """
    type: string, type of entity
    e_b_t: list [eta, beta, tau]
    area_capacity: max area capacity
    warning_area: list of N pairs of x,y values representing a polygon [x0,y0,x1,y1,x2,y2,...xn,yn]
    bboxes: list of 3 pairs of x,y values representing the bbox coords and the bottom center point in the real world [x_min, y_min, x_max, y_max, x0, y0]
    poses: list of 17 pairs of x,y values representing the human body pose coords
    service_type: string, AI service request, [LineCrossing, FallDetection, PandemicMonitoring] or 'all' to ask all of them
    """
    type = 'ai4sdw_worker'
    e_b_t = ArrayAttr
    area_capacity = FloatAttr
    warning_area= ArrayAttr
    bboxes: ArrayAttr
    poses: ArrayAttr
    service_type: TextAttr
    # frame: ArrayAttr

def send(worker_id, nowalk_area, boxes, poses, e_b_t, area_capacity, service):
    # orion = orion_client()
    # orion.upsert_entities([data])

    to_send = WorkerEntity(
                    id=conf["WORKER_ID"],
                    area=ArrayAttr.new(area),
                    bboxes=ArrayAttr.new(boxes),
                    poses=ArrayAttr.new(pose),
                    e_b_t=ArrayAttr.new([1, 3, 1]),
                    area_capacity=FloatAttr.new(5),
                    service_type=TextAttr.new("all")
                )