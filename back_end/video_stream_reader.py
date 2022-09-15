from typing import Tuple

import cv2
import numpy as np

import threading
from queue import Queue


class Capture_cv(threading.Thread):

    def __init__(self, source, queue, queue_stop):

        super().__init__()

        self.source = source
        self.queue = queue
        self.queue_stop = queue_stop
        self.cap = cv2.VideoCapture(self.source)

    def run(self):

        while True:
            ret, frame = self.cap.read()

            if not self.queue.full():
                self.queue.put([ret, frame])

            if self.queue_stop.full():
                self.queue_stop.get()
                break


class VStreamReader(object):
    def __init__(self, stream_path):
        self.stream_path = stream_path

        vidcap = cv2.VideoCapture(stream_path)
        ret, image_bgr = vidcap.read()
        assert ret, f"somethings went wrong opening {stream_path}"
        self.fps = vidcap.get(cv2.CAP_PROP_FPS)
        vidcap.release()
        h, w = image_bgr.shape[:2]
        self.frame_shape_hw = (h, w)
        self.frame_shape_wh = (w, h)

        if isinstance(self.stream_path, str) and self.stream_path.find("rtsp://") >= 0:
            self.queue = Queue(maxsize=1)
            self.queue_stop = Queue(maxsize=1)
            self.video_cap = Capture_cv(source=self.stream_path, queue=self.queue, queue_stop=self.queue_stop)
            self.video_cap.start()
        elif isinstance(self.stream_path, int) or (
                isinstance(self.stream_path, str) and self.stream_path.find("rtsp://") < 0):
            self.video_cap = cv2.VideoCapture(self.stream_path)
            self.tot_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.current_frame = 0

    def stop(self):
        if isinstance(self.stream_path, int) or (
                isinstance(self.stream_path, str) and self.stream_path.find("rtsp://") < 0):
            self.video_cap.release()
        else:
            self.queue_stop.put(None)

    def get_next_frame(self):
        # type: () -> Tuple[bool, np.ndarray]
        """
        :return: tuple of 2 elements:
            >> read_ok: boolean value that is `False` if no frames has been grabbed; `True` otherwise
            >> frame_bgr: current frame; numpy array with shape (H, W, 3)
        """

        if isinstance(self.stream_path, str) and self.stream_path.find("rtsp://") >= 0:
            read_ok, frame_bgr = self.queue.get()

        elif isinstance(self.stream_path, int) or (
                isinstance(self.stream_path, str) and self.stream_path.find("rtsp://") < 0):
            read_ok, frame_bgr = self.video_cap.read()
            self.current_frame += 1

        return read_ok, frame_bgr

