import cv2

import threading
from queue import Queue
import time


class Write_cv(threading.Thread):

    def __init__(self, dest_file, fourcc, fps, shape, queue, queue_stop):

        super().__init__()

        self.dest_file = dest_file
        self.fps = fps

        self.queue = queue
        self.queue_stop = queue_stop
        if fourcc == "xvid":
            fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
        elif fourcc == "mjpg":
            fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        elif fourcc == "mp4v":
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

        self.writer = cv2.VideoWriter(self.dest_file, fourcc, fps, shape)

    def run(self):

        while True:
            if not self.queue.empty():
                frame = self.queue.get()
                self.writer.write(frame)

            if self.queue_stop.full() and self.queue.empty():
                self.queue_stop.get()
                break

            time.sleep(1/self.fps*2)

        self.writer.release()


class VStreamWriter(object):
    def __init__(self, dest_file, fourcc, fps, shape):
        assert fourcc in ["xvid", "mjpg"]

        self.queue = Queue(maxsize=1)
        self.queue_stop = Queue(maxsize=1)
        self.video_writer = Write_cv(dest_file=dest_file, fourcc=fourcc, fps=fps,  shape=shape, queue=self.queue, queue_stop=self.queue_stop)
        self.video_writer.start()

    def stop(self):
        self.queue_stop.put(None)

    def write(self, frame):
        self.queue.put(frame)

