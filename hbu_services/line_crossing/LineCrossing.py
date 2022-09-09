import numpy as np


class LineCrossing:
    """
    Line Crossing Service
    """
    def __init__(self, line):
        """
        :param line: 2x2 numpy array or list
        """
        self.line = np.asarray(line)
        if line[0][0] > line[1][0]:
            self.line = np.asarray([line[1], line[0]])
        assert self.line.shape == (2,2), "line shape not (2,2)"

        self.waiting_tracks = {}

    def update(self, point, track_id):
        """
        :param point: (x,y) point coord, usually the bottome center point of a bounding box
        :param track_id: track number identifier
        :return: False if point is above the line else True
        """

        v1 = self.line[1] - self.line[0]
        v2 = self.line[1] - point
        xp = v1[0] * v2[1] - v1[1] * v2[0]  # Cross product

        above = xp > 0

        if above:
            # self.waiting_tracks[track_id] = track_id
            return False
        elif not above:# and track_id in self.waiting_tracks:
            # self.waiting_tracks.pop(track_id)
            return True