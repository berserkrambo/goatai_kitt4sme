import numpy as np
import sys
from roughnator.ngsy import LineCrossing, BoolAttr


def coords_to_points(coords):

    return [Point(ci[0], ci[1]) for ci in coords]


class Point:
    def __init__(self, x, y):
        """
        A point specified by (x,y) coordinates in the cartesian plane
        """
        self.x = x
        self.y = y


class Polygon:
    def __init__(self, points):
        """
        points: a list of Points in clockwise order.
        """
        self.points = points
        self.edges = self.get_edges()


    def get_edges(self):
        ''' Returns a list of tuples that each contain 2 points of an edge '''
        edge_list = []
        for i,p in enumerate(self.points):
            p1 = p
            p2 = self.points[(i+1) % len(self.points)]
            edge_list.append((p1,p2))

        return edge_list

    def contains(self, point):
        # _huge is used to act as infinity if we divide by 0
        _huge = sys.float_info.max
        # _eps is used to make sure points are not on the same line as vertexes
        _eps = 0.00001

        # We start on the outside of the polygon
        inside = False
        for edge in self.edges:
            # Make sure A is the lower point of the edge
            A, B = edge[0], edge[1]
            if A.y > B.y:
                A, B = B, A

            # Make sure point is not at same height as vertex
            if point.y == A.y or point.y == B.y:
                point.y += _eps

            if (point.y > B.y or point.y < A.y or point.x > max(A.x, B.x)):
                # The horizontal ray does not intersect with the edge
                continue

            if point.x < min(A.x, B.x):  # The ray intersects with the edge
                inside = not inside
                continue

            try:
                m_edge = (B.y - A.y) / (B.x - A.x)
            except ZeroDivisionError:
                m_edge = _huge

            try:
                m_point = (point.y - A.y) / (point.x - A.x)
            except ZeroDivisionError:
                m_point = _huge

            if m_point >= m_edge:
                # The ray intersects with the edge
                inside = not inside
                continue

        return inside


def in_hull(p, polygon, worker):
    polygon = np.asarray(polygon)
    p = np.asarray(p)
    polygon = Polygon(coords_to_points(polygon))
    goods = [polygon.contains(Point(pi[0], pi[1])) for pi in p]

    return LineCrossing(id=worker.id, status=BoolAttr.new(True if len(goods) > 0 else False))