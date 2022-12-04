# -*- coding: utf-8 -*-
# ---------------------

import os


PYTHONPATH = '..:.'
if os.environ.get('PYTHONPATH', default=None) is None:
    os.environ['PYTHONPATH'] = PYTHONPATH
else:
    os.environ['PYTHONPATH'] += (':' + PYTHONPATH)

import yaml
import socket
from path import Path

import numpy as np

from gui_init_points import LineSelector

class Conf(object):
    HOSTNAME = socket.gethostname()


    def __init__(self, settings_name):
        # type: (str) -> None
        """
        :param settings_name: name of the configuration file with app settings
        """

        self.conf_name = settings_name

        # print project name and host name
        self.project_name = Path(__file__).parent.parent.basename()
        m_str = f'┃ {self.project_name}@{Conf.HOSTNAME} ┃'
        u_str = '┏' + '━' * (len(m_str) - 2) + '┓'
        b_str = '┗' + '━' * (len(m_str) - 2) + '┛'
        print(u_str + '\n' + m_str + '\n' + b_str)

        # if the configuration file is not specified
        # try to load a configuration file based on the experiment name
        conf_file_path = Path(__file__).parent / (self.conf_name + '.yaml')

        # read the YAML configuation file
        conf_file = open(conf_file_path, 'r')
        y = yaml.load(conf_file, Loader=yaml.Loader)

        ## getting the hostname by socket.gethostname() method
        hostname = socket.gethostname()
        ## getting the IP address using socket.gethostbyname() method
        addr = socket.gethostbyname(hostname)

        self.res_path = Path(__file__).parent.parent / 'resources'

        self.worker_id = y.get('WORKER_ID')  # type: int
        self.area_capacity = y.get('AREA_CAPACITY', 4)  # type: float
        self.service = y.get('SERVICE', 'all')  # type: str
        self.eta = y.get('ETA', 1)  # type: float
        self.tau = y.get('TAU', 1.5)  # type: float
        self.beta = y.get('BETA', 1)  # type: float
        self.grid_scale = y.get('GRID_SCALE', 1.0)  # type: float
        self.video_path = y.get('VIDEO_PATH', "")  # type: str

        dst_pts = y.get('DEFAULT_REAL_WORLD_POINTS')
        self.dst_points = np.array(dst_pts) * self.grid_scale
        self.src_points = None
        self.calib_file_path = Path(__file__).parent / "calib.yaml"

        if not self.calib_file_path.exists():
            LineSelector(self.video_path, out_calib_file=self.calib_file_path).run()
        calib_file = open(self.calib_file_path, 'r')
        yc = yaml.load(calib_file, Loader=yaml.Loader)
        self.src_points = np.asarray(yc.get("homo_polygon"), dtype='float')
        self.nowalk_points = np.asarray(yc.get("nowalk_polygon"), dtype='float')


