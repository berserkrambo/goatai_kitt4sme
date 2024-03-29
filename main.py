import signal
import cv2
import click
import time
from back_end_logic import AI4SDW

from back_end.video_stream_reader import VStreamReader
from utils import draw_on_image
from conf.conf import Conf
from kitt4sme_utils import ngsy
from fipy.docker import DockerCompose

from kitt4sme_utils.fiware import wait_on_orion, create_subscriptions
import numpy as np

docker = DockerCompose(__file__)

def signal_handler(signal, frame):
    docker.stop()

def bootstrap(env):
    docker.build_images()
    docker.start()

    wait_on_orion(env)

@click.command()
@click.option('--plot', "-p", multiple=True, default=[""], help='multiple value: ["box", "mask", "pose", "pose_c", "track"]')
@click.option('--env')
def main(plot, env):
    assert env in ["docker", "cluster"]
    signal.signal(signal.SIGINT, signal_handler)

    cnf = Conf("config")
    vidcap = VStreamReader(cnf.video_path)
    ai4sdw = AI4SDW()

    if env == "docker":
        docker.stop()
        try:
            bootstrap(env)
            create_subscriptions(env)
            print("sub created")
        except:
            docker.stop()
            return
    elif env == "cluster":
        create_subscriptions(env)
        print("sub created")

    nowalk = np.asarray(cnf.nowalk_area, dtype='int32').reshape((-1, 2))
    while True:
        ret, image_bgr = vidcap.get_next_frame()

        if ret:
            anonymized_image, outputs = ai4sdw.process_next_frame(image_bgr)

            if len(outputs) > 0:

                draw_on_image(image_bgr, plot, outputs)
                cv2.polylines(image_bgr, [nowalk], True, [255,0,0])
                cv2.imshow("", image_bgr)

                if env in ["cluster", "docker"]:
                    ngsy.send(cnf=cnf, data=outputs[:], env=env)

        k = cv2.waitKey(1)

        if k == ord('q') or k == 27 or (not ret):
            print("stop")
            break

if __name__ == '__main__':
    main()
