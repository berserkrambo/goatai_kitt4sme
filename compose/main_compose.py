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


docker = DockerCompose(__file__)


def bootstrap():
    docker.build_images()
    docker.start()

    wait_on_orion()

    create_subscriptions()
    print("sub created")

@click.command()
@click.option('--plot', "-p", multiple=True, default=[""], help='multiple value: ["box", "mask", "pose", "pose_class", "track", "line"]')

def main(plot):
    services_running = False
    cnf = Conf("config")
    vidcap = VStreamReader(cnf.video_path)
    ai4sdw = AI4SDW()

    try:
        bootstrap()
        services_running = True

        print('>>> sending machine entities to Orion...')

        while True:
            ret, image_bgr = vidcap.get_next_frame()

            if ret:
                anonymized_image, outputs = ai4sdw.process_next_frame(image_bgr)

                if len(outputs) > 0:
                    draw_on_image(image_bgr, plot, outputs)
                    cv2.imshow("", image_bgr)
                    ngsy.send(cnf=cnf, data=outputs[:])

            time.sleep(1/30)
            k = cv2.waitKey(1)

            if k == ord('q') or k == 27 or (not ret):
                print("stop")
                break

    except KeyboardInterrupt:
        if services_running:
            docker.stop()

if __name__ == '__main__':
    main()
