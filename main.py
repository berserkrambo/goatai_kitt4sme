import cv2
from path import Path
import click
import time
from back_end_logic import AI4SDW

from back_end.video_stream_reader import VStreamReader
from back_end.video_stream_writer import VStreamWriter
from utils import draw_on_image
from kitt4sme_utils import ngsy


@click.command()
@click.option('--video_path', type=str, default=None)
@click.option('--save_video', is_flag=True)
@click.option('--show_output', is_flag=True)
@click.option('--plot', "-p", multiple=True, default=["box", "mask", "pose", "pose_class", "track", "line"])
@click.option('--no_plot', is_flag=True)
@click.option('--save_detections', is_flag=True)
def main(video_path, save_video, show_output, plot, no_plot, save_detections):
    if no_plot:
        plot = [""]

    vidcap = VStreamReader(video_path)

    ai4sdw = AI4SDW()

    # if save_video:
    #     dest_video = video_path.replace(".avi", "_result.avi").replace(".mp4", "_result.avi")
    #     if Path(dest_video).exists():
    #         dest_video = dest_video.replace(".avi", "0.avi")
    #     writer = VStreamWriter(dest_video, "xvid", vidcap.fps, vidcap.frame_shape_wh)

    while True:
        ret, image_bgr = vidcap.get_next_frame()

        if ret:
            anonymized_image, outputs = ai4sdw.process_next_frame(image_bgr)

            if len(outputs) > 0:
                if not no_plot:
                    draw_on_image(image_bgr, plot, outputs)

                # leggi conf
                # sistema bbox per invio
                # ngsy.send()

        time.sleep(1/30)
        k = cv2.waitKey(1)

        if k == ord('q') or k == 27 or (not ret):
            print("stop")
            break


if __name__ == '__main__':
    main()

    # todo manda via ngsy
    # leggi calib file
