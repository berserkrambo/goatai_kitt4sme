import cv2
import numpy as np

UI_COL1 = (127, 252, 3)[::-1]  # light green
UI_COL2 = (63, 126, 1)[::-1]  # dark green
UI_COL3 = (240, 0, 0)[::-1]  # light red
UI_COL4 = (152, 0, 0)[::-1]  # dark red


def rescale_point(point, scale):
    p = np.array(point) / scale
    p = np.round(p)
    return int(p[0]), int(p[1])


class LineSelector(object):

    def __init__(self, video_path, win_name):
        self.cap = cv2.VideoCapture(video_path)
        assert self.cap.isOpened(), f"cap not opened for {video_path}"
        _, self.background = self.cap.read()
        self.frame = self.background.copy()

        h, w, _ = self.frame.shape
        self.frame_height, self.frame_width = h, w

        self.scale = 1280 / w

        self.state = 'empty'

        self.nowalk_poly = []
        self.homo_poly = []
        self.out_dict = {
            'homo_polygon': None,
            'nowalk_polygon': None
        }

        self.win_name = win_name
        cv2.namedWindow(self.win_name, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
        cv2.setMouseCallback(self.win_name, self.click_cb)
        cv2.setWindowProperty(
            self.win_name,
            cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
        )

        self.stop = False

        self.texts = (
            f'Usa il tasto sinistro del mouse per definire i punti per l\'omografia.',
            f'Usa il tasto destro del mouse per definire almeno 4 punti del poligono per l\'area di divieto.',
            f'Premi [INVIO] per confermare la selezione.',
            f'Premi [F] resettare la selezione.'
        )
        self.next_frame()

    def draw_header(self, img):
        w, h = 1100, 200
        w0, h0 = (img.shape[1] - w) // 2, 64
        top_bar = img[h0:h0 + h, w0:w0 + w]
        top_bar = top_bar * 0.6
        top_bar = top_bar.astype(np.uint8)
        top_bar = cv2.GaussianBlur(top_bar, ksize=(25, 25), sigmaX=13)

        img[h0:h0 + h, w0:w0 + w] = top_bar
        img = cv2.rectangle(
            img, pt1=(w0, h0), pt2=(w0 + w, h0 + h),
            color=(255, 255, 255), thickness=3
        )

        for i, t in enumerate(self.texts):
            font_scale = 0.76
            tk = 2
            w, h = cv2.getTextSize(t, 0, font_scale, 2)[0]
            img = cv2.putText(
                img=img, text=t, org=(w0 + 30, h0 + 30 + h + h * i * 2),
                fontFace=0, fontScale=font_scale, color=(255, 255, 255),
                thickness=tk, lineType=cv2.LINE_AA, bottomLeftOrigin=False
            )

        return img

    def next_frame(self):
        read_ok, bck = self.cap.read()
        if read_ok:
            bck = self.draw_header(bck)
            self.background = bck
            self.frame = self.background.copy()

    def click_cb(self, event, x, y, *_args):

        if event == cv2.EVENT_LBUTTONUP:
            self.homo_poly.append([x, y])

        if event == cv2.EVENT_RBUTTONUP:
            self.nowalk_poly.append([x, y])

    def render(self):
        self.frame = self.background.copy()

        self.frame = cv2.resize(
            self.frame, (0, 0), fx=self.scale, fy=self.scale,
            interpolation=cv2.INTER_AREA
        )

        for center in self.homo_poly:
            if center is not None:
                self.frame = cv2.circle(
                    self.frame, center=center, radius=8,
                    thickness=-1, color=UI_COL1,
                    lineType=cv2.LINE_AA
                )
                self.frame = cv2.circle(
                    self.frame, center=center, radius=3,
                    thickness=-1, color=UI_COL2,
                    lineType=cv2.LINE_AA
                )

        for center in self.nowalk_poly:
            if center is not None:
                self.frame = cv2.circle(
                    self.frame, center=center, radius=8,
                    thickness=-1, color=UI_COL3,
                    lineType=cv2.LINE_AA
                )
                self.frame = cv2.circle(
                    self.frame, center=center, radius=3,
                    thickness=-1, color=UI_COL4,
                    lineType=cv2.LINE_AA
                )

        if self.nowalk_poly.__len__() >= 3:
            self.frame = cv2.polylines(self.frame, [np.asarray(self.nowalk_poly)], True, UI_COL3)

        cv2.imshow(self.win_name, self.frame)

    def run(self):
        while not self.stop:
            self.render()
            key = cv2.waitKey(30)

            if key == -1:
                continue

            if key == 13:
                self.nowalk_poly = [rescale_point(p, self.scale) for p in self.nowalk_poly]
                self.homo_poly = [rescale_point(p, self.scale) for p in self.homo_poly]
                self.out_dict = {'homo_polygon': self.homo_poly, 'nowalk_polygon': self.nowalk_poly, "frame_height": self.frame_height,
                                 "frame_width": self.frame_width}

                if all([len(self.homo_poly) >= 4, len(self.nowalk_poly) >= 4]):
                    self.stop = True

            elif chr(key) == 'f':
                self.nowalk_poly = []
                self.homo_poly = []
                self.next_frame()

        cv2.destroyWindow(self.win_name)
        return self.out_dict

    def __del__(self):
        self.stop = True


def demo():
    od = LineSelector(video_path='resources/MOT20-02-raw-cut.mp4', win_name='GUI for points initialization').run()
    print(od)


if __name__ == '__main__':
    demo()
