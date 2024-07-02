from mss import mss
import win32gui as win32
import numpy as np
import cv2
from PIL import Image


class Screenshot:
    def __init__(self):
        window_handle = win32.FindWindow(None, "BlueStacks App Player")
        client_rect = win32.GetClientRect(window_handle)
        window_location = win32.ClientToScreen(
            window_handle, (client_rect[1], client_rect[0])
        )
        self.win_rect = {
            "left": window_location[0],
            "top": window_location[1],
            "width": client_rect[2],
            "height": client_rect[3],
        }

    def _pil_frombytes(self, im):
        """Efficient Pillow version."""
        return Image.frombytes("RGB", im.size, im.bgra, "raw", "BGRX")

    def get_screenshot(self):
        with mss() as sct:
            im = sct.grab(self.win_rect)
            rgb = self._pil_frombytes(im)
            return np.array(rgb)

    def show_window(self, location, img) -> None:
        cv2.imshow("Client Screenshot", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.moveWindow("Client Screenshot", location[0], location[1])
        cv2.waitKey(1)
