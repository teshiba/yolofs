import numpy as np
import ctypes

#####################################################################
from ahk import AHK
from mss import mss

#####################################################################
# arguments
#####################################################################
windowTitle = ""

#####################################################################
# init vars
#####################################################################

class WindowCapture:
    #####################################################################
    def __init__(self, windowTitle):
        self.windowTitle = self.__getTitle(windowTitle)


    #####################################################################
    def __getTitle(self, window_title):
        ahk = AHK()
        wins = list(ahk.windows())
        titles = [win.title for win in wins]
        for t in titles:
            if window_title in t:
                return t

    #####################################################################
    def __getWindowRectFromName(self, TargetWindowTitle):
        TargetWindowHandle = ctypes.windll.user32.FindWindowW(0, TargetWindowTitle)
        Rectangle = ctypes.wintypes.RECT()
        ctypes.windll.user32.GetWindowRect(
            TargetWindowHandle, ctypes.pointer(Rectangle))
        return   {
            "top": Rectangle.top,
            "left": Rectangle.left,
            "width": Rectangle.right - Rectangle.left,
            "height": Rectangle.bottom - Rectangle.top,
            "mon": 0,
        }

    #####################################################################
    def GetImage(self):
        """Get Window caputured image

        Returns:
            ndarray: window image
        """
        with mss() as sct:
            monitor = self.__getWindowRectFromName(self.windowTitle)
            screenshot = np.array(sct.grab(monitor))[:, :, :3]

        return screenshot