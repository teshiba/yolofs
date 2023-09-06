
from ultralytics import YOLO
from ultralytics.yolo.data.augment import LetterBox
from ultralytics.yolo.utils.plotting import Annotator, colors
from ultralytics.yolo.utils import ops
from copy import deepcopy

#####################################################################
import torch
import numpy as np

useCuda = True

class Yolov8Lib:
    #####################################################################
    def __init__(self, modelFile, poseMode):
        model = YOLO(modelFile)
        if useCuda:
            print("[Yolov8Lib] Use CUDA")
            model.to("cuda") # when using GPU.
        self.model = model
        self.isModePause = poseMode

    #####################################################################
    def __preprocess(self, img, size = 640):
        img = LetterBox(size, True)(image = img)
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)  # contiguous
        img = torch.from_numpy(img).to("cuda")
        img = img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        return img.unsqueeze(0)

    #####################################################################
    def __postprocess(self, preds, img, orig_img):
        preds = ops.non_max_suppression(preds,
                                        0.25,
                                        0.8,
                                        agnostic=False,
                                        max_det=100)

        for i, pred in enumerate(preds):
            shape = orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()

        return preds

    #####################################################################
    def __drawBoundingBox(self, pred, names, annotator):
        for *xyxy, conf, cls in reversed(pred):
            c = int(cls)  # integer class
            label =  f'{names[c]} {conf:.2f}' #
            annotator.box_label(xyxy, label, color = colors(c, True))

    #####################################################################
    def GenerateAnnotatedImage(self, image):
        """Generate Annotated Image

        Args:
            image (ndarray): source image

        Returns:
            ndarray: Annotated Image
        """
        if self.isModePause:
            results = self.model(image, line_width = 1)
            AnnotatedImage = results[0].plot()
        else:
            AnnotatedImage = deepcopy(image)
            annotator = Annotator(AnnotatedImage,
                                line_width = 1,
                                example = str(self.model.model.names))
            img = self.__preprocess(image)
            preds = self.model.model(img, augment = False)
            preds = self.__postprocess(preds, img, AnnotatedImage)
            self.__drawBoundingBox(preds[0], self.model.model.names, annotator)

        return AnnotatedImage