import numpy as np
import torch
import cv2 as cv
from typing import Any

class Base_Watermarker:
    @staticmethod
    def create_watermark(watermark: str|np.ndarray|torch.Tensor|Any, size=None, **args):
        raise NotImplementedError

    def watermark(self, image, position, watermark):
        raise NotImplementedError

class Example_Watermarker(Base_Watermarker):
    # Create a watermark from a string or an image
    # It's just a hand-made example, you can use any method you want
    @staticmethod
    def create_watermark(watermark: str|np.ndarray|torch.Tensor|Any, size=None, **args):
        def get_optimal_font_scale(text, width):
            for scale in reversed(range(0, 60, 1)):
                textSize = cv.getTextSize(text, fontFace=cv.FONT_HERSHEY_DUPLEX, fontScale=scale/10, thickness=1)
                new_width = textSize[0][0]
                if (new_width <= width):
                    print(new_width)
                    return scale/10
            return 1
        if isinstance(watermark, str):
            background = np.zeros((size[0], size[1], 4), dtype=np.uint8) # RGBA
            watermark = cv.putText(background, watermark, (0, size[0]//2), cv.FONT_HERSHEY_DUPLEX, 
                                    get_optimal_font_scale(watermark, size[1]), (255, 255, 255, 255), 3)
        elif size is not None:
            watermark = cv.resize(watermark, size)
        return watermark

    def watermark(self, image, position, watermark):
        assert image.shape[0] >= watermark.shape[0] + position[0]
        assert image.shape[1] >= watermark.shape[1] + position[1]
        image = image.copy()
        dst = image[position[0]:position[0]+watermark.shape[0], position[1]:position[1]+watermark.shape[1]]
        dst[watermark>0] = watermark[watermark>0]
        image[position[0]:position[0]+watermark.shape[0], position[1]:position[1]+watermark.shape[1]] = dst
        return image
