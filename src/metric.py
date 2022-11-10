import cv2 as cv
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from model.interface import get_model

# metric for evaluating the quality of the watermark, i.e. 
# the similarity between the original image and the watermarked image
class QMetric:
    def evaluate(self, watermark_img, orig_img):
        raise NotImplementedError

    def __call__(self, watermark_img, orig_img):
        return self.evaluate(watermark_img, orig_img)

class PSNR(QMetric):
    def evaluate(self, watermark_img, orig_img):
        return peak_signal_noise_ratio(orig_img, watermark_img)

class SSIM(QMetric):
    def evaluate(self, watermark_img, orig_img):
        return structural_similarity(orig_img, watermark_img, channel_axis=2)


# metric for evaluating the robustness of the watermark
class RMetric:
    def __init__(self, inpaint_model, qmetric:QMetric) -> None:
        self.inpaint_model = inpaint_model
        self.qmetric = qmetric
    
    def inpaint(self, watermark_img, watermark_mask):
        # remove the watermark by inpainting
        return NotImplementedError
    
    def show_inpaint(self, watermark_img, watermark_mask):
        return Image.fromarray(self.inpaint(watermark_img, watermark_mask))

    def evaluate(self, watermark_img, watermark_mask, orig_img):
        inpainted_img = self.inpaint(watermark_img, watermark_mask)
        base = np.zeros((watermark_img.shape[0], watermark_img.shape[1], watermark_img.shape[2]), dtype=np.uint8)
        base[watermark_mask==0] = orig_img[watermark_mask==0]
        # 1 + similarity between unpred and original image - similarity between pred and original image
        return 1+self.qmetric(base, orig_img)-self.qmetric(inpainted_img, orig_img)
    
    def __call__(self, watermark_img, watermark_mask, orig_img):
        return self.evaluate(watermark_img, watermark_mask, orig_img)

class Deepfill(RMetric):
    def __init__(self, inpaint_model, qmetric: QMetric) -> None:
        super().__init__(get_model('deepfill'), qmetric)
    def inpaint(self, watermark_img, watermark_mask):
        return self.inpaint_model(watermark_img, watermark_mask)
    def evaluate(self, watermark_img, watermark_mask, orig_img):
        if orig_img.shape[2] == 4:
            orig_img = cv.cvtColor(orig_img, cv.COLOR_RGBA2RGB)
        return super().evaluate(watermark_img, watermark_mask, orig_img)

class AOT_GAN(RMetric):
    def __init__(self, inpaint_model, qmetric: QMetric) -> None:
        super().__init__(get_model('aot_gan'), qmetric)
    def inpaint(self, watermark_img, watermark_mask):
        return self.inpaint_model(watermark_img, watermark_mask)
    def evaluate(self, watermark_img, watermark_mask, orig_img):
        if orig_img.shape[2] == 4:
            orig_img = cv.cvtColor(orig_img, cv.COLOR_RGBA2RGB)
        return super().evaluate(watermark_img, watermark_mask, orig_img)

class Stable_Diffussion(RMetric):
    def __init__(self, inpaint_model, qmetric: QMetric) -> None:
        super().__init__(get_model('stable_diffussion'), qmetric)
    def inpaint(self, watermark_img, watermark_mask):
        return self.inpaint_model(watermark_img, watermark_mask)
    def evaluate(self, watermark_img, watermark_mask, orig_img):
        if orig_img.shape[2] == 4:
            orig_img = cv.cvtColor(orig_img, cv.COLOR_RGBA2RGB)
        return super().evaluate(watermark_img, watermark_mask, orig_img)
