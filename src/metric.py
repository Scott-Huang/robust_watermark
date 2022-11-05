import cv2 as cv
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

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
    
    def evaluate(self, watermark_img, watermark_mask, orig_img):
        raise NotImplementedError
    
    def __call__(self, watermark_img, watermark_mask, orig_img):
        return self.evaluate(watermark_img, watermark_mask, orig_img)

# just an example for the interface, not a real metric
class Fast_Marching(RMetric): 
    # 1 + similarity between unpred and original image - similarity between pred and original image
    def evaluate(self, watermark_img, watermark_mask, orig_img):
        if watermark_img.shape[2] == 4:
            watermark_img = cv.cvtColor(watermark_img, cv.COLOR_RGBA2BGR)
        if orig_img.shape[2] == 4:
            orig_img = cv.cvtColor(orig_img, cv.COLOR_RGBA2BGR)
        # remove the watermark by inpainting
        inpainted_img = cv.inpaint(watermark_img,watermark_mask,3,cv.INPAINT_TELEA)
        base = np.zeros((watermark_img.shape[0], watermark_img.shape[1], 3), dtype=np.uint8)
        base[watermark_mask==0] = orig_img[watermark_mask==0]
        return 1+self.qmetric(base, orig_img)-self.qmetric(inpainted_img, orig_img)
