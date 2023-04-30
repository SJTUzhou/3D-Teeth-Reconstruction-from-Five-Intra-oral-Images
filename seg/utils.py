import numpy as np
import skimage

from seg.seg_const import IMG_SHAPE


def get_contour_from_raw_pred(pred_label, mask_shape, thresh=0.5):
    pred_prob_map = skimage.transform.resize(pred_label, mask_shape)
    pred_mask = pred_prob_map > thresh
    pred_mask = skimage.morphology.skeletonize(pred_mask.astype(np.uint8))
    pred_edge_img = (255.0 * pred_mask).astype(np.uint8)
    return pred_edge_img


def predict_teeth_contour(model, imgfile, resized_width=800):
    img = skimage.io.imread(imgfile)
    h, w = img.shape[:2]
    scale = resized_width / w
    rimg = skimage.transform.resize(img, IMG_SHAPE)
    raw_pred = model.predict(rimg[None, :])
    raw_pred = np.squeeze(raw_pred)
    edge_pred = get_contour_from_raw_pred(
        raw_pred, (int(scale * h), int(scale * w)), thresh=0.5
    )
    return edge_pred
