from pathlib import Path

from PIL import Image
import numpy as np

from .utils import get_mask_generator, generate_masks, adjust_logit_map_numba


def main(img_ind=1):
    print("- loading image")
    image_paths = sorted(list(Path("ours/images").glob("*p512.tiff")))
    image_path = image_paths[img_ind]
    img = Image.open(image_path).convert("RGB")
    print("- getting mask generator")
    mask_generator = get_mask_generator(
        points_per_side=32,
        stability_score_thresh=0.0,
        box_nms_thresh=1.0,
        pred_iou_thresh=0.0,
        points_per_batch=32,
        crop_nms_thresh=1.0,
        crop_n_layers=0,
        device="cuda:1",
    )
    print("- computing masks")
    masks, logits, predicted_ious, stability_scores = generate_masks(
        None, img, mask_generator
    )
    np.save("ours/tmp/logits.npy", logits)
    np.save("ours/tmp/predicted_ious.npy", predicted_ious)
    np.save("ours/tmp/stability_scores.npy", stability_scores)

main()
