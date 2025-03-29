from pathlib import Path
from heapq import heappop, heappush

from PIL import Image
import numpy as np
import heapq
from numba import njit

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator, rle_to_mask


def get_mask_generator(
    points_per_side,
    stability_score_thresh,
    box_nms_thresh,
    pred_iou_thresh,
    points_per_batch,
    crop_nms_thresh,
    crop_n_layers,
    device,
):
    sam2_model = build_sam2(
        config_file="configs/sam2.1/sam2.1_hiera_t.yaml",
        ckpt_path="checkpoints/sam2.1_hiera_tiny.pt",
        device=device,
        mode="eval",
        apply_postprocessing=False,
    )
    mask_generator = SAM2AutomaticMaskGenerator(
        sam2_model,
        points_per_side=points_per_side,
        stability_score_thresh=stability_score_thresh,
        box_nms_thresh=box_nms_thresh,
        pred_iou_thresh=pred_iou_thresh,
        points_per_batch=points_per_batch,
        crop_nms_thresh=crop_nms_thresh,
        crop_n_layers=crop_n_layers,
    )
    return mask_generator


def generate_masks(path_to_image, image, mask_generator):
    assert (path_to_image is None and type(image) == Image.Image) or (
        type(path_to_image) == str and image is None
    ), "Either path_to_image or image should be provided, not both."
    if image is None:
        image = Image.open(path_to_image)
    image = np.array(image.convert("RGB"))
    mask_data = mask_generator._generate_masks(image)
    stability_scores = mask_data["stability_score"]
    logits = mask_data["low_res_masks"]
    ious = mask_data["iou_preds"]
    mask_data["segmentations"] = [
        rle_to_mask(rle) for rle in mask_data["rles"]
    ]  # masks
    masks = np.array(mask_data["segmentations"])
    return masks, logits, ious, stability_scores


def adjust_logit_map_numpy(data):
    """
    Adjust the logits so that, when thresholded at 0, the obtained mask has only one 4-connected component.
    To do this, we enforce that there should always exist a monotonically non-increasing path from the maximum to any other pixel.
    The only way for this to happen is to have logits outside the connected component that contains the max to be negative,
    else they are positive and there is an all positive path that connects them to the max,
    which contradicts that they are outside the max connected component.
    """
    assert type(data) == np.ndarray, "data should be a numpy array"
    C, H, W = data.shape
    adjusted_map = data.copy()
    visited = np.zeros_like(data, dtype=bool)

    for channel in range(C):
        print(f"making c1... channel {channel}", end="\r")
        max_index = np.argmax(adjusted_map[channel])
        seed_row, seed_col = divmod(max_index, W)
        seed_value = adjusted_map[channel, seed_row, seed_col]

        heap = [(-seed_value, seed_row, seed_col)]
        visited[channel, seed_row, seed_col] = True

        while heap:
            neg_val, row, col = heappop(heap)
            current_val = -neg_val
            deltas = [(1, 0), (-1, 0), (0, 1), (0, -1)]

            for d_row, d_col in deltas:
                nrow, ncol = row + d_row, col + d_col
                if 0 <= nrow < H and 0 <= ncol < W and not visited[channel, nrow, ncol]:
                    neighbor_value = adjusted_map[channel, nrow, ncol]
                    # To adjust the map we set each pixel to be the maximum value such that there is a monotonically non-increasing path to the max to the pixel.
                    # I think this changes the logit map the minimum possible in order to ensure one 4-connected component while thresholding.
                    if neighbor_value > current_val:
                        adjusted_map[channel, nrow, ncol] = current_val
                    heappush(heap, (-adjusted_map[channel, nrow, ncol], nrow, ncol))
                    visited[channel, nrow, ncol] = True

    print("- done making c1!")
    return adjusted_map


@njit
def adjust_logit_map_numba(logit_map):
    """
    Adjusts the input logit map so that the level lines are nested,
    decreasing values as little as possible using Numba for optimization.

    Parameters:
    - logit_map: numpy array of shape (256, 256)

    Returns:
    - adjusted_map: numpy array of shape (256, 256)
    """
    H, W = logit_map.shape
    adjusted_map = logit_map.copy()
    visited = np.zeros((H, W), dtype=np.uint8)

    # Find the seed pixel (maximum value). If multiple maxima, pick the first in raster order.
    max_index = np.argmax(logit_map)
    seed_i = max_index // W
    seed_j = max_index % W
    seed_value = logit_map[seed_i, seed_j]

    # Initialize the priority queue with the seed pixel
    heap = [(-seed_value, seed_i, seed_j)]
    visited[seed_i, seed_j] = True

    while heap:
        neg_value, i, j = heapq.heappop(heap)
        current_value = -neg_value  # Convert back to positive

        # Check each of the 4-connected neighbors
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < H and 0 <= nj < W and not visited[ni, nj]:
                neighbor_value = adjusted_map[ni, nj]
                if neighbor_value > current_value:
                    # Decrease the neighbor's value to the current value
                    adjusted_map[ni, nj] = current_value
                # Add the neighbor to the heap for further processing
                heapq.heappush(heap, (-adjusted_map[ni, nj], ni, nj))
                visited[ni, nj] = True  # Mark as visited

    return adjusted_map
