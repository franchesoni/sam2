import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image
import skimage.measure
import numpy as np
import heapq

from .utils import get_mask_generator, generate_masks, adjust_logit_map_numba


def get_all_logits(img_ind=1):
    st = time.time()
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
    masks, logits, predicted_ious, stability_scores, points = generate_masks(
        None, img, mask_generator
    )
    print("generating masks took", time.time() - st, "seconds")
    return logits, predicted_ious, stability_scores, points

def per_min_max(arr, per=0.03):
    arr = np.array(arr)
    min_val = np.percentile(arr, per * 100)
    max_val = np.percentile(arr, (1 - per) * 100)
    if max_val - min_val == 0:
        return arr
    return np.clip((arr - min_val) / (max_val - min_val), 0, 1) 

def nms_masks_gpu(masks: torch.BoolTensor, scores: torch.Tensor, iou_threshold: float):
    """
    A GPU version of nms_masks; inputs are
      masks: (M,H,W) bool
      scores: (M,)
    Returns:
      kept_masks: (K,H,W) bool
      kept_indices: List[int]  # original indices in the sorted order
    """
    M, H, W = masks.shape
    flat = masks.reshape(M, -1).float()  # (M, H*W)
    order = torch.argsort(torch.from_numpy(scores), descending=True)
    flat = flat[order]
    keep = []
    for i in range(M):
        m = flat[i]
        if not keep:
            keep.append(i)
        else:
            # compare to all previously kept
            prev = torch.stack([flat[j] for j in keep], dim=0)  # (K,HW)
            inter = (prev * m).sum(dim=1)
            union = prev.sum(dim=1) + m.sum() - inter
            if torch.all((inter / union) < iou_threshold):
                keep.append(i)
    kept = masks[order[keep]]
    kept_idx = order[keep].tolist()
    return kept, kept_idx



def iou_one_loop(
    masks1: torch.Tensor, masks2: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """
    One for‑loop over masks1. Vectorized over masks2.
    """
    M, H, W = masks1.shape
    N = masks2.shape[0]
    iou = torch.zeros((M, N), dtype=torch.float, device=masks1.device)
    area2 = masks2.view(N, -1).sum(dim=1)  # (N,)
    for i in range(M):
        m = masks1[i].float().view(1, H, W)  # (1,H,W)
        inter = (m * masks2).view(N, -1).sum(dim=1)  # (N,)
        area1 = m.sum()
        union = area1 + area2 - inter  # (N,)
        iou[i] = inter / (union + eps)
    return iou


def iou_two_loops(
    masks1: torch.Tensor, masks2: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """
    Two for‑loops over masks1 and masks2. Lowest memory overhead.
    """
    M, H, W = masks1.shape
    N = masks2.shape[0]
    iou = torch.zeros((M, N), dtype=torch.float, device=masks1.device)
    for i in range(M):
        m = masks1[i].float()
        area1 = m.sum()
        for j in range(N):
            p = masks2[j].float()
            inter = (m * p).sum()
            area2 = p.sum()
            union = area1 + area2 - inter
            iou[i, j] = inter / (union + eps)
    return iou


def incremental_mIoU(M, P_o, IoU):
    """
    M      : list of ground‐truth masks, |M| = M_n
    P_o    : ordered list of predicted masks, length N_max
    IoU    : 2D array of shape (M_n, P_n) with IoU[m][p]
    returns: list mIoU[1..N_max]
    """
    M_n = len(M)
    N_max = len(P_o)
    best_iou = [0.0] * M_n  # best_iou[m] = max IoU seen so far for gt‐mask m
    sum_best = 0.0  # sum of best_iou over all m
    mIoUs = [0.0] * N_max

    for N in range(1, N_max + 1):
        p_idx = P_o[N - 1]  # the Nth prediction
        # update each ground‐truth mask’s best IoU
        for m in range(M_n):
            new_iou = IoU[m][p_idx]
            if new_iou > best_iou[m]:
                sum_best += new_iou - best_iou[m]
                best_iou[m] = new_iou
        # compute mIoU for first N predictions
        mIoUs[N - 1] = sum_best / M_n

    return mIoUs


def oracle_sequence(M, P, IoU):
    """
    M   : ground‐truth list, |M|
    P   : list of all predictions, |P|
    IoU : 2D array IoU[m][p]
    returns: P_o* (list of p‐indices), and mIoU‐curve[1..|P|]
    """
    M_n = len(M)
    P_set = set(P)
    best_iou = [0.0] * M_n
    seq = []
    mIoUs = []
    sum_best = 0.0

    while P_set:
        # find p with max marginal gain
        best_p, best_gain = None, -1.0
        for p in P_set:
            gain = 0.0
            for m in range(M_n):
                delta = IoU[m][p] - best_iou[m]
                if delta > 0:
                    gain += delta
            if gain > best_gain:
                best_gain, best_p = gain, p

        # pick it
        P_set.remove(best_p)
        seq.append(best_p)
        # update best_iou & sum_best
        for m in range(M_n):
            if IoU[m][best_p] > best_iou[m]:
                sum_best += IoU[m][best_p] - best_iou[m]
                best_iou[m] = IoU[m][best_p]

        # record mIoU after adding best_p
        mIoUs.append(sum_best / M_n)

    return seq, mIoUs



def get_superpixels(logits):
    st = time.time()
    # Compute the initial assignment: the channel with highest logit for each pixel
    argmaxes = np.argmax(logits, axis=0)  # shape: (H, W)

    # Assume logits is a numpy array of shape (C, H, W)
    C, H, W = logits.shape

    # Dictionary to hold seed information for each channel.
    # The seed is defined by its coordinates and the logit value at that location.
    labels = np.ones_like(argmaxes) * -1  # Initialize labels with -1

    for c in range(C):
        # Get a boolean mask for pixels where channel c is the argmax.
        mask = (argmaxes == c)
        
        if np.any(mask):
            # For pixels in region c, get the corresponding logits (for channel c)
            channel_logits = logits[c]
            
            # Find the maximum logit value in the region where channel c is the argmax.
            max_val = np.max(channel_logits[mask])
            
            # Find the coordinates (first occurrence) of this maximum in the masked region.
            # np.argwhere returns coordinates in order [row, column] (i.e. [i, j])
            max_coords = np.argwhere((channel_logits == max_val) & mask)
            if max_coords.shape[0] > 0:
                # use the first occurrence as the seed for channel c
                seed_coords = tuple(max_coords[0])
                labels[seed_coords] = c



    # Assume:
    #   labels is a numpy array of shape (H, W) where seeds are labeled
    #       with integers in 0..C-1 and unlabeled pixels are -1.
    #   logits is a numpy array of shape (C, H, W)
    #   argmaxes is a numpy array of shape (H, W)

    # Compute per-pixel optimal logit.
    max_logits = np.max(logits, axis=0)  # shape: (H, W)

    # regret_map stores the regret cost for each pixel.
    regret_map = np.full((H, W), np.inf)
    for i in range(H):
        for j in range(W):
            if labels[i, j] != -1:
                # If the pixel is labeled, its regret is 0 (it was perfectly assigned).
                regret_map[i, j] = 0

    def get_neighbors(i, j):
        # 4-connected neighborhood.
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < H and 0 <= nj < W:
                yield ni, nj

    def push_candidate(i, j, candidate_label):
        # Use the logits for the candidate contiguous label
        candidate_logit = logits[candidate_label, i, j]
        regret = max_logits[i, j] - candidate_logit
        if regret < regret_map[i, j]:  # if it lowers regret, consider it
            heapq.heappush(pq, (regret, candidate_label, (i, j)))

    # Priority queue holds elements: (cumulative_cost, candidate_label, (i, j))
    pq = []

    # Initialize with seed pixels.
    for i in range(H):
        for j in range(W):
            if labels[i, j] != -1:
                for ni, nj in get_neighbors(i, j):
                    if labels[ni, nj] == -1:
                        # Push the candidate label for the neighbor.
                        push_candidate(ni, nj, labels[i, j])

    # Main loop: process candidates by increasing regret.
    while pq:
        regret, cand_label, (i, j) = heapq.heappop(pq)
        if not (regret < regret_map[i, j]) or not (cand_label != labels[i, j]):
            # if it doesn't improve regret or change the label, skip
            continue  
        # else we have a new label with lower regret
        labels[i, j] = cand_label
        regret_map[i, j] = regret
        for ni, nj in get_neighbors(i, j):
            push_candidate(ni, nj, cand_label)

    print("superpixel generation took", time.time() - st, "seconds")
    return labels

def tune_with_superpixels(pred_masks, superpixels):
    approx_masks = []

    for i in range(len(pred_masks)):
        logit_mask = pred_masks[i]

        overlapping_superpixels = np.unique(superpixels[logit_mask])
        overlapping_superpixels_scores = []
        for overlapping_superpixel in overlapping_superpixels:
            mask = (superpixels == overlapping_superpixel)
            intersection = np.logical_and(mask, logit_mask)
            inside = np.sum(intersection)
            outside = np.sum(mask) - inside
            if outside == 0:
                overlapping_superpixels_scores.append(np.inf)
            else:
                overlapping_superpixels_scores.append(inside / outside)

        # argsort scores
        sorted_indices = reversed(np.argsort(overlapping_superpixels_scores, stable=True))
        mask_so_far = np.zeros_like(logit_mask)
        current_iou = 0
        for next_best_ind in sorted_indices:
            if overlapping_superpixels_scores[next_best_ind] < current_iou:
                break
            sp_label = overlapping_superpixels[next_best_ind]
            sp_mask = (superpixels == sp_label)
            mask_so_far = np.logical_or(mask_so_far, sp_mask)
            intersection = np.logical_and(mask_so_far, logit_mask)
            union = np.logical_or(mask_so_far, logit_mask)
            current_iou = np.sum(intersection) / np.sum(union)

        approx_masks.append(mask_so_far)
        print('logit ind:', i, 'iou:', current_iou, end='\r')
    approx_masks = np.array(approx_masks)
    return approx_masks



def evaluate(logits, scores, gt, nms_thresh=1.0, superpixels=None):
    st = time.time()
    # Prepare gt
    values = np.unique(gt)
    gt_masks = np.array([gt == v for v in values]).astype(np.float32)
    gt_torch = torch.from_numpy(gt_masks).cuda()
    # NMS
    pred_masks_torch = (torch.from_numpy(logits)>0).to(gt_torch.device)
    _, kept_indices = nms_masks_gpu(pred_masks_torch, scores, nms_thresh)
    scores = scores[kept_indices]
    logits = logits[kept_indices]
    # Sort
    order = torch.argsort(torch.from_numpy(scores), descending=True)
    logits = torch.from_numpy(logits[order]).cuda()
    logits = torch.nn.functional.interpolate(logits.unsqueeze(1), size=(gt_torch.shape[1], gt_torch.shape[2]), mode="bilinear").squeeze(1)
    # Compute IoU
    pred_masks = (logits>0).to(gt_torch.device)
    if superpixels:
        pred_masks = tune_with_superpixels(pred_masks, superpixels)
    iou_mat = iou_one_loop(gt_torch, pred_masks).cpu().numpy()
    M = list(range(gt_masks.shape[0]))
    P_o = list(range(pred_masks.shape[0]))  # model output order
    miou_curve = np.array(incremental_mIoU(M, P_o, iou_mat))
    oracle_miou_curve = np.array(oracle_sequence(M, P_o, iou_mat)[1])
    print("evaluation took", time.time() - st, "seconds")
    return miou_curve, oracle_miou_curve
 




def main():
    # get predictions and gt
    logits, predicted_ious, stability_scores, points = get_all_logits()
    gt = skimage.measure.label(plt.imread('ours/images/texmos3.s512.tiff')+1)
    i = 0
    Image.fromarray((per_min_max(logits[i])*255).astype(np.uint8)).save(f"ours/tmp/logits_{i}.png")

    # get superpixels
    labels = get_superpixels(logits)
    Image.fromarray((per_min_max(labels)*(256*256-1)).astype(np.uint16)).save(f"ours/tmp/superpixels.png")

    # evaluate
    plt.figure()
    ax = plt.gca()
    # for iou_thresh in [0.25, 0.5, 0.75, 0.9, 0.95, 1.]:
    for iou_thresh in [0.9]:#, 0.95, 1.]:
        miou_curve, oracle_curve = evaluate(logits, predicted_ious, gt, nms_thresh=iou_thresh)
        color = plt.cm.viridis(iou_thresh)
        ax.plot(np.array(np.arange(1, len(miou_curve)+1)), miou_curve, label="mIoU (NMS@{}) predIoU".format(iou_thresh), color=color)
        ax.plot(np.array(np.arange(1, len(oracle_curve)+1)), oracle_curve, label="oracle (NMS@{}) predIoU".format(iou_thresh), linestyle="--", color=color)
    # for iou_thresh in [0.25, 0.5, 0.75, 0.9, 0.95, 1.]:
    for iou_thresh in [0.9]:#, 0.95, 1.]:
        miou_curve, oracle_curve = evaluate(logits, stability_scores, gt, nms_thresh=iou_thresh)
        color = plt.cm.inferno(iou_thresh)
        ax.plot(np.array(np.arange(1, len(miou_curve)+1)), miou_curve, label="mIoU (NMS@{}) stability".format(iou_thresh), color=color)
        ax.plot(np.array(np.arange(1, len(oracle_curve)+1)), oracle_curve, label="oracle (NMS@{}) stability".format(iou_thresh), linestyle="--", color=color)
    for iou_thresh in [0.9]:#, 0.95, 1.]:
    # for iou_thresh in [0.25, 0.5, 0.75, 0.9, 0.95, 1.]:
        miou_curve, oracle_curve = evaluate(logits, (stability_scores+predicted_ious)/2, gt, nms_thresh=iou_thresh)
        color = plt.cm.jet(iou_thresh)
        ax.plot(np.array(np.arange(1, len(miou_curve)+1)), miou_curve, label="mIoU (NMS@{}) avg".format(iou_thresh), color=color)
        ax.plot(np.array(np.arange(1, len(oracle_curve)+1)), oracle_curve, label="oracle (NMS@{}) avg".format(iou_thresh), linestyle="--", color=color)
    # superpixel variant
    miou_curve, oracle_curve = evaluate(logits, (stability_scores+predicted_ious)/2, gt, nms_thresh=iou_thresh, superpixels=labels)
    color = plt.cm.jet(iou_thresh)
    ax.plot(np.array(np.arange(1, len(miou_curve)+1)), miou_curve, label="mIoU (NMS@{}) avg spix".format(iou_thresh), color=color)
    ax.plot(np.array(np.arange(1, len(oracle_curve)+1)), oracle_curve, label="oracle (NMS@{}) avg spix".format(iou_thresh), linestyle="--", color=color)

    ax.set_xscale("log")
    ax.set_xticks([1,4,16,64,256,1024])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("# of masks")
    ax.set_ylabel("mIoU")
    ax.legend(loc="lower center", fontsize=8, bbox_to_anchor=(0.5, 1.02))
    plt.savefig("ours/tmp/miou_curve.png", bbox_inches="tight")
    plt.close()




main()
