import cv2
from copy import deepcopy
import torch
from PIL import Image
import shutil
from collections import deque
import os
from heapq import heappop, heappush
import numpy as np
from scipy import ndimage

S = 256
S2 = S * S


def load_logits(imgname):
    """Loads the logits generated using AutomaticMaskGenerator with SAM2.
    These are floats of shape (L, 256, 256)=(L, S, S) that should be saved in `logits/{imgname}_logits.npy`
    """
    print("trying to load npy file...", end="\r")
    logits = np.load(f"logits/{imgname}_logits.npy")
    print("- loaded npy file!")
    return logits


def make_c1(data):
    """
    Adjust the logits so that, when thresholded at 0, the obtained mask has only one 4-connected component. To do this, we enforce that there should always exist a monotonically non-increasing path from the maximum to any other pixel. The only way for this to happen is to have logits outside the connected component that contains the max to be negative, else they are positive and there is an all positive path that connects them to the max, which contradicts that they are outside the max connected component.
    """
    adjusted_map = data.copy()
    total_size = data.size
    visited = np.zeros_like(data, dtype=bool)

    for channel in range(data.shape[0]):
        print(f"making c1... channel {channel}", end="\r")
        max_index = np.argmax(adjusted_map[channel])
        seed_row, seed_col = divmod(max_index, S)
        seed_value = adjusted_map[channel, seed_row, seed_col]

        heap = [(-seed_value, seed_row, seed_col)]
        visited[channel, seed_row, seed_col] = True

        while heap:
            neg_val, row, col = heappop(heap)
            current_val = -neg_val
            deltas = [(1, 0), (-1, 0), (0, 1), (0, -1)]

            for d_row, d_col in deltas:
                nrow, ncol = row + d_row, col + d_col
                if 0 <= nrow < S and 0 <= ncol < S and not visited[channel, nrow, ncol]:
                    neighbor_value = adjusted_map[channel, nrow, ncol]
                    # To adjust the map we set each pixel to be the maximum value such that there is a monotonically non-increasing path to the max to the pixel. I think this changes the logit map the minimum possible in order to ensure one 4-connected component while thresholding.
                    if neighbor_value > current_val:
                        adjusted_map[channel, nrow, ncol] = current_val
                    heappush(heap, (-adjusted_map[channel, nrow, ncol], nrow, ncol))
                    visited[channel, nrow, ncol] = True

    print("- done making c1!")
    return adjusted_map


def calculate_stability_score(
    logits: torch.Tensor, mask_threshold: float, threshold_offset: float
) -> torch.Tensor:
    """
    Computes the stability score for a batch of masks. The stability
    score is the IoU between the binary masks obtained by thresholding
    the predicted mask logits at high and low values.
    """
    # One mask is always contained inside the other.
    # Save memory by preventing unnecessary cast to torch.int64
    intersections = (logits > (mask_threshold + threshold_offset)).sum(-1).sum(-1)
    unions = (logits > (mask_threshold - threshold_offset)).sum(-1).sum(-1)
    return intersections / unions


def nms_masks(masks, scores, iou_threshold=0.97):
    """
    Perform Non-Maximum Suppression (NMS) on masks based on their scores.

    Args:
        masks (np.ndarray): Array of masks with shape (M, S, S).
        scores (np.ndarray): Array of scores with shape (M,).
        iou_threshold (float): IoU threshold for suppression.

    Returns:
        np.ndarray: Array of masks after NMS with shape (K, S, S).
    """
    M, S, _ = masks.shape
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Convert masks and scores to torch tensors
    mf = torch.from_numpy(masks.reshape(M, -1)).float().to(device)
    scores = torch.from_numpy(scores).to(device)

    # Sort masks by scores in descending order
    sorted_indices = torch.argsort(scores, descending=True)
    mf = mf[sorted_indices]

    keep = []
    original_indices = []
    while mf.shape[0] > 0:
        # Select the mask with the highest score
        current_mask = mf[0]
        keep.append(current_mask)
        original_indices.append(sorted_indices[0].item())

        if mf.shape[0] == 1:
            break

        # Compute IoU of the current mask with the rest
        inters = torch.matmul(mf[1:], current_mask)
        areas = mf[1:].sum(dim=1) + current_mask.sum() - inters
        ious = inters / areas

        # Keep masks with IoU less than the threshold
        mask_to_keep = ious < iou_threshold
        mf = mf[1:][mask_to_keep]
        sorted_indices = sorted_indices[1:][mask_to_keep]

    # Convert kept masks back to original shape
    kept_masks = torch.stack(keep).reshape(-1, S, S).cpu().numpy()
    return kept_masks > 0, original_indices



def clean_dots(labels, min_size=16):
    """
    Cleans the labeled image by removing regions with an area smaller than min_size.
    It relabels the affected pixels based on their four-connected neighbors.

    Parameters:
    - labels (np.ndarray): 2D array of labeled regions. Labels should be positive integers.
    - min_size (int): Minimum area a region must have to be retained.

    Returns:
    - new_labels (np.ndarray): Cleaned labeled image with small regions removed.
    """
    assert (
        labels != 0
    ).all(), "Labels should not contain zero in any case, as the labeling starts at 1."

    unique_labels, label_counts = np.unique(labels, return_counts=True)

    # Identify small regions (labels with count < min_size)
    small_regions = unique_labels[label_counts < min_size]
    if not len(small_regions):
        return labels

    # Clean small regions by examining their surrounding pixels
    new_labels = labels.copy()

    for region_label in small_regions:
        label_mask = labels == region_label
        neighbor_mask = np.zeros((S, S), dtype=bool)
        # Shift up
        neighbor_mask |= np.pad(
            label_mask[:-1, :], ((1, 0), (0, 0)), mode="constant", constant_values=False
        )
        # Shift down
        neighbor_mask |= np.pad(
            label_mask[1:, :], ((0, 1), (0, 0)), mode="constant", constant_values=False
        )
        # Shift left
        neighbor_mask |= np.pad(
            label_mask[:, :-1], ((0, 0), (1, 0)), mode="constant", constant_values=False
        )
        # Shift right
        neighbor_mask |= np.pad(
            label_mask[:, 1:], ((0, 0), (0, 1)), mode="constant", constant_values=False
        )

        # Exclude the label_mask itself
        neighbor_mask &= ~label_mask

        unique_neighbor_labels = np.unique(labels[neighbor_mask])
        if (
            len(unique_neighbor_labels) == 1
        ):  # if the neighbors are of the same label inpaint
            new_labels[label_mask] = unique_neighbor_labels[0]
        # else skip, as coloring is not obvious

    # Now, relabel to have consecutive labels starting from 1 (optional)
    unique_labels = np.unique(new_labels)
    for idx, label in enumerate(unique_labels, start=1):
        new_labels[new_labels == label] = idx

    print("- done removing small regions!")
    return new_labels


def create_rag(labels):
    """Create a region adjacency graph (RAG) where nodes are the regions determined by each one of the labels and the edges represent contiguity of regions."""
    print("creating rag...", end="\r")
    neighbor_offsets = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    rag = {int(label): set() for label in np.unique(labels)}

    for row in range(S):
        for col in range(S):
            current_label = int(labels[row, col])
            assert current_label != 0, "label is 0, which should not happen"
            for d_row, d_col in neighbor_offsets:
                nrow, ncol = row + d_row, col + d_col
                if 0 <= nrow < S and 0 <= ncol < S:
                    neighbor_label = labels[nrow, ncol]
                    if neighbor_label != current_label:
                        neighbor_label = int(neighbor_label)
                        rag[current_label].add(neighbor_label)
                        rag[neighbor_label].add(current_label)

    print("- done creating rag!")
    return rag


def merge_high_iou_pairs_fast(masks, merge_iou=0.97):
    """Merge pairs of masks that have high IoU. The default value is 0.97 because it's very high, but I fear that lower might merge masks that aren't actually duplicates."""
    M = masks.shape[0]
    mf = (
        torch.from_numpy(masks.reshape(M, -1))
        .to(torch.float)
        .to("cuda" if torch.cuda.is_available() else "cpu")
    )
    while True:
        M = mf.shape[0]
        print("masks remaining:", M, end="\r")
        # get ious
        inters = torch.mm(mf, mf.T)
        areas = mf.sum(dim=1)
        unions = areas[:, None] + areas[None, :] - inters
        ious = inters / unions
        ious.fill_diagonal_(0)
        # merge highest
        maxind = torch.argmax(ious)
        r, c = maxind // M, maxind % M
        maxval = ious[r, c]
        if maxval < merge_iou:
            break
        new_mask = torch.logical_or(mf[r], mf[c])
        mf[r] = new_mask
        mf = torch.cat((mf[:c], mf[c + 1 :]))  # remove second mask
    masks = mf.reshape(M, S, S).cpu().numpy()
    return masks


def compute_areas(labels):
    """Count how many pixels each label region has. It's never zero."""
    print("computing areas...", end="\r")
    unique, counts = np.unique(labels, return_counts=True)
    areas = dict(zip(unique.tolist(), counts.tolist()))
    print("- done computing areas!")
    return areas


def reference_masks_as_sets(masks, labels):
    """Define the reference masks `masks` as sets of labeled regions. The labeled regions are non-overlapping, while the masks might be."""
    print("getting reference masks as sets...", end="\r")
    reference_masks = []
    n_channels = masks.shape[0]
    for channel in range(n_channels):
        print("getting reference masks as sets...", channel, end="\r")
        mask_as_set = np.unique(labels[masks[channel]])
        reference_masks.append(mask_as_set)
    print("- done getting reference masks as sets!")
    return reference_masks


def map_labels_to_refs(refs, max_label):
    """
    Creates a mapping from labels to the refs that contain them.

    Parameters:
    - refs: list of sets of labels. Usually obtained from `reference_masks_as_sets`.
    - max_label: Assume labels go from 1 to `max_label` inclusive.

    Returns:
    - dict mapping each label to a set of indices of refs that contain that label.
    """
    label_to_refs = {}
    for label in range(1, max_label + 1):
        label_to_refs[label] = set()
        for i, ref in enumerate(refs):
            if label in ref:
                label_to_refs[label].add(i)
    return label_to_refs


class TreeNode:
    def __init__(
        self, leaves, label, area, inside, outside, inside_over_outside, children=None
    ):
        self.leaves = leaves
        self.label = label
        self.area = area  # area of the node / mask
        self.inside = inside  # area of the node inside each reference mask
        self.outside = outside  # area of the node outside each reference mask
        self.inside_over_outside = inside_over_outside
        self.children = children

    def __lt__(self, other):
        return self.label < other.label


def get_possible_merges(rag, areas, short=False):
    """Possible merges are those allowed by the RAG.
    If `short` then we take shortcuts (see below)."""
    print("getting possible merges...", end="\r")
    if not short:
        possible_merges = []
        for label, neighbors in rag.items():
            for neighbor in neighbors:
                if label < neighbor:  # avoid duplicates
                    possible_merges.append((label, neighbor))
        print("- done getting possible merges! got", len(possible_merges), end="\r")
        return possible_merges
    else:
        return get_possible_merges_short(rag, areas)


def get_possible_merges_short(rag, areas):
    # returns possible merges of 1px regions with each other, with other regions, or all, whatever is non-empty
    # we focus on small regions first as they are the ones that usually limit the potential score the least (if merged correctly)
    print("getting possible merges...", end="\r")
    # first try with 1px 1px pairs
    possible_merges_1px1px, possible_merges_1px, possible_merges = [], [], []
    lenap1 = len(areas) + 1
    for label, neighbors in rag.items():
        if label < lenap1:
            area1 = areas[label]
            is_1px = area1 == 1
        else:
            is_1px = False
        for neighbor in neighbors:
            if label < neighbor:  # avoid duplicates
                if is_1px:
                    possible_merges_1px.append((label, neighbor))
                if neighbor < lenap1:  # if the neighbor is not a new region
                    area2 = areas[neighbor]
                    if area2 == 1:
                        if is_1px:
                            possible_merges_1px1px.append((label, neighbor))
                        else:
                            possible_merges_1px.append((label, neighbor))
                possible_merges.append((label, neighbor))
    if len(possible_merges_1px1px) > 1:
        print(
            f"- done getting possible merges! got {len(possible_merges_1px1px)} 1px1px"
        )
        return possible_merges_1px1px
    elif len(possible_merges_1px) > 1:
        print(f"- done getting possible merges! got {len(possible_merges_1px)} 1px")
        return possible_merges_1px
    else:
        print(f"- done getting possible merges! got {len(possible_merges)}")
        return possible_merges


def compute_potential_iou(orphans, reference_mask_index, ref_areas):
    """Compute the max IoU achievable by any node in a binary tree built from the orphan masks `orphans` and the reference mask at the provided index. To do this we find the combination of orphans that achieves the best IoU. This means including overlapping regions in order until they do not add to the IoU anymore, which happens when the inside/outside ratio being smaller than the current IoU. `ref_areas` are the areas of each reference mask."""
    overlapping_regions = [
        node for node in orphans.values() if node.inside[reference_mask_index] > 0
    ]
    overlapping_regions.sort(
        key=lambda x: x.inside_over_outside[reference_mask_index], reverse=True
    )
    I_current = 0
    U_current = ref_areas[reference_mask_index]
    IoU_current = 0
    for node in overlapping_regions:
        if node.inside_over_outside[reference_mask_index] > IoU_current:
            I_current += node.inside[reference_mask_index]
            U_current += node.outside[reference_mask_index]
            IoU_current = I_current / U_current
        else:
            break
    return IoU_current


def compute_potential_ious_after_merge(
    max_ious,
    max_ious_achieved,
    # updated_ref_ious,  # not sure about this
    label_to_refs,
    merge_pair,
    orphans,
    ref_areas,
):
    """This function computes the maximum IoU achievable for each mask, independently, given that we have merged `merge_pair`. This way we evaluate the impact that `merge_pair` could have in our final IoU to the references.
    Note that the maximum achievable is always greater than the maximum achieved so far.
    """
    max_ious_merge = max_ious.copy()
    label1, label2 = merge_pair
    node1, node2 = orphans[label1], orphans[label2]

    # compute potential iou after merging
    # this involves simulating the merge and updating the iou for the affected refs
    new_label = max(orphans.keys()) + 1
    new_area = node1.area + node2.area
    new_inside = node1.inside + node2.inside
    new_outside = node1.outside + node2.outside
    new_inside_over_outside = np.divide(
        new_inside,
        new_outside,
        out=np.full_like(new_inside, float("inf")),
        where=new_outside != 0,
    )
    new_orphan = TreeNode(
        node1.leaves.union(node2.leaves),
        new_label,
        new_area,
        new_inside,
        new_outside,
        new_inside_over_outside,
        children=[node1, node2],
    )
    new_orphans = {
        k: TreeNode(
            n.leaves, n.label, n.area, n.inside, n.outside, n.inside_over_outside
        )
        for k, n in orphans.items()
    }

    new_orphans[new_label] = new_orphan
    del new_orphans[label1]
    del new_orphans[label2]

    affected_refs = label_to_refs[label1].union(label_to_refs[label2])
    to_reconsider = affected_refs
    # to_reconsider = affected_refs.intersection(  # not sure about this
    #     updated_ref_ious
    # )  # they were either affected by the previous update or will be by this merge
    for reference_index in to_reconsider:  # it was updated in prev step
        max_ious_merge[reference_index] = compute_potential_iou(
            new_orphans, reference_index, ref_areas
        )
    max_ious_merge = np.maximum(max_ious_merge, max_ious_achieved)
    return max_ious_merge, affected_refs


def merge_labels(
    pair_to_merge,
    orphans,
    label_to_refs,
    ref_areas,
    max_label,
    max_ious,
    max_ious_achieved,
    max_ious_per_merge,
    current_score,
    rag,
    labels_img=None,
    imgcounter=None,
):
    """
    Merges two labels in the RAG and updates related data structures.

    Parameters:
    - pair_to_merge: tuple of labels to merge.
    - orphans: dict of current tree nodes.
    - label_to_refs: dict mapping labels to reference regions.
    - max_label: current maximum label value.
    - max_ious: array of maximum IoUs achievable for each reference.
    - max_ious_achieved: array of maximum IoUs already achieved for each reference.
    - max_ious_per_merge: dict of potential max IoUs for merges.
    - current_score: current cumulative IoU score.
    - rag: Region Adjacency Graph.
    - labels_img: (optional) image of labels.
    - imgcounter: (optional) counter for image saving.

    Returns:
    - max_label: updated maximum label value.
    - max_ious: updated array of maximum IoUs.
    - current_score: updated cumulative IoU score.
    - imgcounter: updated image counter (if labels_img is provided).
    """
    node1, node2 = orphans[pair_to_merge[0]], orphans[pair_to_merge[1]]
    new_inside = node1.inside + node2.inside
    new_outside = node1.outside + node2.outside
    new_inside_over_outside = np.divide(
        new_inside,
        new_outside,
        out=np.full_like(new_inside, float("inf")),
        where=new_outside != 0,
    )
    new_node = TreeNode(
        node1.leaves.union(node2.leaves),
        max_label + 1,
        node1.area + node2.area,
        new_inside,
        new_outside,
        new_inside_over_outside,
        children=[node1, node2],
    )
    new_node.iou = new_node.inside / (new_node.outside + ref_areas)
    del orphans[pair_to_merge[0]]
    del orphans[pair_to_merge[1]]
    orphans[new_node.label] = new_node

    # Update max_label
    max_label += 1

    # Update max_ious. Achieved ious is updated here, the other one is precomputed and passed to this function.
    # the achieved ious now has another candidate node
    max_ious_achieved = np.maximum(max_ious_achieved, new_node.iou)
    # the max iou achievable is the one that follows the merge.
    # note that the max_ious_per_merge[pair_to_merge] already considers the max_ious_achieved,
    # including the one achieved by the new node created by the merge (as it was considered in the potential after the merge)
    max_ious = np.minimum(max_ious, max_ious_per_merge[pair_to_merge])

    # Update current_score
    current_score = np.sum(max_ious)

    # Update label_to_refs
    label_to_refs[new_node.label] = label_to_refs[pair_to_merge[0]].union(
        label_to_refs[pair_to_merge[1]]
    )

    # Update rag
    neighbors1 = rag.pop(pair_to_merge[0])
    neighbors2 = rag.pop(pair_to_merge[1])
    new_neighbors = neighbors1.union(neighbors2) - {pair_to_merge[0], pair_to_merge[1]}
    rag[new_node.label] = new_neighbors
    for neighbor in new_neighbors:
        rag[neighbor].add(new_node.label)
        rag[neighbor].discard(pair_to_merge[0])
        rag[neighbor].discard(pair_to_merge[1])

    # Update labels_img if provided
    if labels_img is not None and imgcounter is not None:
        labels_img[
            (labels_img == pair_to_merge[0]) | (labels_img == pair_to_merge[1])
        ] = new_node.label
        imgcounter += 1

    return max_label, max_ious, max_ious_achieved, current_score, imgcounter


def build_tree(
    rag,
    areas,
    refs,
    ref_areas,
    labels_img=None,
    merges_to_try=128,
    score_change_threshold=0.01,
    short_merge_find=False,
    dstdir=".",
):
    """
    Processes a Region Adjacency Graph (RAG), area values, and a list of label sets (refs). The objective is to binary tree that has the maximum possible fitness to the reference masks `refs`.

    Parameters:
    - rag: dict mapping labels to sets of neighboring labels.
    - areas: dict mapping labels to area values.
    - refs: list of sets of labels, typically representing regions of interest.
    - remove_below: area threshold below which regions will be merged.

    Returns:
    - result: root TreeNode of the built tree.
    """
    # create dir for saving resulting labels after merges
    os.makedirs(f"{dstdir}/labels", exist_ok=True)
    # make a deep copies of rag and labels_img
    rag = deepcopy(rag)
    labels_img = labels_img.copy() if labels_img is not None else None
    # initialize a few variables
    max_label = max(rag.keys())
    label_to_refs = map_labels_to_refs(
        refs, max_label
    )  # this one maps the original labels (not the new ones) to the references that contain them
    max_ious = np.ones(
        len(refs)
    )  # this is the maximum iou for each mask achievable by merging the current orphans
    current_score = np.sum(max_ious)
    # Initialize orphans, which are the nodes that can be merged (they hold the tree)
    orphans = {}
    for label in range(1, max_label + 1):
        area = areas[label]
        inside = np.empty(len(refs))
        outside = np.empty(len(refs))
        inside_over_outside = np.empty(len(refs))
        for r_idx in range(len(refs)):
            refm = refs[r_idx]
            if label not in refm:
                inside[r_idx] = 0
                outside[r_idx] = area
                inside_over_outside[r_idx] = 0
            else:
                inside[r_idx] = area
                outside[r_idx] = 0
                inside_over_outside[r_idx] = float("inf")
        node = TreeNode(
            {label}, label, area, inside, outside, inside_over_outside, children=[]
        )
        orphans[label] = (
            node  # the orphans are initially the region labels (i.e. atomic regions)
        )

        # compute iou for each mask achieved by the atomic regions
        node.iou = node.inside / (node.outside + ref_areas)

    # compute the max iou already achieved for each mask
    max_ious_achieved = np.zeros(len(refs))
    for node in orphans.values():
        max_ious_achieved = np.maximum(max_ious_achieved, node.iou)

    # Remove small regions by merging them into the most similar neighbor
    imgcounter = 0
    possible_merges = get_possible_merges(rag, areas, short=short_merge_find)

    if labels_img is not None:
        Image.fromarray(labels_img.astype(np.uint16)).save(
            f"{dstdir}/labels/labels_img_{imgcounter}.png"
        )

    while len(possible_merges) > 0:
        max_ious_for_merge = dict()
        score_per_merge = dict()

        for merge_pair in possible_merges[:merges_to_try]:
            print(len(possible_merges), merge_pair, " " * 30, end="\r")
            max_ious_for_merge[merge_pair], _ = compute_potential_ious_after_merge(
                max_ious,
                max_ious_achieved,
                label_to_refs,
                merge_pair,
                orphans,
                ref_areas,
            )
            score_per_merge[merge_pair] = np.sum(max_ious_for_merge[merge_pair])
            if (
                current_score - score_per_merge[merge_pair]
            ) / current_score < score_change_threshold / len(refs):
                break
        pair_to_merge = max(score_per_merge, key=lambda k: score_per_merge[k])
        # print("max iou", max_ious_per_merge[pair_to_merge].sum())

        max_label, max_ious, max_ious_achieved, current_score, imgcounter = (
            merge_labels(
                pair_to_merge,
                orphans,
                label_to_refs,
                ref_areas,
                max_label,
                max_ious,
                max_ious_achieved,
                max_ious_for_merge,
                current_score,
                rag,
                labels_img=labels_img,
                imgcounter=imgcounter if labels_img is not None else None,
            )
        )

        possible_merges = get_possible_merges(rag, areas, short=short_merge_find)

        if labels_img is not None:
            Image.fromarray(labels_img.astype(np.uint16)).save(
                f"{dstdir}/labels/labels_img_{imgcounter}.png"
            )
            # print(f"Saved merged labels image {imgcounter}")

        # print("merge", pair_to_merge)

    return list(orphans.values())[0]


def get_relevant_nodes(tree_root, ref_areas):
    """Gets those nodes that are involved in the fitness function, i.e. the nodes that are the best match to any reference mask."""
    print("getting relevant nodes...", end="\r")
    # we explore the tree to find the nodes that match the ref the most
    heap = [tree_root]
    best_iou = np.zeros(len(ref_areas))
    best_iou_nodes = np.array([None] * len(ref_areas))
    while heap:
        node = heap.pop()
        intersection = node.inside
        union = node.outside + ref_areas
        node.iou = intersection / union
        best_iou_nodes[node.iou > best_iou] = node
        best_iou = np.maximum(node.iou, best_iou)
        if node.children:
            heap.extend(node.children)
    relevant_nodes = np.unique(best_iou_nodes)
    relevant_nodes = relevant_nodes[relevant_nodes != tree_root]
    relevant_nodes = np.append(relevant_nodes, tree_root)
    assert all([n is not None for n in best_iou_nodes])
    assert relevant_nodes[-1] == tree_root
    print("- done getting relevant nodes!          ")
    return best_iou, relevant_nodes


class FinalTreeNode:
    def __init__(self, leaves, label):
        self.leaves = set(leaves)  # Set of leaf regions
        self.children = []  # List of child nodes
        self.parent = None  # Parent node
        self.label = label  # Label of the node


def find_connected_components(leaves, rag):
    """Groups the label regions in `leaves` in connected components."""
    connected_components = []
    leaves = set(leaves)
    while leaves:
        start_leaf = leaves.pop()
        component = {start_leaf}
        stack = [start_leaf]
        while stack:
            leaf = stack.pop()
            neighbors = rag[int(leaf)]
            for neighbor in neighbors:
                if neighbor in leaves:
                    leaves.remove(neighbor)
                    component.add(neighbor)
                    stack.append(neighbor)
        connected_components.append(component)
    return connected_components


def build_new_tree(nodes, rag):
    """Build a new tree, not necessarily binary, given the relevant nodes and the RAG. To build the tree we assign to each node a parent which is the smallest node that contains it. The tree should be complete, thus we add complements as needed."""
    # Sort nodes based on the number of leaves (smallest to largest)
    nodes = sorted(nodes, key=lambda n: len(n.leaves))
    nodes = [FinalTreeNode(leaves=n.leaves, label=n.label) for n in nodes]
    max_label = max([n.label for n in nodes])

    # Assign Parents to Nodes
    for node in nodes:
        potential_parents = [
            n for n in nodes if n.leaves.issuperset(node.leaves) and n != node
        ]
        if potential_parents:
            # Select the parent with the minimal number of leaves
            parent = min(potential_parents, key=lambda n: len(n.leaves))
            node.parent = parent
            parent.children.append(node)
        else:
            # Node is a root
            node.parent = None

    # Ensure Parents are Unions of Children
    parent_nodes = [node for node in nodes if node.children]

    for parent in parent_nodes:
        # Compute union of children's leaves
        children_leaves_union = set()
        for child in parent.children:
            children_leaves_union.update(child.leaves)

        # Identify missing leaves
        missing_leaves = parent.leaves - children_leaves_union

        if missing_leaves:
            # Find connected components in missing_leaves using RAG
            connected_components = find_connected_components(missing_leaves, rag)

            for component in connected_components:
                # Create new complement node
                complement_node = FinalTreeNode(leaves=component, label=max_label + 1)
                max_label += 1
                complement_node.parent = parent
                parent.children.append(complement_node)

    # Return the list of root nodes
    roots = [node for node in nodes if node.parent is None]
    assert len(roots) == 1, "There should be exactly one root node"
    return roots


def to_0255uint8(img):
    amin, amax = img.min(), img.max()
    if amin == amax:
        return np.zeros_like(img, dtype=np.uint8)
    return ((img - amin) / (amax - amin) * 255).astype(np.uint8)


####################


def traverse_tree(root):
    """Traverse the tree and return a list of all reachable nodes."""
    nodes = []
    stack = [root]
    while stack:
        node = stack.pop()
        nodes.append(node)
        stack.extend(node.children)
    return nodes


def optimize_tree_merges(
    tree_root, rag, areas, refs, ref_areas, lambda_param, target_node_count, labels_img=None
):
    """
    Optimize the tree by performing sibling merges that reduce the score the least.

    Parameters:
    - tree_root: The root node of the tree (FinalTreeNode).
    - rag: Region Adjacency Graph (dict mapping label to set of neighboring labels).
    - areas: dict mapping labels to area values.
    - refs: List of reference masks as sets of labels.
    - ref_areas: Array of areas for each reference mask.
    - lambda_param: Threshold for allowed score reduction.

    Returns:
    - The optimized tree root, the list of total scores at each step, and the list of lambda reductions.
    """
    # Convert refs to list of sets if they aren't already
    refs = [set(ref) for ref in refs]
    num_refs = len(refs)

    # Collect all nodes in the tree
    all_nodes = traverse_tree(tree_root)
    initial_node_count = len(all_nodes)
    current_node_count = initial_node_count
    if target_node_count is not None and target_node_count >= initial_node_count:
        print(
            f"Target node count {target_node_count} is greater than or equal to initial node count {initial_node_count}. No merges will be performed."
        )
        return tree_root, [], []

    # Initialize inside and outside for all nodes
    for node in all_nodes:
        node.inside = np.zeros(num_refs)
        node.outside = np.zeros(num_refs)
        for m_idx, ref in enumerate(refs):
            inside_labels = node.leaves & ref
            outside_labels = node.leaves - ref
            node.inside[m_idx] = sum(areas[label] for label in inside_labels)
            node.outside[m_idx] = sum(areas[label] for label in outside_labels)

    # Compute initial IoUs and total score
    best_iou = np.zeros(num_refs)
    for node in all_nodes:
        node.iou = node.inside / (node.outside + ref_areas)
        best_iou = np.maximum(best_iou, node.iou)
    current_total_score = np.sum(best_iou)
    total_scores = [current_total_score]
    lambdas = []
    max_label = max(n.label for n in all_nodes)

    while True:
        # Check if the number of nodes is below the target
        if target_node_count is not None and current_node_count <= target_node_count:
            print(
                f"Reached target node count: {current_node_count} <= {target_node_count}. Stopping merge process."
            )
            break
        # Identify all possible merges
        possible_merges = []
        parent_nodes = [node for node in all_nodes if node.children]
        for parent in parent_nodes:
            children = parent.children
            if len(children) < 2:
                continue  # Need at least two children to merge

            # Generate all unique pairs of children
            for i in range(len(children)):
                for j in range(i + 1, len(children)):
                    n1 = children[i]
                    n2 = children[j]
                    if are_contiguous_in_rag(n1, n2, rag):
                        possible_merges.append((n1, n2, parent))

        if not possible_merges:
            print("No more possible merges.")
            break  # No more possible merges

        merges_with_reduction = []

        # Evaluate all possible merges and find the one with the least score reduction
        best_merge = None
        least_reduction = float("inf")

        for merge_ind, (n1, n2, parent) in enumerate(possible_merges):
            print(
                f"Evaluating merge {merge_ind + 1}/{len(possible_merges)}: "
                f"Nodes {n1.label} & {n2.label} under Parent {parent.label}",
                end="\r",
            )

            had_many_children = len(parent.children) > 2
            if had_many_children:
                merged_node = FinalTreeNode(n1.leaves.union(n2.leaves), label=-1)
                merged_node.inside = n1.inside + n2.inside
                merged_node.outside = n1.outside + n2.outside
                merged_node.iou = merged_node.inside / (merged_node.outside + ref_areas)
            merged_node_list = [merged_node] if had_many_children else []
            temp_all_nodes = [
                n for n in all_nodes if not n in [n1, n2]
            ] + merged_node_list

            temp_best_iou = np.zeros(num_refs)
            for temp_node in temp_all_nodes:
                temp_best_iou = np.maximum(temp_best_iou, temp_node.iou)
            reduction = current_total_score - np.sum(temp_best_iou)

            print(f"Merge {merge_ind + 1}: Reduction in score = {reduction}", end="\r")
            reduction = reduction / 2 if not had_many_children else reduction
            merges_with_reduction.append((reduction, (n1, n2, parent)))

            if reduction < least_reduction:
                least_reduction = reduction
                best_merge = (n1, n2, parent)

        if best_merge is None:
            print("No valid merge found.")
            break

        print(
            f"Selected Merge: Nodes {best_merge[0].label} & {best_merge[1].label} under Parent {best_merge[2].label} "
            f"with Reduction {least_reduction}"
        )

        merges_with_reduction.sort(key=lambda x: x[0])

        if lambda_param is not None and least_reduction > lambda_param:
            print(
                f"Reduction {least_reduction} exceeds lambda_param {lambda_param}. Stopping merge process."
            )
            break  # Stop merging if reduction exceeds threshold

        # Apply the best merge to the actual tree
        n1, n2, parent = best_merge

        if len(parent.children) > 2:
            # Create a new merged node
            merged_node = FinalTreeNode(
                leaves=n1.leaves.union(n2.leaves), label=max_label + 1
            )
            max_label += 1
            merged_node.parent = parent
            parent.children = [
                child for child in parent.children if child not in [n1, n2]
            ] + [merged_node]
            merged_node.inside = n1.inside + n2.inside
            merged_node.outside = n1.outside + n2.outside
            # This merge reduces the node count by 1 (remove 2 nodes, add 1)
            current_node_count -= 1
        else:
            # Merge directly into the parent
            parent.leaves = n1.leaves.union(n2.leaves)
            parent.children = n1.children + n2.children
            for child in parent.children:
                child.parent = parent
            parent.inside = n1.inside + n2.inside
            parent.outside = n1.outside + n2.outside
            # This merge reduces the node count by 2 (remove 2 children nodes)
            current_node_count -= 2

        # Update IoUs after the actual merge
        all_nodes = traverse_tree(tree_root)  # Re-traverse to get the updated list

        best_iou = np.zeros(num_refs)
        for node in all_nodes:
            node.iou = node.inside / (node.outside + ref_areas)
            best_iou = np.maximum(best_iou, node.iou)
        current_total_score = np.sum(best_iou)
        total_scores.append(current_total_score)
        lambdas.append(least_reduction)

        print(f"After merge: Total score = {current_total_score}, Nodes remaining: {current_node_count}")

    print("- Done pruning tree!")
    # After all merges are done, return the final tree and the final score
    final_all_nodes = traverse_tree(tree_root)
    print(f"Number of nodes in all_nodes: {len(all_nodes)}")
    print(f"Number of nodes reachable from tree_root: {len(final_all_nodes)}")

    # Optional: Validate that all nodes are reachable
    if len(all_nodes) != len(final_all_nodes):
        print("Discrepancy detected: Some nodes are not reachable from tree_root.")

    return tree_root, total_scores, lambdas


def are_contiguous_in_rag(node1, node2, rag):
    """
    Check if two nodes are contiguous in the RAG.

    Parameters:
    - node1: First node (FinalTreeNode).
    - node2: Second node (FinalTreeNode).
    - rag: Region Adjacency Graph.

    Returns:
    - True if nodes are contiguous, False otherwise.
    """
    for label1 in node1.leaves:
        neighbors = rag.get(int(label1), set())
        if neighbors & node2.leaves:
            return True
    return False


####################


def create_images_at_multiple_levels(labels_image, roots):
    # Step 1: Assign levels to nodes
    assign_levels_to_nodes(roots)

    # Step 2: Collect nodes at each level
    nodes_per_level = get_nodes_per_level(roots)

    # Step 3: Assign unique labels to nodes
    assign_labels_to_nodes(nodes_per_level)

    # Step 4: Generate images for each level
    images_per_level = generate_images_per_level(labels_image, nodes_per_level)

    return images_per_level


def assign_levels_to_nodes(roots):
    """Assign levels to each node in the tree starting from the roots."""
    from collections import deque

    queue = deque()
    for root in roots:
        root.level = 0
        queue.append(root)
    while queue:
        node = queue.popleft()
        for child in node.children:
            child.level = node.level + 1
            queue.append(child)


def get_nodes_per_level(roots):
    """Collect nodes at each level of the tree."""
    nodes_per_level = {}
    from collections import deque

    queue = deque()
    for root in roots:
        queue.append(root)
    while queue:
        node = queue.popleft()
        level = node.level
        if level not in nodes_per_level:
            nodes_per_level[level] = []
        nodes_per_level[level].append(node)
        for child in node.children:
            queue.append(child)
    return nodes_per_level


def assign_labels_to_nodes(nodes_per_level):
    """Assign a unique label to each node."""
    label_counter = 1  # Start labeling from 1
    for level in sorted(nodes_per_level.keys()):
        for node in nodes_per_level[level]:
            node.label_in_level_img = label_counter
            label_counter += 1


def generate_images_per_level(labels_image, nodes_per_level):
    """Generate an image for each level of the tree."""
    level_images = {}
    for level in sorted(nodes_per_level.keys()):
        level_images[level] = np.zeros_like(labels_image)
        # Build a mapping from leaf labels to node labels at this level
        for node in nodes_per_level[level]:
            for leaf_label in node.leaves:
                level_images[level][
                    labels_image == leaf_label
                ] = node.label_in_level_img
    return level_images


####################
def extract_binary_masks_from_tree(root, labels_img, height=S, width=S):
    """
    Extract binary masks from the tree hierarchy.
    
    Parameters:
    - root: The root node of the tree.
    - labels_img: The original labels image.
    - height, width: Dimensions of the output masks.
    
    Returns:
    - masks: A numpy array of shape (M, H, W) where M is the number of nodes in the tree.
    - node_labels: List of node labels corresponding to each mask.
    """
    # Traverse the tree to get all nodes
    nodes = traverse_tree(root)
    
    # Create binary masks for each node
    masks = np.zeros((len(nodes), height, width), dtype=bool)
    node_labels = []
    
    for i, node in enumerate(nodes):
        # Create a binary mask where pixels belonging to this node's leaves are True
        mask = np.zeros((height, width), dtype=bool)
        for leaf_label in node.leaves:
            mask |= (labels_img == leaf_label)
        
        masks[i] = mask
        node_labels.append(node.label)
    
    print(f"Extracted {len(nodes)} binary masks from the tree")
    return masks, node_labels





def main():
    dstdir = 'tmp'
    masks = np.load('tmp/approx_logits.npy') > 0

    # # filter by size 
    # min_accepted_size = 16
    # masks = np.stack([m for m in masks if m.sum() >= min_accepted_size])
    # structure = ndimage.generate_binary_structure(2, 1)
    # for ind in range(len(masks)):
    #     mask = masks[ind]
    #     holes = ~mask
    #     labeled_holes, num_features = ndimage.label(holes, structure=structure)
    #     small_holes = np.zeros_like(mask, dtype=bool)
    #     for i in range(1, num_features + 1):
    #         hole_size = np.sum(labeled_holes == i)
    #         if hole_size < min_accepted_size:
    #             small_holes[labeled_holes == i] = True
    #     masks[ind] = mask | small_holes

    labels = np.load('tmp/superpixels.npy').copy() + 1  # +1 to avoid 0 label
    # now restore the labels to a range
    current_label = 1
    for label_value in np.unique(labels):
        labels[labels == label_value] = current_label
        current_label += 1

    # extra variables
    rag = create_rag(labels)  # region adjacency graph
    areas = compute_areas(labels)
    refs = reference_masks_as_sets(masks, labels)
    ref_areas = np.array(
        [sum([areas[label] for label in refs[r_idx]]) for r_idx in range(len(refs))]
    )
    # build binary tree
    tree_root = build_tree(
        rag,
        areas,
        refs,
        ref_areas,
        labels,
        merges_to_try=128,
        # merges_to_try=2048,
        short_merge_find=False,
        score_change_threshold=0.0001,
        dstdir=dstdir,
    )
    # build new tree
    best_iou, relevant_nodes = get_relevant_nodes(tree_root, ref_areas)
    list_with_root = build_new_tree(relevant_nodes, rag)
    # create images
    images_per_level = create_images_at_multiple_levels(labels, list_with_root)
    for level in images_per_level:
        Image.fromarray(to_0255uint8(images_per_level[level])).save(
            f"{dstdir}/raw_level_{level}.png"
        )
    new_root, scores, lambdas = optimize_tree_merges(
        list_with_root[0],
        rag,
        areas,
        refs,
        ref_areas,
        lambda_param=0.00,
        target_node_count=280,
        labels_img=labels,
    )
    import matplotlib.pyplot as plt

    plt.figure()
    # plt.plot(np.array(scores))
    plt.plot(np.array(lambdas))
    plt.savefig(f"{dstdir}/lambdas.png")

    # create images
    images_per_level = create_images_at_multiple_levels(labels, [new_root])
    for level in images_per_level:
        Image.fromarray(to_0255uint8(images_per_level[level])).save(
            f"{dstdir}/level_{level}.png"
        )

    print("Done :)")
    # Extract masks from the final tree
    binary_masks, mask_labels = extract_binary_masks_from_tree(new_root, labels)
    # Save the masks
    np.save(f"{dstdir}/tree_masks.npy", binary_masks)


if __name__ == "__main__":
    main()
