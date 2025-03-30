import numpy as np
import argparse
from collections import defaultdict

import skimage
import matplotlib.pyplot as plt


def compute_region_stats(label_img, logits):
    """Compute per-region sum (for all channels) and area.
    - regions[label]['sum'] is a vector of sums for each channel.
    - regions[label]['area'] is the area (number of pixels) of the region.
    """
    regions = {}
    unique_labels = np.unique(label_img)
    for label in unique_labels:
        mask = label_img == label
        area = np.count_nonzero(mask)
        # sum over all channels for this region (result is shape: (C,))
        sum_val = logits[:, mask].sum(axis=1)
        regions[label] = {"sum": sum_val, "area": area}
    return regions


def build_rag(label_img):
    """Build region adjacency graph using 4-neighbor connectivity.
    Returns a dictionary where each region maps to the set of its neighboring region labels.
    """
    H, W = label_img.shape
    neighbors = defaultdict(set)
    # check right and down neighbors for each pixel
    for i in range(H):
        for j in range(W):
            current = label_img[i, j]
            if j + 1 < W:
                right = label_img[i, j + 1]
                if right != current:
                    neighbors[current].add(right)
                    neighbors[right].add(current)
            if i + 1 < H:
                down = label_img[i + 1, j]
                if down != current:
                    neighbors[current].add(down)
                    neighbors[down].add(current)
    return neighbors


def update_rag(neighbors, merge_from, merge_into):
    """
    When merging region merge_from into region merge_into:
    - merge_into gets union of neighbors (excluding self and merge_from)
    - update neighbor sets of all neighbors of merge_from: replace merge_from with merge_into.
    - remove merge_from entry.
    """
    # Update neighbors for merge_into region
    new_neighbors = neighbors[merge_into].union(neighbors[merge_from])
    new_neighbors.discard(merge_into)
    new_neighbors.discard(merge_from)
    neighbors[merge_into] = new_neighbors

    # For each neighbor of merge_from, replace merge_from with merge_into.
    for nb in list(neighbors[merge_from]):
        neighbors[nb].discard(merge_from)
        if nb != merge_into:
            neighbors[nb].add(merge_into)
    # Remove merge_from entry from rag.
    del neighbors[merge_from]
    # Remove self-loops if any.
    for key in neighbors:
        neighbors[key].discard(key)


def greedy_binary_merge(label_img, logits, target_regions):
    """
    Greedy merge of superpixels based on the union's max mean logit over any channel.
    Merges continue until the number of regions becomes target_regions.
    Returns the final labeling.
    """
    print("- computing stats")
    regions = compute_region_stats(label_img, logits)
    print("- building rag")
    neighbors = build_rag(label_img)

    # Current labels in the segmentation.
    current_labels = set(regions.keys())

    # Number of merges to perform is current number of regions - target_regions.
    num_merges = len(current_labels) - target_regions
    merge_history = []  # store tuple(merge_into, merge_from) for each merge

    for merge_iter in range(num_merges):
        plt.imsave("current.png", label_img / label_img.max(), cmap="nipy_spectral")
        best_score = -np.inf
        best_pair = None

        # Iterate over all pairs in the current rag.
        for reg in list(current_labels):
            if reg not in neighbors:  # region might have been removed
                continue
            for nb in neighbors[reg]:
                # To avoid duplicate pair processing, enforce ordering.
                if reg < nb:
                    total_sum = regions[reg]["sum"] + regions[nb]["sum"]  # shape: (C,)
                    total_area = regions[reg]["area"] + regions[nb]["area"]
                    union_mean = total_sum / total_area  # mean per channel
                    score = np.max(union_mean)  # maximum over channels
                    if score > best_score:
                        best_score = score
                        best_pair = (reg, nb)
        if best_pair is None:
            # No more merges possible.
            break

        # Choose the lower label as the merged label.
        r1, r2 = best_pair
        merge_into = min(r1, r2)
        merge_from = max(r1, r2)

        # Update region stats.
        regions[merge_into]["sum"] += regions[merge_from]["sum"]
        regions[merge_into]["area"] += regions[merge_from]["area"]
        del regions[merge_from]

        # Update the labeling array: reassign all pixels of merge_from to merge_into.
        label_img[label_img == merge_from] = merge_into

        # Update current labels.
        current_labels.discard(merge_from)

        # Update rag.
        update_rag(neighbors, merge_from, merge_into)

        merge_history.append((merge_into, merge_from))
        print(
            f"Merge {merge_iter+1}/{num_merges}: merged {merge_from} -> {merge_into} with score {best_score:.4f}"
        )

    return label_img, merge_history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Greedy binary merging of superpixels using max mean logit over all channels."
    )
    parser.add_argument(
        "--regions",
        type=int,
        default=50,
        help="Number of regions to visualize after merging",
    )
    args = parser.parse_args()

    # --- Load data ---
    logits = np.load("tmp/c1logits.npy")  # shape (C, H, W)
    superpixels = np.load("tmp/superpixels.npy")  # shape (H, W)
    superpixels = skimage.measure.label(superpixels, background=-1, connectivity=1)
    final_labels, merges = greedy_binary_merge(superpixels.copy(), logits, args.regions)

    # Save the resulting labeling.
    import matplotlib.pyplot as plt

    plt.figure()
    plt.imsave("merged.png", final_labels, cmap="nipy_spectral")
    np.save("tmp/merged_superpixels.npy", final_labels)
    print(
        f"Final labeling saved to tmp/merged_superpixels.npy with unique regions: {np.unique(final_labels).size}"
    )
