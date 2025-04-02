
# steps

- fork sam2 repo
- install sam2 with `uv pip install -e .`  (from the forked repo root)
- create `ours` folder in the root of the repo and this file
- download to `ours/images/` dir two brodatz mosaics from `https://sipi.usc.edu/database/database.php?volume=textures&image=64#top` (see description here `https://sipi.usc.edu/database/USCTextureMosaics.pdf`), the image / mask pairs are `('texmos2.p512.tiff', 'texmos2.s512.tiff')` and `('texmos3b.p512.tiff', 'texmos3.s512.tiff')`
- comment out the bigger checkpoints in `checkpoints/download_ckpts.sh` and run `cd checkpoints; ./download_ckpts.sh` to download the tiny checkpoint 
- implement functions to extract c1 logits using sam2 in `ours/utils.py`
- extract c1logits
- `superpixels.ipynb`
    - extract the argmax of the c1 logits
    - take the max prob pixel of each of the argmax values as seed
    - grow regions where it's obvious (the argmax doesn't change)
    - grow regions by minimizing regret <- we found superpixels!
- `evaluate.ipynb`: can we improve over sam2 predictions?
    - doing binary merging, where the "score" is the mean of the logits on the two regions to be merged for any channel, didn't work very well qualitatively
    - we have merged superpixels to reconstruct the masks and it seems that the scores are slightly better!
    - we evaluated this for c1 logits and for logits. c1 logits are slightly better but they aren't needed, we'll remove them for simplicity
    - now the question is if we can get something better than sam when restricting the number of predicted masks? yes! indeed refining selected masks (279) with our superpixels (279) improves from 0.819 to 0.835! (in the second image, need to try in the first one too)




