
# steps

- fork sam2 repo
- install sam2 with `uv pip install -e .`  (from the forked repo root)
- create `ours` folder in the root of the repo and this file
- download to `ours/images/` dir two brodatz mosaics from `https://sipi.usc.edu/database/database.php?volume=textures&image=64#top` (see description here `https://sipi.usc.edu/database/USCTextureMosaics.pdf`), the image / mask pairs are `('texmos2.p512.tiff', 'texmos2.s512.tiff')` and `('texmos3b.p512.tiff', 'texmos3.s512.tiff')`
- comment out the bigger checkpoints in `checkpoints/download_ckpts.sh` and run `cd checkpoints; ./download_ckpts.sh` to download the tiny checkpoint 




