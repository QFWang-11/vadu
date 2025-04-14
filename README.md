# View-aware Decomposition and Unification for Fast Ground-to-Aerial Person Search (VADU)`

[![Code](https://img.shields.io/badge/Code-GitHub-green)](https://github.com/QFWang-11/vadu)

Official implementation of the paper **"View-aware Decomposition and Unification for Fast Ground-to-Aerial Person Search"**.  
This work addresses the cross-view person search challenge between ground and aerial (UAV) cameras by proposing a framework that decomposes view-specific modules and unifies cross-view features for efficient and robust retrieval.

---

## Key Features
- **View-aware Decomposition**: Learns view-oriented modules for feature encoding and proposal generation, improving detection and feature discriminability within each view.
- **View-aware Unification**: Enhances cross-view feature alignment via shared re-id heads and prototypical contrastive learning.
- **Real-time Efficiency**: Achieves 32+ FPS on high-resolution images (1500×900) with ResNet-34 backbone.

## Dataset
The G2APS dataset contains 31,770 images (260K bounding boxes) captured by paired UAV and ground cameras.

  - Train subset: 21,962 images (10,810 aerial + 11,152 ground) with 2,048 identities.

  - Test subset: 9,808 images with 566 identities (1 ground query + 50 aerial galleries per identity).
## Results

### Performance on G2APS:

| Method   | Backbone | mAP | Top-1 | FPS(1500×900) |
| ---------| -------- | --- | ----- | ------------- |
| SeqNet   | Resnet34 | 27.60 | 35.34 | 23.79 |
| VADU(ours) |Resnet34| 32.23 | 38.87 | 32.07 |

### Ablation Study:

| Components | mAP  | Top-1 |
| --------- | ---- | ----- |
| Baseline (SeqNet) | 27.60 | 35.34  |
| + View-aware Decomposition   | 29.39 | 36.57  |
| + View-aware Unification   | 32.23 | 38.87  |


## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/QFWang-11/vadu.git
   cd vadu
2. Run `pip install -r requirements.txt` in the root directory of the project.

## Training
1. Download the G2APS dataset.
2. Pick one configuration file you like in `$ROOT/configs`, and run with it.

```
python train_method_sample_decomp.py --cfg configs/G2APS.yaml
```

## Test
Suppose the output directory is `$ROOT/exp_G2APS`. Test the trained model:

```
python test_re_id.py --cfg configs/G2APS.yaml --eval --ckpt $ROOT/exp_G2APS/epoch_10.pth
```




