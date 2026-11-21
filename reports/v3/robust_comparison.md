# V2 vs V3 robustness on the held-out test split

Samples: 640 from 80/92 classes. Font groups are held out from training; the source is synthetic font renders.

| Condition | V2 accuracy | V3 accuracy | Change |
|---|---:|---:|---:|
| Clean | 98.75% | 98.59% | -0.16 pp |
| Gaussian σ=0.02 | 71.56% | 98.59% | +27.03 pp |
| Gaussian σ=0.05 | 31.41% | 98.91% | +67.50 pp |
| Gaussian σ=0.10 | 4.53% | 98.91% | +94.37 pp |
| Salt & pepper 1% | 53.44% | 99.22% | +45.78 pp |
| Low contrast 0.45 | 1.25% | 98.28% | +97.03 pp |

V3 was selected using the validation split. The v2 and v3 checkpoints were evaluated on the same held-out test images. Gaussian noise σ is measured on normalized [0,1] pixels. Low contrast uses `x × 0.45 + 0.55 × 0.25`. Results do not measure handwriting generalization.
