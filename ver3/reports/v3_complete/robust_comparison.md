# Robustness comparison — complete 92-class test split

Samples: 928 across all 92 labels. The original flat font families and system-font families are held out by split.

| Condition | V2 baseline | Previous v3 (80 labels) | Complete v3 | Complete vs previous | Complete vs v2 |
|---|---:|---:|---:|---:|---:|
| Clean | 99.35% | 97.52% | 99.25% | +1.73 pp | -0.10 pp |
| Gaussian σ=0.02 | 76.19% | 97.41% | 99.25% | +1.84 pp | +23.06 pp |
| Gaussian σ=0.05 | 30.82% | 97.20% | 99.25% | +2.05 pp | +68.43 pp |
| Gaussian σ=0.10 | 7.87% | 96.66% | 99.25% | +2.59 pp | +91.38 pp |
| Salt & pepper 1% | 53.77% | 96.98% | 99.14% | +2.16 pp | +45.37 pp |
| Low contrast 0.45 | 3.45% | 96.44% | 99.14% | +2.70 pp | +95.69 pp |

This test set includes locally rendered font images for the 12 classes missing from the supplied corpus. It is not a handwritten benchmark. Gaussian σ is measured on normalized [0,1] pixels; low contrast uses `x × 0.45 + 0.55 × 0.25`.
