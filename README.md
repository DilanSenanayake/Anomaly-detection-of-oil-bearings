# Anomaly Detection of Oil Bearings

Data-driven detection and classification of bearing anomalies in power generation plants using a convolutional neural network (CNN).

Power plants are exposed to mechanical faults that can cause costly downtime and safety risks. This project learns patterns from multi-sensor plant signals and classifies each time window as **normal** or one of **three anomaly types**, so early bearing issues can be flagged before failure.

**Research paper:** [A Data-Driven Approach Based on Artificial Neural Networks for the Detection and Classification of Bearing Anomalies in Power Generation Plants](https://www.researchgate.net/publication/371016182_A_Data-Driven_Approach_Based_on_Artificial_Neural_Networks_for_the_Detection_and_Classification_of_Bearing_Anomalies_in_Power_Generation_Plants)

---

## Repository layout

| File | Description |
|------|-------------|
| `Model_03(05).ipynb` | End-to-end pipeline: data load, windowing, training, validation |
| `final_plot.py` | PyQt5 desktop app that loads a trained model and visualizes live-style plots |
| `Test.ui` | Qt Designer UI for the **Bearing Condition Monitor** GUI |

---

## Problem setup

**Task:** 4-class supervised classification of short sensor windows.

| Label | Class |
|------:|-------|
| 0 | Normal / good condition |
| 1 | Anomaly type 01 |
| 2 | Anomaly type 02 |
| 3 | Anomaly type 03 |

**Input sensors** (four channels, stacked as CNN input channels):

| Column | Role (as used in the GUI / notebook) |
|--------|--------------------------------------|
| `bb10` | Bearing vibration |
| `TNH` | Generator / turbine speed (RPM %) |
| `DWATT` | Active power |
| `btgj1` | Oil / bearing temperature-related channel |

---

## Dataset

Source data comes from a real power-generation plant (KCCP). In the notebook it is loaded from Excel / CSV files such as:

- `Good.xlsx` — normal operating data  
- `Anomaly/Type 01.xlsx`, `Type 02.xlsx`, `Type 03.xlsx` — labelled anomaly recordings  
- Optional trend CSV for additional validation  

### Raw series used for modelling

For each class, the four sensor columns are taken and truncated to **100,000** timesteps per anomaly type (full file lengths noted in the notebook are larger: ~923k / ~963k / ~261k). Normal data is likewise taken as **100,000** samples.

### Windowing → model samples

1. Split each 1-D series into non-overlapping windows of length **100**.  
2. Keep **1,000 windows per class** → **4,000 samples** total.  
3. Min–max normalize each channel using the **normal (good) data** min/max for that channel.  
4. Reshape each 100-point window into a **10 × 10** “signal image”.  
5. Stack the four channels → tensor shape **`(N, 4, 10, 10)`**.

| Stage | Shape |
|-------|-------|
| Raw per channel (per class, after truncate) | `(100000,)` |
| Windows per channel (all classes) | `(4000, 100)` |
| Per-channel image | `(4000, 10, 10)` |
| Final model input `X` | **`(4000, 4, 10, 10)`** |
| Labels | **`(4000,)`** with values `{0,1,2,3}` |

### Train / test split

- `train_test_split(..., test_size=0.2, random_state=20)`  
- **Train:** 3,200 samples  
- **Test:** 800 samples  
- Stored as `float16` / `double` tensors for PyTorch loaders  

---

## Model architecture (`CNN`)

2-D CNN over the 4-channel 10×10 signal images (PyTorch).

```
Input:  (batch, 4, 10, 10)
  Conv2d(4 → 32, kernel=4, stride=1, padding=1) + MaxPool2d(2, stride=1) + ReLU
  Conv2d(32 → 64, kernel=4, stride=1, padding=1) + MaxPool2d(2, stride=1) + ReLU
  Flatten → 2304 features
  Linear(2304 → 256) + ReLU
  Dropout(p=0.4)
  Linear(256 → 4)
  LogSoftmax (dim=1)
```

| Hyperparameter | Value |
|----------------|-------|
| Loss | `CrossEntropyLoss` |
| Optimizer | Adam, `lr = 1e-4` |
| Epochs | 4 |
| Train batch size | 128 |
| Test batch size | 16 |
| Output classes | 4 |

In the recorded training run, batch train accuracy rises from ~30% to **~100%** by epoch 4; held-out batches in the notebook also report high accuracy (often 100% on individual test batches). Treat those numbers as experimental results from the Colab run, not a guaranteed production metric.

Saved weights are referenced in code as `Model_03.pth`.

---

## GUI (`final_plot.py` + `Test.ui`)

Desktop **Bearing Condition Monitor** (PyQt5 + Matplotlib) that:

- Loads plant CSV data and the trained CNN  
- Builds the same 100-sample → 10×10 × 4-channel input  
- Shows plots for vibration, RPM, oil temperature, and active power  
- Returns the predicted class for a selected time index  

Paths for `dataset.csv` and `Model_03.pth` in `final_plot.py` are currently machine-specific and need to be updated for your environment.

---

## Dependencies

Typical stack used by the notebook and GUI:

- Python 3  
- `numpy`, `pandas`  
- `torch` (PyTorch)  
- `scikit-learn`  
- `matplotlib`  
- `openpyxl` / Excel support for `.xlsx` loads  
- `PyQt5` (GUI only)  

Notebook development was done in **Google Colab** with Google Drive–mounted data paths.

---

## How to run

### Train / evaluate

1. Place the Excel/CSV plant files where the notebook expects them (or edit the paths).  
2. Open and run `Model_03(05).ipynb` top to bottom.  
3. Optionally save the trained model with `torch.save(...)`.

### GUI

1. Point `final_plot.py` at your `dataset.csv` and `Model_03.pth`.  
2. Ensure `Test.ui` (and any helper modules such as `TestPlot`) are available.  
3. Run:

```bash
python final_plot.py
```

---

## Method summary

```text
Plant sensors (bb10, TNH, DWATT, btgj1)
        │
        ▼
  Window length 100  →  min–max (vs good data)
        │
        ▼
  Reshape to 10×10 per channel  →  stack to (4, 10, 10)
        │
        ▼
           CNN classifier  →  {Normal, Type01, Type02, Type03}
```

This converts multi-channel time series into compact multi-channel images so a small CNN can learn spatial structure in the window and separate normal operation from three anomaly regimes.
