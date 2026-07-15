# End-of-Turn (EOT) Detection System

An optimized, multilingual End-of-Turn prediction pipeline built with Random Forest using acoustic frame-level dynamics.

## Performance Metrics
* **Mean Response Delay:** 612 ms (Baseline: 1600 ms)
* **Area Under Curve (AUC):** 0.965
* **Interrupted Turns:** 4.0% (Within the <= 5.0% threshold constraint)

## Engineered Features
1. **Normalized Energy & Drop:** Calculated using trailing audio windows to spot sudden decays in conversational speech.
2. **Pitch Variance & Slope:** Tracked over voiced components via F0 contours to map specific conversational sentence terminations (EOT vs. Hold).
3. **Voicing Context Ratio:** Captures conversational pauses vs. active articulation distribution.

## How to Run Inference
Ensure the `model.pkl` file exists in the directory, then run:
```bash
python predict.py --data_dir <path_to_data_folder> --out <output_csv_path>
```