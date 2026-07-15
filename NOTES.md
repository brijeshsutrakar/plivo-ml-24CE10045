# Engineering Notes
- Model Used: Random Forest Classifier for robust sequence predictions.
- Features: Extracted frame energy (dB) and F0 pitch contours from audio signals strictly before the pause.
- Strategy: Addressed class imbalance using balanced weights to minimize false cutoffs below the 5% threshold.
