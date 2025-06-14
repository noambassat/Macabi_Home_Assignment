# Macabi Home Assignment

This repository contains a Python script `pipeline.py` that was automatically
converted from the Jupyter notebook `macabi_home_assignment.ipynb`.
The script executes the full analysis pipeline used in the original
notebook.

## Requirements
* Python 3.8+
* pip packages: pandas, numpy, matplotlib, seaborn, scikit-learn, lightgbm,
  shap, sentence-transformers, torch, tqdm, ydata-profiling, sweetviz,
  wordcloud, python-bidi, arabic-reshaper

Some packages may require additional system dependencies.

## Usage
1. Place the data file `ds_assignment_data.csv` in this directory.
2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt  # or install packages manually
   ```
3. Run the pipeline:
   ```bash
   python pipeline.py
   ```

The script follows the same steps as the notebook, including feature
engineering, model training and evaluation. Because it was generated
from the notebook, it may still contain calls that display figures or
intermediate results.
