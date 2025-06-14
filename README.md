# Macabi Home Assignment

This repository contains a Python script `pipeline.py` automatically converted from the Jupyter notebook `macabi_home_assignment.ipynb`. It executes the full analysis pipeline from the original notebook.

## Getting Started

Clone the repository and move into the project directory:

```bash
git clone https://github.com/your-username/Macabi_Home_Assignment.git
cd Macabi_Home_Assignment
```

Place the data file `ds_assignment_data.csv` in this directory and install the required Python packages:

```bash
pip install -r requirements.txt
```

Run the pipeline:

```bash
python pipeline.py
```

## Requirements
* Python 3.8+
* pip packages: pandas, numpy, matplotlib, seaborn, scikit-learn, lightgbm, shap, sentence-transformers, torch, tqdm, ydata-profiling, sweetviz, wordcloud, python-bidi, arabic-reshaper

Some packages may require additional system dependencies.

The script follows the same steps as the notebook, including feature engineering, model training and evaluation. Because it was generated from the notebook, it may still contain calls that display figures or intermediate results.
