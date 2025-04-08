

# Installations
Create a virtual environment by following the information [here](https://docs.python.org/3/library/venv.html). 
The typical instructions are:
```
python -m venv /path/to/new/virtual/environment
source <venv>/bin/activate
```
Install all the requirements in this virtual environment by running:
```
pip install -r requirements.txt
```

## Installing roundingsat (Optional)
```
cd ./utils
./installrounding.sh
```

# Running 
----------------------
The relevant code is present in `./src`
To run use the following command:
```
python sensitive.py <model file> --solver <solvername> --gap <float> --precision <int> --features 
```

For help run: 
```
python sensitive.py -h
```


Sample command:
```
python sensitive.py ../models/tree_verification_models/diabetes_robust/0020.model --solver z3 --gap 1.3 --precision 100
```


# Options
```
positional arguments:
  file               Filename containing the saved XGBoost model.

options:
  -h, --help            show this help message and exit
  --solver {z3,naive_z3,rounding,roundingsoplex,veritas}
                        Solving method to use. 
  --gap GAP             Gap for checking sensitivity
  --precision PRECISION
                        Scale for checking sensitivity
  --features FEATURES [FEATURES ...]
                        Indexes of the features for which to do sensitivity analysis
  --all_single          run on all singular feature sets
  --time TIME           Stopping time (in seconds), only for veritas
  --max_trees MAX_TREES
                        Maximum number of trees to consider.

```
# Citations
[Sensitivity Verification for Decision Tree Ensembles](https://openreview.net/forum?id=h0vC0fm1q7).
Arhaan Ahmad, Tanay Vineet Tayal, Ashutosh Gupta and S. Akshay. _ICLR 2025_
