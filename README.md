

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
To run, use the following command in the `./src` folder:
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

# Gap and Precision:
The tool uses two important parameters. The math is detailed in the ICLR paper cited.

Gap (float): The aim is to find two inputs $x^{(1)}$ and $x^{(2)}$ such that ${Output}(x^{(1)}) = 0.5  + \text{gap}$ and ${Output}(x^{(2)}) = 0.5 - \text{gap}$. Increasing the gap will lead to more significant counterexamples. As gap $\to 0$, all models become sensitive. A typical value to use for gap is 0.15.

Precision (float): SensPB is an approximate tool. We use a pseudo-Boolean encoding and want integers to represent leaf weights. To this end, we multiply the leaf values by the parameter precision and then take ceil/floor of the values. Our tool is sound but not complete for checking sensitivity, and increasing precision reduces the amount of wrong counterexamples produced. A rule of thumb is to use precision $= 10\times $ Number of trees in the ensemble. The math behind the theoretical guarantees is present in the ICLR paper cited. 

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
