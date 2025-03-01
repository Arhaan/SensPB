# Installing roundingsat
----------------------

```
cd ./utils
./installrounding.sh
```

# Installations
```
pip install -r requirements.txt
```

# Running 
----------------------
The relevant code is present in `./src`
To run use the following command:
```
python sensitive.py <model file> --solver <solvername> --gap <int> --precision <int> --features 
```

Run 
```
python sensitive.py -h for help
```

Sample command:
```
python sensitive.py ../models/tree_verification_models/diabetes_robust/0020.model --solver z3 --gap 1.3 --precision 100
```

# Options
```
positional arguments:
  filenum               An integer file number. (Look in utils.py for list of files) or a filename

options:
  -h, --help            show this help message and exit
  --solver {z3,naive_z3,rounding,roundingsoplex,veritas}
                        The solver to use. Choose either 'z3' or 'rounding'.
  --max_trees MAX_TREES
                        Maximum number of trees to consider
  --all_single          run on all singular feature sets
  --plot                plot the results
  --gap GAP             Gap for checking sensitivity
  --precision PRECISION
                        Scale for checking sensitivity
  --features FEATURES [FEATURES ...]
                        Indexes of the features for which to do sensitivity analysis
  --details DETAILS     File containing names of features and their bounds
  --time TIME           Stopping time (in seconds), only for veritas
  
