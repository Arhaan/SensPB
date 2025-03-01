from joblib import Parallel, delayed
from tqdm import tqdm
import argparse
from utils import model_files
import veritas
import xgboost as xgb
import time
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
from utils import open_model

import pdb
import pickle
import time

start_time = 0
if __name__ != "__main__":
    debug = False
else:
    debug = True  # Flip if you want to debug


def contrasting_examples_from_solutions(feat_map, sol, columns):
    nb_features = len(feat_map)
    two_examples = np.zeros((2, nb_features))
    box = sol.box()

    for instance in (0, 1):
        for column in columns:
            feat_id = feat_map.get_feat_id(column, instance)
            feat_id_untrasformed = feat_map.get_feat_id(column, 0)
            if feat_id in box:
                interval = box[feat_id]
                if interval.lo_is_unbound():
                    assert not interval.hi_is_unbound()
                    value = interval.hi - 1e-4  # not inclusive
                else:
                    value = interval.lo

                two_examples[instance, feat_id_untrasformed] = value

    return pd.DataFrame(
        two_examples, columns=columns, index=["instance 0", "instance 1"]
    )


def constrast_two_examples(
    at, columns, nonfixed_columns, multiclass=False, truelabel=-1, otherlabel=-1
):
    """Create an `veritas.AddTree` that contrast two instances.

    The new AddTree outputs difference between the original AddTree's outputs
    for instances 0 and 1.

    at: The original veritas.AddTree tree ensemble model
    columns: array with column names
    nonfixed_columns: columns that are allowed to change between the two instances
    """
    feat_map = veritas.FeatMap(columns)
    for column in columns:
        if column not in nonfixed_columns:
            index_for_instance0 = feat_map.get_index(column, 0)
            index_for_instance1 = feat_map.get_index(column, 1)
            feat_map.use_same_id_for(index_for_instance0, index_for_instance1)

    at_for_instance1 = feat_map.transform(at, 1)
    if multiclass:
        at.swap_class(truelabel)
        at_for_instance1.swap_class(truelabel)
        at_for_instance1.swap_class(otherlabel)
        at_contrast = at.concat_negated(at_for_instance1.negate_leaf_values())
    else:
        at_contrast = at.concat_negated(at_for_instance1)

    print("  Feature IDs used by instance 0\n   and instance 1 respectively:")
    print("-" * (25 + 4 + 4))
    for column in columns:
        mark = "*" if column in nonfixed_columns else ""
        feat_id_instance0 = feat_map.get_feat_id(column, 0)
        feat_id_instance1 = feat_map.get_feat_id(column, 1)
        print(f"{column:25s} {feat_id_instance0:3d} {feat_id_instance1:3d}", mark)

    return at_contrast, feat_map


def search_anomaly_for_features(cols, model, max_trees, args):
    at = veritas.get_addtree(model, silent=True)

    # at = veritas.get_addtree(model, silent=True)
    # We optimize the output of the contrasting ensemble `at_contrast`
    if args.multiclass:
        config = veritas.Config(veritas.HeuristicType.MULTI_MAX_MAX_OUTPUT_DIFF)
    else:
        config = veritas.Config(veritas.HeuristicType.MAX_OUTPUT)

    # We are interested in cases where the output of instance 0 is greater
    # than the output of instance 1, i.e., their output difference is greater
    # than 0.0.
    config.ignore_state_when_worse_than = 0.0

    # Veritas uses an approximate search.
    # We modify the parameters of the approximate search to more aggressively work
    # on lowering the upper bound instead of also trying to find suboptimal solutions.
    if not args.multiclass:
        config.focal_eps = 0.95
        config.max_focal_size = 100

    # Obtain the search object from the search configuration
    # We can optionally constrain certain feature values (e.g., look only for
    # cases where instance 0 describes a male person)
    # prune_box = {feat_map.get_feat_id("personal_status", 0): veritas.Interval(0, 0.5)}

    nonfixed_cols = [f"F{col}" for col in cols]
    at_contrast, feat_map = constrast_two_examples(
        at,
        [f"F{i}" for i in range(model.num_features())],
        nonfixed_cols,
        multiclass=args.multiclass,
        truelabel=args.truelabel,
        otherlabel=args.otherlabel,
    )

    prune_box = {}
    search = config.get_search(at_contrast, prune_box)

    bounds = []
    num_search_steps_per_iteration = 100
    stop_reason = search.step_for(args.time, 100)
    print(stop_reason)

    # while stop_reason != stop_reason.OPTIMAL:  # search until it is certain that the best solution is optimal
    #     stop_reason = search.steps(num_search_steps_per_iteration)
    #     bound_lh = search.current_bounds()
    #     bounds.append((bound_lh.atleast, bound_lh.top_of_open))
    try:
        sol = search.get_solution(0)
        two_examples = contrasting_examples_from_solutions(
            feat_map, sol, [f"F{i}" for i in range(model.num_features())]
        )
        predictions = pd.Series(
            model.predict(xgb.DMatrix(two_examples)), index=["instance 0", "instance 1"]
        )
        ret = np.array(two_examples).tolist()
        for col in cols:
            ret[0][col] = (ret[0][col], ret[1][col])
        print(
            cols, " ", time.time() - start_time, " ", np.array(predictions), " ", ret[0]
        )
    except Exception as e:
        print(cols, " ", time.time() - start_time, " NO SOLS")


# Parse the arguments
def main(args):
    close = args.close
    max_trees = args.max_trees
    features = args.features
    mfile = args.filenum
    start_time = time.time()
    # model,trees,n_trees,n_features,num_classes= open_model(mfile, max_trees, veritas=True)
    model = xgb.Booster({"nthread": 4})  # Use XGBRegressor() for regression tasks
    model.load_model(mfile)
    n_features = model.num_features()

    if features is None:
        features = [0]
    tasks = [(features, 0, 0)]  #  [4,7,10,17,18] did not solve overnight
    if args.all_single:

        tasks = [
            ([f], 0, 0) for f in range(0, args.n_features)
        ]  #  [4,7,10,17,18] did not solve overnight
    else:
        tasks = [(features, 0, 0)]  #  [4,7,10,17,18] did not solve overnight

    # tasks = [ ([f, k], precision, gap) for f in range(0, n_features) for k in range(0,n_features) ] #  [4,7,10,17,18] did not solve overnight
    def runner(tupl):
        f = tupl[0]
        precision = tupl[1]
        gap = tupl[2]
        # limit_range_list[10] = (age,age)
        # print(f'Searching for prediction change from {gap} to -{gap} by changing feature {names[f]}:')
        start_time = time.time()
        result = search_anomaly_for_features(f, model, max_trees, args)

    print(tasks)
    if True:
        results = [runner(params) for params in tqdm(tasks)]
    else:
        results = Parallel(n_jobs=-1, timeout=2 * 60 * 60)(
            delayed(runner)(params) for params in tqdm(tasks)
        )
