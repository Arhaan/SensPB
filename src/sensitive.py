#!/usr/bin/python3
import sys
import pickle
from utils import model_files
import json
import sys
import argparse
import joblib
import xgboost as xgb
import os
import numpy as np
import pandas as pd
import itertools
import ast
import z3
import math
import random
import time
from joblib import Parallel, delayed
from tqdm import tqdm
from rangedbooster import ExtendedBooster
import pdb
from converttoopb import roundingSolve

from subprocess import check_output
from xyplot import Curve

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from solve_veritas import main as veritas_solver
from utils import open_model_xgb

sureofcounter = False

pd.set_option("display.max_rows", 500)

# --------------------------------------
# Utilities
# --------------------------------------


def dump_solver(solver, filename):
    smt2 = solver.sexpr()
    with open(filename, mode="w", encoding="ascii") as f:  # overwrite
        f.write(smt2)
        f.close()


def solve(phi):
    tic = time.perf_counter()
    s = z3.Solver()
    s.add(phi)
    r = s.check()
    toc = time.perf_counter()
    if r == z3.sat:
        m = s.model()
        return m
    return None


# --------------------------------------
# Model handle
# --------------------------------------


def plot_variations(model, data, features, trees, feature_names, op_range_list):
    fvalues = []
    for feature in features:
        sliced = trees[(trees["Feature"] == f"f{feature}")][["Feature", "Split"]].copy()
        sliced.sort_values(["Split"], inplace=True)
        sliced.drop_duplicates(inplace=True)
        sliced = sliced[
            (op_range_list[feature][0] < sliced["Split"])
            & (sliced["Split"] <= op_range_list[feature][1])
        ]
        # values = [op_range_list[feature][0]] + sliced['Split'].tolist()
        values = sliced["Split"].tolist()
        fvalues.append(values)

    # print(fvalues[0])
    predictions = []
    if len(fvalues) == 2:
        for v in fvalues[1]:
            data[features[1]] = v
            rows = []
            for f0 in fvalues[0]:
                data[features[0]] = f0
                rows.append(data.copy())
            predict_col = model.predict(xgb.DMatrix(rows))
            predictions.append(predict_col)
    else:
        rows = []
        for f0 in fvalues[0]:
            data[features[0]] = f0
            rows.append(data.copy())
        predict_col = model.predict(xgb.DMatrix(rows))
        # predict_col = [ sigmoid_inv(x) for x in predict_col]
        predictions.append(predict_col)

    plt.style.use("_mpl-gallery")
    if len(fvalues) == 1:
        fig, ax = plt.subplots()
        ax.plot(fvalues[0], predictions[0], linewidth=2.0)
        plt.ylabel("Predict")
        plt.xlabel(feature_names[features[0]])
    else:
        ax = plt.figure().add_subplot(projection="3d")
        X, Y = np.meshgrid(np.array(fvalues[0]), np.array(fvalues[1]))
        Z = np.array(predictions)
        ax.plot_surface(
            X,
            Y,
            Z,
            cmap=cm.Blues,
            # edgecolor='royalblue',
            # lw=0.5, #rstride=8, cstride=8,
            # alpha=0.3
        )
        # ax.contour(X, Y, Z, zdir='z', offset=-100, cmap='coolwarm')
        # ax.contour(X, Y, Z, zdir='x', offset=-40, cmap='coolwarm')
        # ax.contour(X, Y, Z, zdir='y', offset=40, cmap='coolwarm')
        ax.set(
            xlabel=feature_names[features[0]],
            ylabel=feature_names[features[1]],
            zlabel="Predict",
        )
    plt.show()


def search_anomaly_for_features(
    features,
    gap,
    precision,
    n_classes,
    model,
    trees,
    n_trees,
    args,
    op_range_list,
):
    gap = int(gap * precision)
    testing = False
    trees = trees[trees["Tree"] < n_trees]
    vars1 = {}
    vars2 = {}

    # ------------------------------------------------------
    # For deugging only
    # ------------------------------------------------------
    interested_bits = {}
    # -------------------------------------------------------
    # Make value for each node
    # -------------------------------------------------------
    if encoding == "allsum":
        for idx, row in trees.iterrows():
            vars1[row["ID"]] = z3.Real("v1-" + "i-" + row["ID"])
            vars2[row["ID"]] = z3.Real("v2-" + "i-" + row["ID"])
    else:
        for idx, row in trees.iterrows():
            vars1[row["ID"]] = z3.Bool("v1-" + "b-" + row["ID"])
            vars2[row["ID"]] = z3.Bool("v2-" + "b-" + row["ID"])

    # -----------------------------------------------------------
    # Make bits for each feature and constrains on the feature bits
    # -----------------------------------------------------------
    def make_bits_for_features(i, prefix, sliced, vmap, cons):
        prev = False
        for r, row in sliced.iterrows():
            split = row["Split"]
            split = str(split)
            fname = f"f{i}_{split}"
            v = z3.Bool(f"{prefix}_b_" + fname)
            cons.append(z3.Implies(prev, v))
            prev = v
            vmap[fname] = v

    split_bit_map = {}
    split_value_map = {}
    ord_bits_cons = []
    n_features = model.num_features()

    for i in range(n_features):
        sliced = trees[(trees["Feature"] == f"f{i}")][["Feature", "Split"]].copy()
        sliced.sort_values(["Split"], inplace=True)
        sliced.drop_duplicates(inplace=True)
        sliced = sliced[
            (op_range_list[i][0] < sliced["Split"])
            & (sliced["Split"] <= op_range_list[i][1])
        ]
        split_bit_map[i] = []
        prev = op_range_list[i][0]
        for r, row in sliced.iterrows():
            var_name = f"f{i}" + "_" + str(row["Split"])
            split_bit_map[i].append(var_name)
            split_value_map[var_name] = prev
            prev = float(row["Split"])
        make_bits_for_features(i, "v1", sliced, vars1, ord_bits_cons)
        make_bits_for_features(i, "v2", sliced, vars2, ord_bits_cons)
        split_value_map[f"f{i}" + "_" + str("Last")] = prev

    def not_too_far(d_idx, vars1, vars2, cons):  # TODO
        num_splits = len(split_bit_map[d_idx])
        allowed_diff = max(5, int(num_splits / 10))
        for r in range(-1, num_splits):
            if r + allowed_diff >= num_splits:
                continue
            if r != -1:
                b_r0 = vars1[split_bit_map[d_idx][r]]
            else:
                b_r0 = False
            edge_cond = z3.And(z3.Not(b_r0), vars1[split_bit_map[d_idx][r + 1]])
            cons.append(
                z.Implies(edge_cond, vars2[split_bit_map[d_idx][r + allowed_diff]])
            )
            if r != -1:
                b_r0 = vars2[split_bit_map[d_idx][r]]
            else:
                b_r0 = False
            edge_cond = z3.And(z3.Not(b_r0), vars2[split_bit_map[d_idx][r + 1]])
            cons.append(
                z3.Implies(edge_cond, vars1[split_bit_map[d_idx][r + allowed_diff]])
            )

    def limit_range(d_idx, vars_list, cons):
        for i, b_name in enumerate(split_bit_map[d_idx]):
            value = split_value_map[b_name]
            try:
                if value < limit_range_list[d_idx][0]:
                    for vars in vars_list:
                        cons.append(z3.Not(vars[b_name]))
                if limit_range_list[d_idx][1] <= value:
                    for vars in vars_list:
                        cons.append(vars[b_name])
            except:
                pass

    def all_equal_but_a_few(d_idxs, vars_list, num_features):
        cons = []
        vars1, vars2 = vars_list[0], vars_list[1]
        for idx in range(0, num_features):
            if idx in d_idxs:
                continue
                # if not close:
                #     continue
                exactly1neq = [
                    (z3.Not(vars1[fname] == vars2[fname]), 1)
                    for fname in split_bit_map[idx]
                ]
                if len(exactly1neq) == 0:
                    continue
                cons.append(z3.PbLe(exactly1neq, 100000000))
            for fname in split_bit_map[idx]:
                cons.append(vars1[fname] == vars2[fname])
        if args.small_change:
            for d_idx in d_idxs:
                not_too_far(d_idx, vars1, vars2, cons)
        return cons

    def get_feature_bit(feature, split, vars):
        f = int(feature[1:])
        try:
            if split <= op_range_list[f][0]:
                return False
            elif split > op_range_list[f][1]:
                return True
        except:
            if split <= 0:
                return False
        return vars[feature + "_" + str(split)]

    # ---------------------------------------
    #
    # ---------------------------------------

    def gen_cons_tree(trees, vars, up):
        cons = []
        for idx, row in trees.iterrows():
            v = vars[row["ID"]]
            if row["Feature"] == "Leaf":
                if up:
                    expr = int(np.ceil(row["Gain"] * precision))
                else:
                    expr = int(np.floor(row["Gain"] * precision))
            else:
                split = row["Split"]
                f = int(row["Feature"][1:])
                if split <= op_range_list[f][0]:
                    cond = False
                elif split > op_range_list[f][1]:
                    cond = True
                else:
                    cond = vars[row["Feature"] + "_" + str(split)]
                yes = vars[row["Yes"]]
                no = vars[row["No"]]
                expr = z3.If(cond, yes, no)
            cons.append(v == expr)
        return cons

    pows = [-64, -32, -16, -8, -4, -2, -1, 0, 1, 2, 4, 8, 16, 32, 64]

    def abstract_prob(p, precision, up):
        if up:
            if sureofcounter:
                val = int(np.floor(p * precision))
            else:
                val = int(np.ceil(p * precision))
            # for i,abst in enumerate(pows):
            #     if val <= abst: break
        else:
            if sureofcounter:
                val = int(np.ceil(p * precision))
            else:
                val = int(np.floor(p * precision))
            # for i,abst in enumerate(pows):
            #     if val < abst: break
        return val

    def gen_pb_cons_tree(trees, vars, up, rangemap={}, stop=lambda x, y: False):
        cons = []
        values = {}
        ignore = []
        parent = {}
        if up:
            up_name = "u-"
        else:
            up_name = "d-"
        for tid in range(n_trees):
            values[tid] = {}
            parent[f"{tid}-0"] = None
        for idx, row in trees.iterrows():
            v = vars[row["ID"]]
            tid = row["Tree"] *  row["class"]
            if row["ID"] in ignore:
                ignore.append(row["Yes"])
                ignore.append(row["No"])
                continue
            if row["Feature"] == "Leaf":
                tid = row["Tree"] + row["class"]
                val = abstract_prob(row["Gain"], precision, up)  # What is this? Arhaan
                bit = z3.Bool(f"{up_name}{tid}-{val}")
                if val in values[tid]:
                    values[tid][val][1].append(v)
                else:
                    values[tid][val] = (bit, [v])
            elif stop(row, rangemap):
                # ----------------------------------------------------------
                # Do not explore the subtree that have similar output leaves
                # ----------------------------------------------------------
                if up:
                    val = int(np.ceil(precision * rangemap[row["ID"]][1]))
                else:
                    val = int(np.floor(precision * rangemap[row["ID"]][0]))
                bit = z3.Bool(f"{up_name}{tid}-{val}")
                if val in values[tid]:
                    values[tid][val][1].append(v)
                else:
                    values[tid][val] = (bit, [v])
                # Don't traverse this tree further
                ignore.append(row["Yes"])
                ignore.append(row["No"])
            else:
                cond = get_feature_bit(row["Feature"], row["Split"], vars)
                cons.append(z3.And(v, cond) == vars[row["Yes"]])
                cons.append(z3.And(v, z3.Not(cond)) == vars[row["No"]])
                parent[row["Yes"]] = (row["ID"], cond, int(row["Feature"][1:]))
                parent[row["No"]] = (row["ID"], z3.Not(cond), int(row["Feature"][1:]))

        cons += [vars[f"{tid}-0"] for tid in range(n_trees)]  # Root nodes are true
        all_leaves = []

        for tid in range(n_trees):
            bits_map = values[tid]
            tree_leaves = []
            for val, (bit, leaves) in bits_map.items():
                all_leaves.append((val, bit))
                tree_leaves.append((1, bit))
                cons.append(z3.Or(leaves) == bit)
            cons.append(z3.PbEq(tree_leaves, 1))
            # for pair in bits: all_leaves.append(pair)
        return cons, all_leaves

    # ---------------------------------------
    # Output constrains
    # ---------------------------------------
    ugap = gap
    lgap = -gap
    model = ExtendedBooster(model)
    if args.stop:
        model.compute_node_ranges()
    rangemap = model.node_ranges
    if args.stop:
        stop = (
            lambda row, ran: ran[row["ID"]][1] - ran[row["ID"]][0]
            < args.stop_param * gap / precision
        )
    else:
        stop = lambda x, y: False
    model = model.booster
    if encoding == "allsum":
        cs1 = gen_cons_tree(trees, vars1, up=True)
        cs2 = gen_cons_tree(trees, vars2, up=False)
        expr1 = sum([vars1[f"{tid}-0"] for tid in range(n_trees)])
        expr2 = sum([vars2[f"{tid}-0"] for tid in range(n_trees)])
        prop = [(expr1 > ugap), (expr2 < lgap)]
    else:
        cs1, up_leaves = gen_pb_cons_tree(
            trees, vars1, up=True, rangemap=rangemap, stop=stop
        )
        cs2, down_leaves = gen_pb_cons_tree(
            trees, vars2, up=False, rangemap=rangemap, stop=stop
        )
        unchanged = []

        def merge_and_negate(list1, list2):
            return list1 + [(-w, var) for w, var in list2]

        prop = [z3.PbGe(up_leaves, ugap), z3.PbLe(down_leaves, lgap)]

    # ---------------------------------------
    # Collect all constraints
    # ---------------------------------------
    aone = all_equal_but_a_few(features, [vars1, vars2], n_features)

    all_cons = ord_bits_cons + cs1 + cs2 + aone + prop

    tic = time.perf_counter()
    if args.solver == "z3" or args.solver == "naive_z3":
        m = solve(all_cons)
    elif args.solver == "rounding":
        m = roundingSolve(all_cons)
    elif args.solver == "roundingsoplex":
        m = roundingSolve(all_cons, soplex=True)
    toc = time.perf_counter()
    solvingtime = toc - tic

    if m:
        d1 = []
        d2 = []
        for idx in range(0, model.num_features()):
            if len(split_bit_map[idx]) == 0:
                v1 = 0
                v2 = 0
            else:
                v1 = f"f{idx}_Last"
                next_v1 = f""
                breaknext = False
                for fname in split_bit_map[idx]:
                    if breaknext:
                        next_v1 = fname
                        break
                    if args.solver == "z3" or args.solver == "naive_z3":
                        cond = z3.is_true(m[vars1[fname]])
                    else:
                        cond = m[vars1[fname]]
                    if cond:
                        v1 = fname
                        breaknext = True
                v2 = f"f{idx}_Last"
                next_v2 = ""
                breaknext = False
                for fname in split_bit_map[idx]:
                    if breaknext:
                        next_v2 = fname
                        break
                    if args.solver == "z3" or args.solver == "naive_z3":
                        cond = z3.is_true(m[vars2[fname]])
                    else:
                        cond = m[vars2[fname]]
                    if cond:
                        v2 = fname
                        breaknext = True
                v1 = split_value_map[v1]
                v2 = split_value_map[v2]
            d1.append(v1)
            d2.append(v2)
        return [d1, d2], solvingtime
    else:
        return None, solvingtime


def main(args):
    global encoding
    if args.solver == "naive_z3":
        encoding = "allsum"
    else:
        encoding = "pb"

    if args.sure_counterexamples:
        sureofcounter = True

    # Accessing arguments
    solver = args.solver
    close = args.close
    gap = args.gap
    precision = args.precision
    max_trees = args.max_trees
    features = args.features
    debug = args.debug

    
    if args.filenum.isdigit():
        model_file = model_files[int(args.filenum)]
    else:
        model_file = args.filenum
    mfile = model_file

    model, trees, n_trees, n_features, n_classes = open_model_xgb(
        mfile, max_trees
    )

    feature_names = [""] * n_features
    op_range_list = [(i.min()-1, i.max()+1) for i in (trees[trees['Feature'] == f'f{j}']['Split'] for j in list(range(n_features)))]
    if args.details:
        details_fname = args.details
        if not os.path.exists(details_fname):
            print(f"Missing : {details_fname}")
            exit()
        details = pd.read_csv(details_fname)
        for index, row in details.iterrows():
            i = row["feature"]
            feature_names[i] = row["name"]
            op_range_list[i] = (row["lb"], row["ub"])

    def runner(tupl):
        f = tupl[0]
        precision = tupl[1]
        gap = tupl[2]
        start_time = time.time()
        result, solvingtime = search_anomaly_for_features(
            f,
            gap,
            precision,
            n_classes,
            model,
            trees,
            n_trees,
            args,
            op_range_list,
        )
        if result != None:
            if max_trees is not None and max_trees > 0:
                vals = model.predict(xgb.DMatrix(result), ntree_limit=max_trees)
            else:
                vals = model.predict(xgb.DMatrix(result))
            result_copy = result[0].copy()
            for x in f:
                result[0][x] = (result[0][x], result[1][x])
            print(f, precision, int(time.time() - start_time), vals, result[0])

            if args.plot:
                plot_variations(
                    model, result_copy, f, trees, feature_names, op_range_list
                )

        else:
            print(f, precision, int(time.time() - start_time), "Insensitive")

    if features is None:
        features = [0]
    tasks = [(features, precision, gap)]
    if args.all_single:
        tasks = [([f], precision, gap) for f in range(0, n_features)]
    else:
        tasks = [(features, precision, gap)]
    # if debug:
    results = [runner(params) for params in tasks]
    # else:
    #     results = Parallel(n_jobs=-1, timeout=60*60)( delayed(runner)(params) for params in tqdm(tasks) )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find sensitivity on any single feature"
    )
    parser.add_argument(
        "filenum",
        help="An integer file number. (Look in utils.py for list of files) or a filename",
    )

    # Add the 'solver' argument with choices
    parser.add_argument(
        "--solver",
        choices=["z3", "naive_z3", "rounding", "roundingsoplex", "veritas"],
        help="The solver to use. Choose either 'z3' or 'rounding'.",
    )

    # Add the 'close' argument which is a boolean (true/false)
    parser.add_argument(
        "--close",
        type=lambda x: x.lower() in ("true", "1"),
        default=False,
        help="Close option, either 'true' or 'false'. Default is 'false'.",
    )
    parser.add_argument(
        "--max_trees",
        type=int,
        default=None,
        help="Maximum number of trees to consider",
    )
    parser.add_argument(
        "--stop",
        action="store_true",
        help="whether to stop when the range of a node becomes less than a threshold",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Run serially and stop on pdb statements"
    )
    parser.add_argument(
        "--stop_param",
        type=float,
        default=0.1,
        help="Tunes how aggresssively we fold nodes",
    )
    parser.add_argument(
        "--all_single", action="store_true", help="run on all singular feature sets"
    )

    for feature in [
        "small_change",
    ]:
        parser.add_argument(f"--{feature}", action="store_true", default=False)
        parser.add_argument(f"--no-{feature}", dest="feature", action="store_false")

    parser.add_argument("--plot", action="store_true", help="plot the results")

    parser.add_argument(
        "--sure_counterexamples",
        action="store_true",
        help="Be sure about counterexamples and unsure about fairness",
    )
    parser.add_argument(
        "--gap", type=float, default=1, help="Gap for checking sensitivity"
    )
    parser.add_argument(
        "--precision", type=float, default=100, help="Scale for checking sensitivity"
    )
    parser.add_argument(
        "--features",
        type=int,
        nargs="+",
        default=None,
        help="Indexes of the features for which to do sensitivity analysis",
    )

    parser.add_argument(
        "--details",
        type=str,
        default=None,
        help="File containing names of features and their bounds",
    )
    parser.add_argument(
        "--time",
        type=float,
        default=1e8,
        help="Stopping time (in seconds), only for veritas",
    )

    # Parse the arguments
    args = parser.parse_args()
    if args.solver == "veritas":
        veritas_solver(args)
    else:
        main(args)
