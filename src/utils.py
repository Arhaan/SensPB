import xgboost as xgb
import json
import os
import pickle
import pandas as pd

model_files = [
    #  '../models/tree_verification_models/binary_mnist_unrobust/1000.resaved.json',
    # '../models/tree_verification_models/ori_mnist_robust_new/0200.resaved.json',
    # '../models/tree_verification_models/ori_mnist_unrobust_new/0200.resaved.json',
    # '../models/tree_verification_models/covtype_robust/0080.resaved.json',
    # '../models/tree_verification_models/covtype_unrobust/0080.resaved.json',
    # '../models/tree_verification_models/fashion_robust_new/0200.resaved.json',
    # '../models/tree_verification_models/fashion_unrobust_new/0200.resaved.json',
    # '../models/tree_verification_models/webspam_robust_new/0100.resaved.json',
    # '../models/tree_verification_models/webspam_unrobust_new/0100.resaved.json',
    "../models/tree_verification_models/breast_cancer_robust/0004.resaved.json",
    "../models/tree_verification_models/breast_cancer_unrobust/0004.resaved.json",
    "../models/tree_verification_models/diabetes_robust/0020.resaved.json",
    "../models/tree_verification_models/diabetes_unrobust/0020.resaved.json",
    "../models/tree_verification_models/cod-rna_unrobust/0080.resaved.json",
    "../models/tree_verification_models/binary_mnist_robust/1000.resaved.json",
    # '../models/tree_verification_models/higgs_robust/0300.resaved.json',
    "../models/tree_verification_models/higgs_unrobust/0300.resaved.json",
    "../models/tree_verification_models/ijcnn_robust_new/0060.resaved.json",
    # '../models/tree_verification_models/ijcnn_unrobust_new/0060.resaved.json',
    # 'smallmodel.pkl',
    # 'selftrained_model1.pkl',
    # 'self_model2.pkl',
    # '../models/rf_mnist_100_6.pkl',
    # '../models/new_model_200_6-0.pkl',
    # '../models/rf_db_50_6.pkl',
    # '../models/rf_db_75_6.pkl',
    # '../models/rf_db_100_6.pkl',
    "../models/new_model_5_6.pkl",
    "../models/new_model_10_6.pkl",
    "../models/new_model_20_6.pkl",
    "../models/new_model_25_6.pkl",
    "../models/new_model_35_6.pkl",
    "../models/new_model_50_6.pkl",
    "../models/new_model_65_6.pkl",
    "../models/new_model_75_6.pkl",
    "../models/new_model_80_6.pkl",
    "../models/new_model_100_6.pkl",
    "../models/new_model_125_6.pkl",
    "../models/new_model_150_6.pkl",
    "../models/new_model_175_6.pkl",
    "../models/new_model_200_6.pkl",
    # '../models/mult_feat_100.pkl'
    # '../models/mult_feat_75.pkl',
    # '../models/mult_feat_50.pkl',
    # '../models/mult_feat_20.pkl',
    # '../models/mult_feat_15.pkl',
    # '../models/mult_feat_10.pkl',
]


def get_bench_info(benchidx: int):
    """
    Returns ntrees, benchname, nfeat, depth
    """
    benchname = model_files[benchidx].split("/")[-2]
    if benchname == "models":
        benchname = model_files[benchidx].split("/")[-1].split(".")[0]
    model = xgb.Booster({"nthread": 4})  # init model
    try:
        model = pickle.load(open(model_files[benchidx], "rb"))
    except:
        model.load_model(model_files[benchidx])  # load data
    # dump_dotty(model)
    ntrees = model.num_boosted_rounds()
    nfeat = model.num_features()
    dump = model.get_dump(with_stats=True)
    tree_depths = []
    for tree in dump:
        lines = tree.split("\n")
        # The depth of the tree is the maximum number of tabs (representing levels) in any line
        max_depth = max(line.count("\t") for line in lines if line.strip() != "")
        tree_depths.append(max_depth)
    depth = max(tree_depths)
    return ntrees, benchname, nfeat, depth


basepath = os.path.dirname(__file__)
model_files = list(map(lambda x: basepath + "/" + x, model_files))


def open_model(
    model_file, max_trees=None, model_library=0, veritas=False, multiclass=False
):
    if model_library == 0:
        return open_model_xgb(model_file, max_trees, veritas=veritas)


def open_model_xgb(model_file, max_trees=None, max_classes=None, veritas=False):
    if model_file.isdigit():
        model_file = model_files[int(model_file)]
    model = xgb.Booster({"nthread": 4})  # init model
    try:
        model = pickle.load(open(model_file, "rb"))
    except:
        model.load_model(model_file)  # load data
    # dump_dotty(model)
    # model = model.get_booster()
    if veritas:
        print("reaching")
        return model, 0, 0, 0, 0
    try:
        model = model.get_booster()
    except:
        pass
    model.orig_f_names = model.feature_names
    if model.feature_names is not None:
        model.feature_names = [f"f{i}" for i in range(len(model.feature_names))]
    model.feature_names = None

    trees = model.trees_to_dataframe()
    n_trees = model.num_boosted_rounds()
    n_features = model.num_features()
    dump = model.get_dump(with_stats=True)
    num_classes = len(dump) // (n_trees)

    if num_classes != 1:
        trees["class"] = trees["Tree"] % num_classes
        trees["Tree"] = trees["Tree"] // num_classes
    else:
        trees["class"] = 0

    tree_depths = []
    for tree in dump:
        lines = tree.split("\n")
        # The depth of the tree is the maximum number of tabs (representing levels) in any line
        max_depth = max(line.count("\t") for line in lines if line.strip() != "")
        tree_depths.append(max_depth)
    depth = max(tree_depths)
    if max_trees is not None and max_trees != -1 and n_trees > max_trees:
        trees = trees[trees["Tree"] < max_trees]
        n_trees = max_trees  # TODO: This does not edit the model, so final solving might not show any unfairness
    # Comment out before running
    if max_classes is not None and max_classes != -1 and num_classes > max_classes:
        max_classes = 3
        trees = trees[trees["class"] < max_classes]
        num_classes = max_classes
    data = [
        ("model name", model_file),
        ("trees per class", n_trees),
        ("number of classes", num_classes),
        ("number of features", n_features),
        ("max depth", depth),
        ("all trees", len(dump)),
    ]

    for label, value in data:
        print(f"{label}: {value}")

    return model, trees, n_trees, n_features, num_classes


def resave_model(model_file):
    # outfile = model_file[:-5]+'resaved.model'
    # os.system(f"rm {outfile}")
    outfile = model_file[:-5] + "resaved.json"
    model = xgb.Booster({"nthread": 4})  # init model
    model.load_model(model_file)  # load data
    model.save_model(outfile)  # load data
    print(model_file, outfile)



