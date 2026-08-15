import re
import math
import statistics
import random
from collections import Counter, defaultdict
import json
from datetime import datetime


class TreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None, is_leaf=False):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.is_leaf = is_leaf

    def to_dict(self):
        if self.is_leaf:
            return {"leaf": True, "value": self.value}
        return {
            "feature": self.feature,
            "threshold": self.threshold,
            "left": self.left.to_dict() if self.left else None,
            "right": self.right.to_dict() if self.right else None,
        }


def _variance(arr):
    if len(arr) < 2:
        return 0
    mean = sum(arr) / len(arr)
    return sum((x - mean) ** 2 for x in arr) / len(arr)


def _entropy(labels):
    counts = Counter(labels)
    total = len(labels)
    ent = 0
    for count in counts.values():
        if count > 0:
            p = count / total
            ent -= p * math.log2(p)
    return ent


def _gini(labels):
    counts = Counter(labels)
    total = len(labels)
    gini = 1.0
    for count in counts.values():
        p = count / total
        gini -= p ** 2
    return gini


def _information_gain(X, y, feature_idx, threshold):
    left_y = [y[i] for i in range(len(y)) if X[i][feature_idx] <= threshold]
    right_y = [y[i] for i in range(len(y)) if X[i][feature_idx] > threshold]

    if not left_y or not right_y:
        return 0

    parent_impurity = _entropy(y)
    n = len(y)
    left_weight = len(left_y) / n
    right_weight = len(right_y) / n

    child_impurity = left_weight * _entropy(left_y) + right_weight * _entropy(right_y)
    return parent_impurity - child_impurity


def _best_split(X, y, min_samples_split=2, min_samples_leaf=1):
    n_features = len(X[0]) if X else 0
    best_gain = -1
    best_feature = None
    best_threshold = None

    for f in range(n_features):
        values = sorted(set(X[i][f] for i in range(len(X))))
        thresholds = []
        for j in range(len(values) - 1):
            thresholds.append((values[j] + values[j + 1]) / 2)

        if not thresholds:
            thresholds = values

        for t in thresholds:
            left_idx = [i for i in range(len(X)) if X[i][f] <= t]
            right_idx = [i for i in range(len(X)) if X[i][f] > t]

            if len(left_idx) < min_samples_leaf or len(right_idx) < min_samples_leaf:
                continue

            gain = _information_gain(X, y, f, t)
            if gain > best_gain:
                best_gain = gain
                best_feature = f
                best_threshold = t

    return best_feature, best_threshold, best_gain


def _build_tree(X, y, depth=0, max_depth=10, min_samples_split=2, min_samples_leaf=1):
    if (depth >= max_depth or
            len(y) < min_samples_split or
            len(set(y)) == 1):
        value = sum(y) / len(y) if y else 0
        return TreeNode(value=value, is_leaf=True)

    feature, threshold, gain = _best_split(X, y, min_samples_split, min_samples_leaf)

    if feature is None or gain <= 0:
        value = sum(y) / len(y) if y else 0
        return TreeNode(value=value, is_leaf=True)

    left_X = [X[i] for i in range(len(X)) if X[i][feature] <= threshold]
    left_y = [y[i] for i in range(len(y)) if X[i][feature] <= threshold]
    right_X = [X[i] for i in range(len(X)) if X[i][feature] > threshold]
    right_y = [y[i] for i in range(len(y)) if X[i][feature] > threshold]

    left = _build_tree(left_X, left_y, depth + 1, max_depth, min_samples_split, min_samples_leaf)
    right = _build_tree(right_X, right_y, depth + 1, max_depth, min_samples_split, min_samples_leaf)

    return TreeNode(feature=feature, threshold=threshold, left=left, right=right)


def _predict_one(node, row):
    if node.is_leaf:
        return node.value
    if row[node.feature] <= node.threshold:
        return _predict_one(node.left, row)
    else:
        return _predict_one(node.right, row)


def _predict(tree, X):
    return [_predict_one(tree, row) for row in X]


def _feature_importance(tree, n_features):
    importance = [0.0] * n_features

    def _traverse(node):
        if node.is_leaf:
            return
        importance[node.feature] += 1
        _traverse(node.left)
        _traverse(node.right)

    _traverse(tree)

    total = sum(importance) or 1
    return {i: round(importance[i] / total, 4) for i in range(n_features)}


def _get_depth(node):
    if node.is_leaf:
        return 0
    return 1 + max(_get_depth(node.left), _get_depth(node.right))


def _get_leaf_count(node):
    if node.is_leaf:
        return 1
    return _get_leaf_count(node.left) + _get_leaf_count(node.right)


def _cross_validate(X, y, k=5, max_depth=10, min_samples_split=2):
    n = len(X)
    if n < k:
        k = n
    fold_size = n // k
    indices = list(range(n))
    random.shuffle(indices)

    scores = []
    for i in range(k):
        test_idx = indices[i * fold_size:(i + 1) * fold_size]
        train_idx = indices[:i * fold_size] + indices[(i + 1) * fold_size:]

        train_X = [X[j] for j in train_idx]
        train_y = [y[j] for j in train_idx]
        test_X = [X[j] for j in test_idx]
        test_y = [y[j] for j in test_idx]

        tree = _build_tree(train_X, train_y, max_depth=max_depth, min_samples_split=min_samples_split)
        predictions = _predict(tree, test_X)

        mse = sum((predictions[i] - test_y[i]) ** 2 for i in range(len(test_y))) / max(1, len(test_y))
        ss_res = sum((predictions[i] - test_y[i]) ** 2 for i in range(len(test_y)))
        ss_tot = sum((test_y[i] - statistics.mean(test_y)) ** 2 for i in range(len(test_y))) if len(test_y) > 1 else 1
        r2 = 1 - (ss_res / max(1, ss_tot))

        scores.append({"mse": round(mse, 4), "r2": round(max(0, r2), 4)})

    avg_mse = statistics.mean([s["mse"] for s in scores])
    avg_r2 = statistics.mean([s["r2"] for s in scores])

    return {
        "folds": scores,
        "avg_mse": round(avg_mse, 4),
        "avg_r2": round(avg_r2, 4),
        "std_mse": round(statistics.stdev([s["mse"] for s in scores]) if len(scores) > 1 else 0, 4),
    }


def _prune(tree, X_val, y_val, min_impurity_decrease=0.01):
    if tree.is_leaf:
        return tree

    if tree.left:
        tree.left = _prune(tree.left, X_val, y_val, min_impurity_decrease)
    if tree.right:
        tree.right = _prune(tree.right, X_val, y_val, min_impurity_decrease)

    if tree.left and tree.right and tree.left.is_leaf and tree.right.is_leaf:
        preds_without = _predict(tree, X_val)
        mse_without = sum((preds_without[i] - y_val[i]) ** 2 for i in range(len(y_val))) / max(1, len(y_val))

        leaf_value = sum(y_val) / len(y_val) if y_val else 0
        preds_with = [leaf_value] * len(y_val)
        mse_with = sum((preds_with[i] - y_val[i]) ** 2 for i in range(len(y_val))) / max(1, len(y_val))

        if mse_with <= mse_without + min_impurity_decrease:
            return TreeNode(value=leaf_value, is_leaf=True)

    return tree


def _visualize_tree(node, prefix="", is_left=True, feature_names=None):
    lines = []
    if node.is_leaf:
        connector = "└── " if is_left else "┌── "
        lines.append(f"{prefix}{connector}Leaf: {node.value:.4f}")
        return lines

    feature_label = f"F{node.feature}" if feature_names is None else feature_names[node.feature]
    connector = "└── " if is_left else "┌── "
    lines.append(f"{prefix}{connector}{feature_label} <= {node.threshold:.4f}")

    new_prefix = prefix + ("    " if is_left else "│   ")
    if node.left:
        lines.extend(_visualize_tree(node.left, new_prefix, True, feature_names))
    if node.right:
        lines.extend(_visualize_tree(node.right, new_prefix, False, feature_names))

    return lines


def trainTree(data, params=None):
    if params is None:
        params = {}

    features = data.get("features", [])
    targets = data.get("targets", [])
    predict_features = data.get("predict_features")

    max_depth = params.get("maxDepth", 10)
    min_samples_split = params.get("minSamplesSplit", 2)
    min_samples_leaf = params.get("minSamplesLeaf", 1)

    if not features or not targets:
        raise ValueError("features and targets required")

    tree = _build_tree(features, targets, max_depth=max_depth,
                       min_samples_split=min_samples_split,
                       min_samples_leaf=min_samples_leaf)

    predictions = _predict(tree, predict_features) if predict_features else []

    tree_lines = _visualize_tree(tree)

    return {
        "tree_depth": _get_depth(tree),
        "leaf_count": _get_leaf_count(tree),
        "predictions": predictions,
        "training_samples": len(targets),
        "tree_structure": tree_lines,
        "timestamp": datetime.now().isoformat(),
    }


def predict(tree, features):
    return _predict(tree, features)


def processItem(input_str, params=None):
    if params is None:
        params = {}

    parsed = json.loads(input_str) if isinstance(input_str, str) else input_str
    features = parsed.get("features", [])
    targets = parsed.get("targets", [])
    predict_features = parsed.get("predict_features")

    if not features or not targets:
        raise ValueError("features and targets required")

    tree = _build_tree(features, targets,
                       max_depth=params.get("max_depth", 10),
                       min_samples_split=params.get("min_samples_split", 2))

    predictions = predict(tree, predict_features) if predict_features else []
    importance = _feature_importance(tree, len(features[0]))

    cv_result = _cross_validate(features, targets, k=min(5, len(features)),
                                max_depth=params.get("max_depth", 10))

    pruned_tree = _prune(tree, features[:max(1, len(features) // 5)],
                         targets[:max(1, len(targets) // 5)])
    pruned_predictions = predict(pruned_tree, predict_features) if predict_features else []

    return {
        "tree_depth": _get_depth(tree),
        "leaf_count": _get_leaf_count(tree),
        "predictions": predictions,
        "feature_importance": importance,
        "accuracy": cv_result,
        "tree_structure": _visualize_tree(tree),
        "pruned_nodes": _get_leaf_count(tree) - _get_leaf_count(pruned_tree),
        "pruned_predictions": pruned_predictions,
    }



if __name__ == "__main__":
    X = [[1, 2], [3, 4], [5, 6], [7, 8], [2, 3], [4, 5]]
    y = [0, 0, 1, 1, 0, 1]
    result = trainTree({"features": X, "targets": y, "predict_features": [[3, 3], [6, 7]]})
    print(json.dumps(result, indent=2))