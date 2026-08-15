#!/usr/bin/env python3
import json
import argparse
from src.main import processItem as _process_item


def run(input_str: str, params: dict | None = None) -> dict:
    """Train a decision tree regressor and make predictions.

    input must contain:
      - features: 2D array of numeric features
      - targets: 1D array of numeric targets
      - predict_features (optional): 2D array to predict on
    """
    try:
        return _process_item(input_str, params or {})
    except Exception as e:
        return {"error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Decision Tree Forecast")
    parser.add_argument("--input", type=str, help="JSON string with features, targets, predict_features")
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--min-samples-split", type=int, default=2)
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    args = parser.parse_args()

    if not args.input:
        demo = json.dumps({
            "features": [[1, 2], [3, 4], [5, 6], [7, 8], [2, 3], [4, 5]],
            "targets": [0, 0, 1, 1, 0, 1],
            "predict_features": [[3, 3], [6, 7]],
        })
        args.input = demo

    params = {
        "max_depth": args.max_depth,
        "min_samples_split": args.min_samples_split,
        "min_samples_leaf": args.min_samples_leaf,
    }
    print(json.dumps(run(args.input, params), indent=2))


if __name__ == "__main__":
    main()
