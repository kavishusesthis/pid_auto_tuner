"""Small multi-output Decision Tree Regressor implemented without ML libraries."""


class DecisionTreeRegressor:
    def __init__(self, max_depth=5, min_samples_leaf=3):
        self.max_depth = int(max_depth)
        self.min_samples_leaf = int(min_samples_leaf)
        self.tree = None
        self.n_features = 0

    @staticmethod
    def _mean(rows):
        width = len(rows[0])
        return [sum(row[j] for row in rows) / len(rows) for j in range(width)]

    @staticmethod
    def _sse(rows):
        if not rows:
            return 0.0
        mean = DecisionTreeRegressor._mean(rows)
        return sum(
            sum((value - mean[j]) ** 2 for j, value in enumerate(row))
            for row in rows
        )

    def fit(self, X, y):
        if not X or len(X) != len(y):
            raise ValueError("X and y must be non-empty and have equal row counts.")
        if any(len(row) != len(X[0]) for row in X):
            raise ValueError("All feature rows must have the same length.")
        if any(len(row) != len(y[0]) for row in y):
            raise ValueError("All target rows must have the same length.")
        if self.max_depth < 1 or self.min_samples_leaf < 1:
            raise ValueError("max_depth and min_samples_leaf must be positive.")

        self.n_features = len(X[0])
        rows = [(list(map(float, features)), list(map(float, target))) for features, target in zip(X, y)]
        self.tree = self._build(rows, depth=0)
        return self

    def _build(self, rows, depth):
        targets = [row[1] for row in rows]
        leaf_value = self._mean(targets)
        node = {"value": leaf_value}
        if depth >= self.max_depth or len(rows) < 2 * self.min_samples_leaf:
            return node

        parent_error = self._sse(targets)
        best = None
        best_error = parent_error

        for feature_index in range(self.n_features):
            values = sorted(set(row[0][feature_index] for row in rows))
            if len(values) < 2:
                continue
            thresholds = [(values[i] + values[i + 1]) / 2.0 for i in range(len(values) - 1)]
            for threshold in thresholds:
                left = [row for row in rows if row[0][feature_index] <= threshold]
                right = [row for row in rows if row[0][feature_index] > threshold]
                if len(left) < self.min_samples_leaf or len(right) < self.min_samples_leaf:
                    continue
                error = self._sse([r[1] for r in left]) + self._sse([r[1] for r in right])
                if error < best_error - 1e-12:
                    best_error = error
                    best = (feature_index, threshold, left, right)

        if best is None:
            return node

        feature_index, threshold, left, right = best
        return {
            "feature_index": feature_index,
            "threshold": threshold,
            "value": leaf_value,
            "left": self._build(left, depth + 1),
            "right": self._build(right, depth + 1),
        }

    def predict_one(self, features):
        if self.tree is None:
            raise RuntimeError("Model has not been trained or loaded.")
        if len(features) != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {len(features)}.")
        node = self.tree
        while "feature_index" in node:
            if float(features[node["feature_index"]]) <= node["threshold"]:
                node = node["left"]
            else:
                node = node["right"]
        return list(node["value"])

    def predict(self, X):
        return [self.predict_one(row) for row in X]

    def to_dict(self):
        if self.tree is None:
            raise RuntimeError("Cannot save an untrained model.")
        return {
            "model_type": "DecisionTreeRegressor_from_scratch",
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "n_features": self.n_features,
            "tree": self.tree,
        }

    @classmethod
    def from_dict(cls, payload):
        if payload.get("model_type") != "DecisionTreeRegressor_from_scratch":
            raise ValueError("Unsupported model file.")
        model = cls(payload["max_depth"], payload["min_samples_leaf"])
        model.n_features = int(payload["n_features"])
        model.tree = payload["tree"]
        return model
