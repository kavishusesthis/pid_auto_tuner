"""Basic smoke test for the from-scratch multi-output regression tree."""
from decision_tree import DecisionTreeRegressor


def main():
    X = [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]]
    y = [[0.0, 0.0], [0.1, 0.2], [0.2, 0.4], [1.0, 2.0], [1.1, 2.2], [1.2, 2.4]]
    model = DecisionTreeRegressor(max_depth=3, min_samples_leaf=1).fit(X, y)
    low = model.predict_one([0.5])
    high = model.predict_one([4.5])
    assert len(low) == 2 and len(high) == 2
    assert high[0] > low[0], (low, high)
    restored = DecisionTreeRegressor.from_dict(model.to_dict())
    assert restored.predict_one([4.5]) == high
    print("Decision Tree smoke test passed.")


if __name__ == "__main__":
    main()
