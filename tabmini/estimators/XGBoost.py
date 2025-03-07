import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_X_y
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.validation import check_is_fitted, check_array
from xgboost import XGBClassifier
from itertools import product
import pandas as pd

class XGBoost(BaseEstimator, ClassifierMixin):
    """A scikit-learn compatible estimator that uses XGBoost to fit and predict data."""

    def __init__(
            self,
            time_limit: int = 3600,
            device: str = "cpu",
            seed: int = 42,
            kwargs: dict = {}
    ):
        self.time_limit = time_limit
        self.device = device
        self.seed = seed
        self.kwargs = kwargs
        
        # specify that this is a binary classifier
        self.n_classes_ = 2
        self.classes_ = [0, 1]

    def train_and_evaluate(self, x_train, y_train, x_test, y_test, save_dir):
        x_train_check, y_train_check = check_X_y(x_train, y_train, accept_sparse=True)
        
        results = []
        # We do the experiment with these hyper parameter
        # + n_estimators
        # + learning_rate

        n_estimators = [50, 100, 200]
        learning_rates = [0.01, 0.05, 0.1]
        max_depths = [4, 10]
        
        # Generate all combinations
        combinations = list(product(n_estimators, learning_rates, max_depths))
        
        for combo in combinations:
            n_estimator = combo[0]
            learning_rate = combo[1]
            max_depth = combo[2]
            print(f"Do the experiment on these parameter: n_estimator: {n_estimator}, learning_rate: {learning_rate}, max_depth: {max_depth}")

            xgb = XGBClassifier(
                n_estimators=n_estimator,
                max_depth=max_depth,
                learning_rate=learning_rate,
                objective='binary:logistic',
                eval_metric='auc',
                use_label_encoder=False,
                random_state=42,
            )

            # Training
            model = xgb.fit(x_train_check, y_train_check)

            # Inference
            y_inference = model.predict(x_test)

            # Calculate 
            f1 = f1_score(y_test, y_inference, average="binary")
            acc = accuracy_score(y_test, y_inference)
            results.append({"params": f'{n_estimator}-{max_depth}-{learning_rate}',"accuracy": acc, "f1_score": f1})
        
        result_df = pd.DataFrame(results)
        result_df.to_csv(save_dir, index=False)