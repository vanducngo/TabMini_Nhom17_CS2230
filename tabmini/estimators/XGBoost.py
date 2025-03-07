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

        n_estimators_sample = [50, 100, 200]
        learning_rate_sample = [0.05, 0.2, 0.5]
        # Generate all combinations
        combinations = list(product(n_estimators_sample, learning_rate_sample))
        
        for combo in combinations:
            print(f"Do the experiment on these parameter: n_estimators: {combo[0]}, learning_rate: {combo[1]}")

            xgb = XGBClassifier(
                n_estimators=combo[0],
                max_depth=2,
                learning_rate=combo[1],
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
            results.append({"params": f'{combo[0]}-{combo[1]}',"accuracy": acc, "f1_score": f1})
        
        result_df = pd.DataFrame(results)
        result_df.to_csv(save_dir, index=False)