from itertools import product
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import f1_score, accuracy_score
from sklearn.utils import check_X_y
from sklearn.utils.validation import check_is_fitted, check_array


class TabR(BaseEstimator, ClassifierMixin):
    """A scikit-learn compatible estimator that uses CatBoost to fit and predict data."""

    def __init__(
            self,
            time_limit: int = 3600,
            device: str = "cpu",
            seed: int = 0,
            kwargs: dict = {}
    ):
        self.time_limit = time_limit
        self.device = device
        self.kwargs = kwargs
        self.seed = seed

        # specify that this is a binary classifier
        self.n_classes_ = 2
        self.classes_ = [0, 1]

    def train_and_evaluate(self, x_train, y_train, x_test, y_test, save_dir):
        x_train_check, y_train_check = check_X_y(x_train, y_train, accept_sparse=True)
        
        results = []
        # We do the experiment with these hyper parameter
        # + n_estimators
        # + learning_rate

        iterations = [50, 100, 200]
        learning_rate_sample = [0.1, 0.3]
        depth = [4, 8, 16]
        # Generate all combinations
        combinations = list(product(iterations, learning_rate_sample, depth))
        
        for combo in combinations:
            iterations = combo[0]
            learning_rate = combo[1]
            depth = combo[2]
            print(f"Do the experiment on these parameter: iterations: {iterations}, learning_rate: {learning_rate}, depth: {depth}")
            
            catboost = CatBoostClassifier(
                iterations=iterations,
                depth=depth,
                learning_rate=learning_rate,
                loss_function='Logloss',
                eval_metric='AUC',
                task_type='GPU' if self.device == 'cuda' else 'CPU',
                devices='0',
                random_seed=self.seed,
                **self.kwargs
            )

            # Training
            model = catboost.fit(x_train_check, y_train_check)

            # Inference
            y_inference = model.predict(x_test)

            # Calculate 
            f1 = f1_score(y_test, y_inference, average="binary")
            acc = accuracy_score(y_test, y_inference)
            results.append({"params": f'{iterations}-{learning_rate}-{depth}',"accuracy": acc, "f1_score": f1})
        
        result_df = pd.DataFrame(results)
        result_df.to_csv(save_dir, index=False)