from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pytorch_tabular import TabularModel
from pytorch_tabular.models import TabTransformerConfig
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import f1_score, accuracy_score
from sklearn.utils import check_X_y
from sklearn.utils.validation import check_array
from pytorch_tabular.config import TrainerConfig, DataConfig


class TabTransformerClassifier(BaseEstimator, ClassifierMixin):
    """A scikit-learn compatible estimator that uses TabTransformer to fit and predict data."""

    def __init__(
            self,
            epochs: int = 20,
            batch_size: int = 64,
            learning_rate: float = 0.001,
            seed: int = 0,
            device: str = "cpu",
            kwargs: dict = {}
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device
        self.seed = seed
        self.kwargs = kwargs
        self.model = None

        # specify that this is a binary classifier
        self.n_classes_ = 2
        self.classes_ = [0, 1]

    def train_and_evaluate(self, x_train, y_train, x_test, y_test, save_dir):
        x_train_check, y_train_check = check_X_y(x_train, y_train, accept_sparse=True)
        
        results = []
        # Hyperparameter tuning grid
        embedding_dims = [16, 32]
        attn_heads = [4, 8]
        depth = [2, 4]
        
        # Generate all combinations
        combinations = list(product(embedding_dims, attn_heads, depth))
        
        for combo in combinations:
            embed_dim, heads, depth = combo
            print(f"Training with embed_dim: {embed_dim}, attn_heads: {heads}, depth: {depth}")
            
            # Configurations
            data_config = DataConfig(
                target=['target'],
                continuous_cols=list(x_train.columns),
                categorical_cols=[]
            )
            
            model_config = TabTransformerConfig(
                task="classification",
                learning_rate=self.learning_rate,
                embedding_dim=embed_dim,
                attn_heads=heads,
                attn_dropout=0.1,
                num_attn_blocks=depth,
                seed=self.seed,
                metrics=["accuracy", "f1"]
            )
            
            trainer_config = TrainerConfig(auto_lr_find=True, max_epochs=self.epochs)
            
            # Initialize and train model
            model = TabularModel(
                data_config=data_config,
                model_config=model_config,
                trainer_config=trainer_config
            )
            
            df_train = pd.DataFrame(x_train, columns=x_train.columns)
            df_train['target'] = y_train
            df_test = pd.DataFrame(x_test, columns=x_test.columns)
            df_test['target'] = y_test
            
            model.fit(train=df_train, test=df_test)
            
            # Inference
            preds = model.predict(df_test)
            y_pred = preds["prediction"]
            
            # Calculate metrics
            f1 = f1_score(y_test, y_pred, average="binary")
            acc = accuracy_score(y_test, y_pred)
            results.append({"params": f'{embed_dim}-{heads}-{depth}', "accuracy": acc, "f1_score": f1})
        
        result_df = pd.DataFrame(results)
        result_df.to_csv(save_dir, index=False)
