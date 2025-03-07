from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

import tabmini
from tabmini.estimators import XGBoost, CatBoost
from tabmini.estimators.TabR import TabR
from tabmini.estimators.MLP_PLR import MLP_PLR
from tabmini.estimators.TabTransfromer import TabTransformer
from tabmini.types import TabminiDataset
from sklearn.model_selection import train_test_split
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="TabMini-Classification Options.")
    parser.add_argument(
        "--model",
        type=int,
        # choices = [1, 4, 6, 8 , 11],
        choices=[1, 3, 7, 8, 9],
        default= 999,
        help="Type of model (1: XGBoost, 3: CatBoost, 7: MLP-PLR, 8: TabR, 9: TabTransformer)",
    )
    # parser.add_argument(
    #     "--selection", 
    #     type = bool, 
    #     default= False, 
    #     help= "Implement feature selections or not."
    # )
    # parser.add_argument(
    #     "--scale", 
    #     type= bool, 
    #     default= False, 
    #     help= "Apply Standard Scaler or not."
    # )
    parser.add_argument(
        "--save_dir", type=str, default="result_dir", help="Folder to save result."
    )
    return parser.parse_args()

def get_model(model_id):
    models = {
        1: "XGBoost",
        3: "CatBoost",
        7: "MLP-PLR",
        8: "TabR",
        9: "TabTransformer",
    }
    return models.get(model_id, "invalid")

def main(args):
    working_directory = Path.cwd() / "working_dir" / get_model(args.model)
    working_directory.mkdir(parents=True, exist_ok=True)

    # define pipeline
    method_name = "Logistic Regression"
    pipe = Pipeline(
        [
            ("scaling", MinMaxScaler()),
            ("classify", LogisticRegression(random_state=42)),
        ]
    )

    # define hyperparameters
    REGULARIZATION_OPTIONS = ["l2"]
    LAMBDA_OPTIONS = [0.5, 0.01, 0.002, 0.0004]
    param_grid = [
        {
            "classify__penalty": REGULARIZATION_OPTIONS,
            "classify__C": LAMBDA_OPTIONS,
        }
    ]



    # inner cross-validation for logistic regression
    estimator = GridSearchCV(pipe, param_grid=param_grid, cv=3, scoring="neg_log_loss", n_jobs=-1)

    # load dataset
    dataset: TabminiDataset = tabmini.load_dataset()

    dataset_name_lst = list(dataset.keys())

    for dt_name in dataset_name_lst:
        X, y = dataset[dt_name]
        if 2 in y.values:
            y = (y == 2).astype(int)
        num_records = len(X)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        if args.model == 1:
            model = XGBoost()
        elif args.model == 3:
            model = CatBoost()
        elif args.model == 7:
            model = MLP_PLR()
        elif args.model == 8:
            model = TabR()
        elif args.model == 8:
            model = TabTransformer()
        else:
            print("Sai argument roi")
            return
        model.train_and_evaluate(X_train, y_train, X_test, y_test, working_directory / f"{dt_name}_{num_records}.csv")
        # model.fit(X_test, y_test)
        # model.save_results(filename=working_directory / f"{dt_name}_{num_records}.csv")
        
if __name__ == "__main__":
    args = parse_arguments()
    main(args)


# import tabmini
# import argparse
# from pathlib import Path
# # from tabmini.data.data_processing import DataProcessor
# from tabmini.estimators import XGBoost
# # from tabmini.estimators.RF import RandomForest
# # from tabmini.estimators.TabR import TabRClassifier
# from tabmini.types import TabminiDataset
# # from tabmini.estimators.FTTransformer import FTTransformerClassifier
# # from tabmini.estimators.SAINT import SAINTClassifier
# from sklearn.model_selection import train_test_split


# def parse_arguments():
#     parser = argparse.ArgumentParser(description="TabMini-Classification Options.")
#     parser.add_argument(
#         "--model",
#         type=int,
#         #choices=[1, 2, 4, 8, 10],
#         choices = [1, 4, 6, 8 , 11],
#         default= 10,
#         help="Type of model (1: XGBoost, 4: Random Forest, 6: FTTransformer, 8: TabR, 11: SAINTabNet)",
#     )
#     parser.add_argument(
#         "--selection", 
#         type = bool, 
#         default= False, 
#         help= "Implement feature selections or not."
#     )
#     parser.add_argument(
#         "--scale", 
#         type= bool, 
#         default= False, 
#         help= "Apply Standard Scaler or not."
#     )
#     parser.add_argument(
#         "--save_dir", type=str, default="result", help="Folder to save result."
#     )
#     return parser.parse_args()


# def main(args):

#     working_directory = Path.cwd() / args.save_dir
#     working_directory.mkdir(parents=True, exist_ok=True)

#     # load dataset
#     # data_processor = DataProcessor()
#     dataset: TabminiDataset = tabmini.load_dataset()
#     dataset_name_lst = list(dataset.keys())

#     # process
#     for dt_name in dataset_name_lst:
#         X, y = dataset[dt_name]
#         if 2 in y.values:
#             y = (y == 2).astype(int)
#         num_records = len(X)

#         # preprocessing data 
#         # if args.selection: 
#         #     X = data_processor.feature_selection(X, y) 
#         # if args.scale: 
#         #     X = data_processor.normalize_data(X)
        
#         # X_train, X_test, y_train, y_test = split_train_test(X, y)
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#         # train and predict        
#         if args.model == 1:
#             model = XGBoost(small_dataset=True)
#         else:
#             print("Sai argument roi")
#         # elif args.model == 4:
#         #     model = RandomForest(small_dataset=True)
#         # elif args.model == 6:
#         #     model = FTTransformerClassifier(small_dataset= True) 
#         # elif args.model == 8: 
#         #     model = TabRClassifier(small_dataset=True)
#         # elif args.model == 11: 
#         #     model = SAINTClassifier(small_dataset = True)
#         model.fit(X_train, y_train, X_test, y_test)
#         model.save_results(filename=working_directory / f"{dt_name}_{num_records}.csv")


# if __name__ == "__main__":
#     args = parse_arguments()
#     main(args)