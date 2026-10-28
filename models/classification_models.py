from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report


import pandas as pd
import warnings




from typing import Optional, Union


from utils.metrics.classification_metrics import ClassificationMetrics
from utils.pipelines.ClassificationPipeline import ClassificationPipeline as   Pipeline



class ClassificationModels:
    """A factory class to provide instances of various classification models for tabular data."""

    def __init__(self):
        pass

    def getXGBClassifier(self, objective: str, n_estimators: int, learning_rate: float, max_depth: int, subsample: float, colsample_bytree: float, random_state: Optional[int] = None) -> XGBClassifier:
        """
        Returns an XGBoost Classifier model with specified hyperparameters.

        Parameters:
            objective (str): Learning objective (e.g., 'binary:logistic', 'multi:softmax').
            n_estimators (int): Number of boosting rounds.
            learning_rate (float): Step size shrinkage used in updates.
            max_depth (int): Maximum depth of a tree.
            subsample (float): Subsample ratio of the training instance.
            colsample_bytree (float): Subsample ratio of columns when constructing each tree.
            random_state (int | None): Controls randomness for reproducibility.

        Returns:
            XGBClassifier: Configured XGBoost classifier instance.
        """
        return XGBClassifier(
            objective=objective,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state
        )
    

    def getLGBMClassifier( self, objective: str, n_estimators: int, learning_rate: float, num_leaves: int, max_depth: int, subsample: float, colsample_bytree: float, random_state: Union[int,None] = None ) -> LGBMClassifier:
        """
        Returns a LightGBM Classifier model with specified hyperparameters.

        Parameters:
            objective (str): Learning objective (e.g., 'binary', 'multiclass').
            n_estimators (int): Number of boosting iterations.
            learning_rate (float): Boosting learning rate.
            num_leaves (int): Maximum number of leaves per tree.
            max_depth (int): Maximum tree depth.
            subsample (float): Subsample ratio of the training instance.
            colsample_bytree (float): Subsample ratio of columns when constructing each tree.
            random_state (int | None): Controls randomness for reproducibility.

        Returns:
            LGBMClassifier: Configured LightGBM classifier.
        """
        return LGBMClassifier(
            objective=objective,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state
        )


    def getMLPClassifier(self, hidden_layer_sizes: tuple[int, ...], activation: str, solver: str, max_iter: int, learning_rate_init: float, momentum: float, random_state: Union[int, None] = None ) -> MLPClassifier:
        """
        Returns a Multi-layer Perceptron Classifier with specified hyperparameters.

        Parameters:
            hidden_layer_sizes (tuple[int, ...]): Size of each hidden layer.
            activation (str): Activation function (e.g., 'relu', 'tanh', 'logistic').
            solver (str): Optimization algorithm ('adam', 'sgd', or 'lbfgs').
            max_iter (int): Maximum number of iterations.
            learning_rate_init (float): Initial learning rate.
            momentum (float): Momentum for SGD optimizer.
            random_state (int | None): Controls randomness for reproducibility.

        Returns:
            MLPClassifier: Configured neural network classifier.
        """
        return MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            max_iter=max_iter,
            learning_rate_init=learning_rate_init,
            momentum=momentum,
            random_state=random_state
        )

    def getRandomForestClassifier( self, criterion: str, max_depth: Optional[int], n_estimators: int, min_samples_split: int, min_samples_leaf: int, max_features: Union[str, int, float, None], bootstrap: bool, random_state: Union [int, None] = None ) -> RandomForestClassifier:
        """
        Returns a Random Forest Classifier model with specified hyperparameters.

        Parameters:
            criterion (str): Function to measure split quality ('gini' or 'entropy').
            max_depth (int | None): Maximum depth of the tree.
            n_estimators (int): Number of trees in the forest.
            min_samples_split (int): Minimum samples required to split an internal node.
            min_samples_leaf (int): Minimum samples required at a leaf node.
            max_features (str | int | float | None): Number of features to consider for the best split.
            bootstrap (bool): Whether bootstrap samples are used when building trees.
            random_state (int | None): Controls randomness for reproducibility.

        Returns:
            RandomForestClassifier: Configured random forest classifier.
        """
        return RandomForestClassifier(
            criterion=criterion,
            max_depth=max_depth,
            n_estimators=n_estimators,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            bootstrap=bootstrap,
            random_state=random_state
        )

    def getLogisticRegression(self, penalty: str, C: float, solver: str, max_iter: int, random_state: Optional[int] = None ) -> LogisticRegression:
        """
        Returns a Logistic Regression model with specified hyperparameters.

        Parameters:
            penalty (str): Regularization type ('l1', 'l2', 'elasticnet', or 'none').
            C (float): Inverse of regularization strength.
            solver (str): Algorithm to use in optimization ('liblinear', 'saga', 'lbfgs', etc.).
            max_iter (int): Maximum number of iterations.
            random_state (int | None): Controls randomness for reproducibility.

        Returns:
            LogisticRegression: Configured logistic regression classifier.
        """
        return LogisticRegression(
            penalty=penalty,
            C=C,
            solver=solver,
            max_iter=max_iter,
            random_state=random_state
        )

    def getSVC(self, C: float, kernel: str, degree: int, gamma: Union[str, float], probability: bool = False, random_state:Optional[int] = None ) -> SVC:
        """
        Returns a Support Vector Classifier with specified hyperparameters.

        Parameters:
            C (float): Regularization parameter.
            kernel (str): Kernel type ('linear', 'poly', 'rbf', 'sigmoid').
            degree (int): Degree for polynomial kernel.
            gamma (str | float): Kernel coefficient.
            probability (bool): Whether to enable probability estimates.
            random_state (int | None): Controls randomness for reproducibility.

        Returns:
            SVC: Configured support vector classifier.
        """
        return SVC(
            C=C,
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            probability=probability,
            random_state=random_state
        )

    def getKNeighborsClassifier( self, n_neighbors: int, weights: str, algorithm: str, leaf_size: int, p: int ) -> KNeighborsClassifier:
        """
        Returns a K-Nearest Neighbors Classifier with specified hyperparameters.

        Parameters:
            n_neighbors (int): Number of neighbors to use.
            weights (str): Weight function ('uniform' or 'distance').
            algorithm (str): Algorithm used to compute nearest neighbors ('auto', 'ball_tree', etc.).
            leaf_size (int): Leaf size for tree-based algorithms.
            p (int): Power parameter for the Minkowski metric (1 for Manhattan, 2 for Euclidean).

        Returns:
            KNeighborsClassifier: Configured k-nearest neighbors classifier.
        """
        return KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm,
            leaf_size=leaf_size,
            p=p
        )


    def getCatBoostClassifier( self, iterations: int, learning_rate: float, depth: int, loss_function: str, l2_leaf_reg: float, border_count: int, random_state: Optional [int] = None, verbose: bool = False ) -> CatBoostClassifier:
            """
            Returns a CatBoost Classifier model with specified hyperparameters.

            Parameters:
                iterations (int): Number of boosting iterations.
                learning_rate (float): Learning rate.
                depth (int): Depth of the tree.
                loss_function (str): Objective function (e.g., 'Logloss', 'MultiClass').
                l2_leaf_reg (float): L2 regularization term.
                border_count (int): Number of splits for numerical features.
                random_state (int | None): Controls randomness for reproducibility.
                verbose (bool): Whether to output training logs.

            Returns:
                CatBoostClassifier: Configured CatBoost classifier.
            """
            return CatBoostClassifier(
                iterations=iterations,
                learning_rate=learning_rate,
                depth=depth,
                loss_function=loss_function,
                l2_leaf_reg=l2_leaf_reg,
                border_count=border_count,
                random_state=random_state,
                verbose=verbose
            )
    def getBalancedBaggingClassifier(self, n_estimators: int = 10, max_samples: Union[int, float] = 1.0, max_features: Union[int, float] = 1.0, bootstrap: bool = True, bootstrap_features: bool = False, random_state: Optional[int] = None) -> BalancedBaggingClassifier:
        """
        Returns a Balanced Bagging Classifier with specified hyperparameters.

        Parameters:
            base_estimator (object | None): The base estimator to fit on random subsets of the dataset.
            n_estimators (int): The number of base estimators in the ensemble.
            max_samples (int | float): The number of samples to draw from X to train each base estimator.
            max_features (int | float): The number of features to draw from X to train each base estimator.
            bootstrap (bool): Whether samples are drawn with replacement.
            bootstrap_features (bool): Whether features are drawn with replacement.
            random_state (int | None): Controls randomness for reproducibility.

        Returns:
            BalancedBaggingClassifier: Configured balanced bagging classifier.
        """
        return BalancedBaggingClassifier(
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            bootstrap=bootstrap,
            bootstrap_features=bootstrap_features,
            random_state=random_state
        )
    def getPerceptron(self, penalty: Optional[str] = None, alpha: float = 0.0001, max_iter: int = 1000, tol: Optional[float] = 1e-3, random_state: Optional[int] = None) -> Perceptron:
        """
        Returns a Perceptron model with specified hyperparameters.

        Parameters:
            penalty (str | None): The penalty (regularization term) to be used ('l2', 'l1', 'elasticnet', or None).
            alpha (float): Constant that multiplies the regularization term.
            max_iter (int): The maximum number of passes over the training data.
            tol (float | None): The stopping criterion.
            random_state (int | None): Controls randomness for reproducibility.

        Returns:
            Perceptron: Configured perceptron classifier.
        """
        return Perceptron(
            penalty=penalty,
            alpha=alpha,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state
        )
    def getSGDClassifier(self, loss: str = 'hinge', penalty: Optional[str] = 'l2', alpha: float = 0.0001, max_iter: int = 1000, tol: Optional[float] = 1e-3, random_state: Optional[int] = None) -> SGDClassifier:
        """
        Returns a Stochastic Gradient Descent Classifier with specified hyperparameters.

        Parameters:
            loss (str): The loss function to be used ('hinge', 'log', 'modified_huber', etc.).
            penalty (str | None): The penalty (regularization term) to be used ('l2', 'l1', 'elasticnet', or None).
            alpha (float): Constant that multiplies the regularization term.
            max_iter (int): The maximum number of passes over the training data.
            tol (float | None): The stopping criterion.
            random_state (int | None): Controls randomness for reproducibility.

        Returns:
            SGDClassifier: Configured stochastic gradient descent classifier.
        """
        return SGDClassifier(
            loss=loss,
            penalty=penalty,
            alpha=alpha,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state
        )
    def getDecisionTreeClassifier(self, criterion: str = 'gini', max_depth: Optional[int] = None, min_samples_split: int = 2, min_samples_leaf: int = 1, max_features: Optional[Union[int, float, str]] = None, random_state: Optional[int] = None) -> DecisionTreeClassifier:
        """
        Returns a Decision Tree Classifier with specified hyperparameters.

        Parameters:
            criterion (str): The function to measure the quality of a split ('gini' or 'entropy').
            max_depth (int | None): The maximum depth of the tree.
            min_samples_split (int): The minimum number of samples required to split an internal node.
            min_samples_leaf (int): The minimum number of samples required to be at a leaf node.
            max_features (int | float | str | None): The number of features to consider when looking for the best split.
            random_state (int | None): Controls randomness for reproducibility.

        Returns:
            DecisionTreeClassifier: Configured decision tree classifier.
        """
        return DecisionTreeClassifier(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state
        )

    # -------------------------------------------------------------------------
    # All models getter using above builders
    # -------------------------------------------------------------------------


    def get_cls_models(self, random_state: Optional[int] = 42) -> dict[str, object]:
        """
        Returns a dictionary of all supported classification models
        using their respective builder methods and default parameters.

        Parameters:
            random_state (int | None): Controls randomness for reproducibility.

        Returns:
            dict[str, object]: Dictionary mapping model names to instantiated models.
        """
        ml_cls_models = {
            "XGBClassifier": self.getXGBClassifier(
                objective='binary:logistic',  # str: 'binary:logistic', 'multi:softmax', 'reg:squarederror', etc.
                n_estimators=100,             # int: usually 50-1000
                learning_rate=0.1,            # float: 0.01-0.3 typical
                max_depth=3,                  # int: 1-10 typical
                subsample=0.8,                # float: 0-1
                colsample_bytree=0.8,         # float: 0-1
                random_state=42               # int or None
            ),
            "LGBMClassifier": self.getLGBMClassifier(
                objective='binary',           # str: 'binary', 'multiclass', 'multiclassova'
                n_estimators=100,             # int: 50-1000
                learning_rate=0.1,            # float: 0.01-0.3
                num_leaves=31,                # int: 10-100 typical
                max_depth=-1,                 # int: -1 (no limit) or positive integer
                subsample=0.8,                # float: 0-1
                colsample_bytree=0.8,         # float: 0-1
                random_state=42               # int or None
            ),
             "CatBoostClassifier": self.getCatBoostClassifier(
                iterations=1000,              # int: 100-10000 typical
                learning_rate=0.1,            # float: 0.01-0.3
                depth=6,                      # int: 1-10
                loss_function='Logloss',      # str: 'Logloss', 'MultiClass', 'RMSE', etc.
                l2_leaf_reg=3.0,              # float: 0-10 typical
                border_count=32,              # int: 1-255
                random_state=42,              # int or None
                verbose=False                 # bool
            ),
            "RandomForestClassifier": self.getRandomForestClassifier(
                criterion='gini',             # str: 'gini' or 'entropy'
                max_depth=None,               # int or None
                n_estimators=100,             # int: 10-1000
                min_samples_split=2,          # int or float
                min_samples_leaf=1,           # int or float
                max_features='sqrt',          # int, float, 'auto', 'sqrt', 'log2', None
                bootstrap=True,               # bool
                random_state=42               # int or None
            ),
            "MLPClassifier": self.getMLPClassifier( #missing preproc step is mandatory
                hidden_layer_sizes=(100, 50), # tuple[int,...]: e.g., (50,), (100,50), etc.
                activation='relu',            # str: 'identity', 'logistic', 'tanh', 'relu'
                solver='adam',                # str: 'lbfgs', 'sgd', 'adam'
                max_iter=200,                 # int: typically 100-1000
                learning_rate_init=0.001,     # float: 0.0001-0.01
                momentum=0.9,                 # float: 0-1, only used with 'sgd'
                random_state=42               # int or None
            ),
            "LogisticRegression": self.getLogisticRegression( #missing preproc step is mandatory
                penalty='l2',                 # str: 'l1', 'l2', 'elasticnet', 'none'
                C=1.0,                        # float >0
                solver='lbfgs',               # str: 'lbfgs', 'liblinear', 'saga', 'newton-cg', 'sag'
                max_iter=100,                 # int
                random_state=42               # int or None
            ),
            "SVC": self.getSVC(
                C=1.0,                        # float >0   #missing preproc step is mandatory
                kernel='rbf',                 # str: 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
                degree=3,                     # int: only used with 'poly'
                gamma='scale',                # str or float: 'scale', 'auto', or float value
                probability=True,             # bool: True/False
                random_state=42               # int or None
            ),
            "KNeighborsClassifier": self.getKNeighborsClassifier( #missing preproc step is mandatory
                n_neighbors=5,                # int: 1-n 
                weights='uniform',            # str: 'uniform', 'distance', or callable
                algorithm='auto',             # str: 'auto', 'ball_tree', 'kd_tree', 'brute'
                leaf_size=30,                 # int: 10-50 typical
                p=2                           # int: 1 (Manhattan), 2 (Euclidean)
            ),
            "getBalancedBaggingClassifier":self.getBalancedBaggingClassifier(
                n_estimators=10,
                max_samples=1.0,
                max_features=1.0,
                bootstrap=True,
                bootstrap_features=False,
                random_state=42
            ),
            "Perceptron":self.getPerceptron(
                penalty=None,
                alpha=0.0001,
                max_iter=100,
                tol=1e-4,
                random_state=42
            ),
            "SGDClassifier":self.getSGDClassifier(
                loss='hinge',
                penalty='l2',
                alpha=0.0001,
                max_iter=1000,
                tol=1e-3,
                random_state=42
            ),
            "DecisionTreeClassifier":self.getDecisionTreeClassifier(
                criterion='gini',
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features=None,
                random_state=42
            )
        }
        return ml_cls_models  


# if __name__ == "__main__":
#     warnings.filterwarnings("ignore")
#     import argparse
#     from modules.DataLoader import DataLoader
#     loader = DataLoader()
#     cls_models = ClassificationModels()
#     metrics = ClassificationMetrics()   
#     preproc = Pipeline()

#     # define the path of the dataset from the terminal
#     parser = argparse.ArgumentParser("put the path of the dataset")
#     parser.add_argument("path", help="path of the dataset")
#     parser.add_argument("--target", help="target column")
#     parser.add_argument("--model", help="model to use")
#     args = parser.parse_args()
#     print("path of the dataset", args.path)
#     print("target column", args.target)    
    
#     #put the target column name in variable target
#     target = args.target
#     list_params = []
    
#     #must determine X, y columns for training, columns to drop and categorical columns
    
#     #df = pd.DataFrame({
#     #        "age": [25, 30, 35, np.nan, 50, 45, 40, np.nan],
#     #        "income": [50000, 60000, 80000, 200000, 120000, np.nan, 90000, 100000],
#     #        "gender": ["M", "F", "F", "M", "M", "F", "F", "M"],
#     #        "purchased": ["Yes", "No", "Yes", "No", "Yes", "No", "Yes", "Yes"]
#     #    })
#     ## Separate features and target 
#     #X = df.drop(columns=["purchased"])
#     #y = df["purchased"]
    
#     #X_ = X

#     #load the dataset
#     df = loader.load_dataset(args.path)
#     # Separate features and target
#     X = df.drop(columns=[target])
#     y = df[target]
    
#     X_ = X.copy()
#     #Non optional preproc steps
#     y = preproc.encode_labels(y) #encode labels
#     X_train, X_test, y_train, y_test = preproc.classification_split(X, y, test_size=0.3)
    
#     # Apply pipelines
#     for pipeline in preproc.get_defined_pipelines():   
#         print(f"im starting pipeline: {pipeline}")
#         #applying preproc steps
#         for preproc_step in pipeline: 
#             X_train, y_train, test_params = preproc.preproc_steps(X_train, y_train, phase="train")[preproc_step](X_train, y_train, phase="train")#apply preproc step
#             #for the validation 
#             # list_params.append(test_params)
#             X_test = preproc.preproc_steps(X_test, None, phase="test", test_params=test_params)[preproc_step]( X_test, phase="test", y=None, test_params=test_params)

#         X_train_scaled, X_test_scaled = preproc.scale_features(X_train, X_test, method="standard")


    
#         # get ML model
#         name_model = args.model
#         model = cls_models.get_cls_models(random_state=42)[name_model] #change model here
#         model.fit(X_train, y_train)

#         # --- Make predictions ---
#         y_pred = model.predict(X_test)

#         # --- Display results ---
#         # print("accuracy ", metrics.accuracy(y_test, y_pred))
#         print("cls report", metrics.cls_report(y_test, y_pred) )
#         print("f1_score ", metrics.f1_score(y_test, y_pred))
#         print(f"Applied Pipeline : {pipeline}")
#         print("fin.")
#         print(" ")
#         X_train, X_test, y_train, y_test = preproc.classification_split(X, y, test_size=0.3)
    



def run_classification_model(
    df: pd.DataFrame,
    target_column: str,
    model_name=None,
    kb=None,
    context_stack=None,
    num_random_pipelines: int = 3,
    random_state: int = 42,
    kb_mode: str = "kb_high",  # "no_kb" | "kb_low" | "kb_high"
    random_pipelines: bool = False,
):
    """
    Same pattern as run_regression_model:
      - optional KB retrieval of best preprocessing pipeline
      - fallback: checker-based suggestions OR random pipelines
      - loop pipelines & models, pick best by Recall(class=1), tie-break accuracy
      - optionally store to KB if better than existing
    """

    warnings.filterwarnings("ignore")

    cls_models = ClassificationModels()
    metrics = ClassificationMetrics()
    preproc = Pipeline()

    # --- 1) Split features/target ---
    if target_column not in df.columns:
        raise ValueError(f"target_column '{target_column}' not found in df.columns")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Mandatory preproc: encode labels (keep as you do)
    y = preproc.encode_labels(y)

    # --- 2) Init containers for pipelines + KB meta ---
    pipelines_to_test: dict = {}
    kb_best = None
    dist = None
    sim = None

    # --------- CAS AVEC KB ----------
    if kb is not None and kb_mode in ("kb_low", "kb_high"):
        kb_best, meta = kb.get_best_classification_pipeline_with_mode(
            use_case=getattr(context_stack, "use_case", "Generic Classification"),
            model_type=None if isinstance(model_name, (list, tuple)) else model_name,
            mode=kb_mode,
            query=None,
        )

        if meta is not None:
            dist, sim = meta

        print(f"[KB] mode={kb_mode} best={kb_best} dist={dist} sim={sim}")

        if kb_best is not None:
            # expected normalized keys: best_pipeline (list/str) or best_pipeline name
            pip_name = kb_best.get("best_pipeline") or kb_best.get("pipeline")
            if pip_name is not None:
                # If KB stores a predefined pipeline NAME -> use get_predefined_pips
                # If KB stores a LIST of steps -> use directly
                if isinstance(pip_name, str):
                    try:
                        pipelines_to_test[pip_name] = preproc.get_predefined_pips(pip_name)
                        print(f"[KB] Using pipeline from KB (predefined): {pip_name}")
                    except Exception:
                        # fallback: maybe it's a stringified list
                        pipelines_to_test[pip_name] = pip_name
                        print(f"[KB] Using pipeline from KB (raw str): {pip_name}")
                else:
                    # list of steps
                    pipelines_to_test[str(pip_name)] = pip_name
                    print(f"[KB] Using pipeline from KB (steps): {pip_name}")
            else:
                print("[KB] 'best_pipeline' key missing in KB record → fallback checkers.")
        else:
            print("[KB] No suitable pipeline found (or similarity too low) → fallback checkers.")

    # --------- FALLBACK / CAS SANS KB ----------
    if not pipelines_to_test:
        if kb_mode != "no_kb":
            print(f"[KB] Fallback → mode={kb_mode} best={kb_best} dist={dist} sim={sim}")
        print("--- Analyzing dataset to suggest pipelines... ---")

        if random_pipelines:
            print(f"[RANDOM] Selecting {num_random_pipelines} random pipelines.")
            ranked_pips = preproc.get_random_pipelines(num_random_pipelines)
            pipelines_to_test = {name: preproc.get_predefined_pips(name) for name in ranked_pips}
        else:
            # checker-based list of pipelines (you already have it)
            # NOTE: your current classif pipeline exposes get_defined_pipelines(X,y)
            # We'll treat those as candidates directly.
            ranked_pips = preproc.get_defined_pipelines(df, target_column)

            print("[CHECKERS] Ranked pipelines:", ranked_pips)

            # normalize into dict[name -> pipeline]
            # if ranked_pips is list of pipelines (list-of-steps),
            # store them keyed by str(pipeline)
            pipelines_to_test = {}
            for p in ranked_pips:
                pipelines_to_test[str(p)] = p

    # --- 3) Decide models ---
    all_available_models = cls_models.get_cls_models(random_state=random_state)
    allowed_models = list(all_available_models.keys())

    if model_name is None:
        model_names = allowed_models
    elif isinstance(model_name, (list, tuple)):
        model_names = [m for m in model_name if m in all_available_models]
    else:
        model_names = [model_name] if model_name in all_available_models else []

    if not model_names:
        raise ValueError(
            "No valid model names passed to run_classification_model(). "
            f"Allowed: {allowed_models}"
        )

    print("[MODELS] Will run:", model_names)

    # --- helper: recall of class 1 ---
    def extract_recall_class_1(report_dict) -> float:
        """
        Robust extraction of recall for class 1 from sklearn classification_report(output_dict=True)
        """
        if not isinstance(report_dict, dict):
            return 0.0
        for key in ("1", 1, "1.0"):
            if key in report_dict and isinstance(report_dict[key], dict):
                val = report_dict[key].get("recall", 0.0)
                try:
                    return float(val)
                except Exception:
                    return 0.0
        return 0.0

    # --- 4) Loop over pipelines & models ---
    all_results = []
    best_combo = {
        "recall_class_1": -1.0,
        "accuracy": -1.0,
        "model": None,
        "pipeline": None,
        "cls_report": None,
    }

    # base split (re-used inside loop like your current implementation)
    X_train_base, X_test_base, y_train_base, y_test_base = preproc.classification_split(
        X, y, test_size=0.3, random_state=random_state
    )

    for pipeline_key, pip in pipelines_to_test.items():
        print(f"\n--- [RUNNING] Pipeline: {pipeline_key} ---")

        # Rebuild split for parity between pipelines
        X_train, X_test, y_train, y_test = preproc.classification_split(
            X, y, test_size=0.3, random_state=random_state
        )

        # Apply pipeline steps
        X_tr, y_tr = X_train.copy(), y_train.copy()
        X_te = X_test.copy()

        # If pip is a predefined pipeline dict (like regression preproc.get_predefined_pips),
        # adapt accordingly; otherwise pip is list-of-steps.
        pipeline_steps = None
        if isinstance(pip, dict) and "pip_func" in pip:
            # If your ClassificationPipeline also has dict-style pipelines with pip_func,
            # you can implement it similarly; otherwise ignore this branch.
            try:
                out = pip["pip_func"](X_tr, y_tr)
                if out is None or not isinstance(out, tuple) or len(out) != 2:
                    print(f"[SKIP] {pipeline_key} → pip_func did not return (X, y).")
                    continue
                X_tr, y_tr = out
                # For test, you would need a consistent transform pipeline; if not available, skip.
                # This branch is optional and depends on your design.
            except Exception as e:
                print(f"[SKIP] {pipeline_key} → pipeline dict raised exception: {repr(e)}")
                continue
        else:
            # standard list-of-steps like your existing run_classification_model
            pipeline_steps = pip if isinstance(pip, (list, tuple)) else None
            if pipeline_steps is None:
                print(f"[SKIP] {pipeline_key} → pipeline is not list/tuple of steps.")
                continue

            for preproc_step in pipeline_steps:
                try:
                    X_tr, y_tr, test_params = preproc.preproc_steps(X_tr, y_tr, phase="train")[preproc_step](
                        X_tr, y_tr, phase="train"
                    )
                    X_te = preproc.preproc_steps(X_te, None, phase="test", test_params=test_params)[preproc_step](
                        X_te, phase="test", y=None, test_params=test_params
                    )
                except Exception as e:
                    print(f"[SKIP] {pipeline_key} → step '{preproc_step}' failed: {repr(e)}")
                    X_tr = None
                    break

            if X_tr is None:
                continue

        # Scale (as you already do)
        try:
            X_tr_scaled, X_te_scaled = preproc.scale_features(X_tr, X_te, method="standard")
        except Exception as e:
            print(f"[SKIP] {pipeline_key} → scaling failed: {repr(e)}")
            continue
        for m_name in model_names:
            print(f"\n   → [MODEL] {m_name}")
            model = cls_models.get_cls_models(random_state=random_state)[m_name]
        
            try:
                model.fit(X_tr_scaled, y_tr)
                y_pred = model.predict(X_te_scaled)
            except Exception as e:
                print(f"[SKIP] {pipeline_key} + {m_name} → fit/predict failed: {repr(e)}")
                continue
            
            # Metrics robustes (pas de parsing texte)
            rep = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            recall_1 = float(rep.get("1", {}).get("recall", 0.0))
            acc_val  = float(rep.get("accuracy", 0.0))
        
            # Optionnel: version texte
            rep_txt = classification_report(y_test, y_pred, zero_division=0)
        
            print(f"   --- [DONE] Recall(class=1): {recall_1:.4f}, Accuracy: {acc_val:.4f} ---")
        
            rec = {
                "model": m_name,
                "pipeline": pipeline_key,
                "recall_class_1": recall_1,
                "accuracy": acc_val,
                "cls_report": rep_txt,
            }
            all_results.append(rec)
        
            if (recall_1 > best_combo["recall_class_1"]) or (
                recall_1 == best_combo["recall_class_1"] and acc_val > best_combo["accuracy"]
            ):
                best_combo.update(
                    recall_class_1=recall_1,
                    accuracy=acc_val,
                    model=m_name,
                    pipeline=pipeline_key,
                    cls_report=rep_txt,
                )
        


    # --- 5) Compare ---
    if all_results:
        df_res = pd.DataFrame(all_results).sort_values(
            by=["recall_class_1", "accuracy"], ascending=[False, False]
        )
        print("\n\n--- 📊 Final Comparison (Classification) ---")
        try:
            print(df_res.to_markdown(index=False))
        except Exception:
            print(df_res)
    else:
        print("No results were generated.")
        return

    # --- 6) Optionnel : mise à jour de la KB ---
    if kb is not None and best_combo["model"] is not None:
        existing_best, existing_meta = kb.get_best_classification_pipeline_with_mode(
            use_case=getattr(context_stack, "use_case", "Generic Classification"),
            model_type=best_combo["model"],
            mode=kb_mode,
            query=None,
        )

        existing_recall = None
        existing_acc = None

        if existing_best is not None:
            m = (existing_best.get("metrics") or existing_best.get("evaluation") or {})
            existing_recall = m.get("recall_class_1")
            existing_acc = m.get("accuracy")

            try:
                existing_recall = float(existing_recall) if existing_recall is not None else None
            except Exception:
                existing_recall = None
            try:
                existing_acc = float(existing_acc) if existing_acc is not None else None
            except Exception:
                existing_acc = None

        better = False
        if existing_recall is None:
            better = True
        else:
            if best_combo["recall_class_1"] > existing_recall:
                better = True
            elif best_combo["recall_class_1"] == existing_recall:
                if existing_acc is None or best_combo["accuracy"] > existing_acc:
                    better = True

        if better:
            pipeline_data = {
                "best_model": best_combo["model"],
                "best_pipeline": best_combo["pipeline"],
                "use_case": getattr(context_stack, "use_case", "Generic Classification"),
                "target": getattr(context_stack, "data_profiling", {}).get("target", target_column),
                "evaluation": {
                    "recall_class_1": best_combo["recall_class_1"],
                    "accuracy": best_combo["accuracy"],
                },
                "model_type": best_combo["model"],
                "cls_report": best_combo.get("cls_report"),
            }

            print(f"[INFO] Storing best combo in KB: {pipeline_data}")
            kb.store_best_classification_pipeline_in_kb(pipeline_data, context_stack)
            kb.store_best_classification_pipeline_to_file(pipeline_data, context_stack)
        else:
            print(
                f"[INFO] New Recall1={best_combo['recall_class_1']:.4f}, Acc={best_combo['accuracy']:.4f} "
                f"is not better than KB Recall1={existing_recall}, Acc={existing_acc} → skip storing."
            )
