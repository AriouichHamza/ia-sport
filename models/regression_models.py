# Import necessary regression models from scikit-learn and XGBoost
import pip
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
import torch
import torch.nn as nn
import pandas as pd



import re

from typing import List, Tuple, Optional



from sklearn.model_selection import train_test_split
from utils.metrics.regression_metrics import RegressionMetrics
import numpy as np
from utils.pipelines.RegressionPipeline import RegressionPipeline as Pipeline
from utils.data_tools import build_mutlivariate_training_data,convert_to_pytorch_tensor, create_custom_datasets,get_dataloaders,add_extra_dim,convert_to_numpy_arry,convert_from_3D_to_2D
from utils.training_inference_tools import train_model,test_model,train_model_with_cross_val
"""
This module defines various regression models for machine learning tasks.
It includes:
1. Standard regression models from scikit-learn (XGBoost, MLP, RandomForest).
2. Custom PyTorch neural network models (N-Linear, GRUNet) for time series forecasting.
"""
from utils.Arguments import arguments as args


from utils.data_tools import (
    build_mutlivariate_training_data,
    create_custom_datasets,
    get_dataloaders,
    add_extra_dim,
    convert_to_numpy_arry,
    convert_from_3D_to_2D
)
from utils.training_inference_tools import train_model, test_model, train_model_with_cross_val
from utils.Arguments import arguments as args




class N_Linear(nn.Module):
    """
    A simple linear model for time series forecasting, known as N-Linear.

    This model implements a simple linear layer for forecasting. It first normalizes
    the input sequence by subtracting the last value, applies a linear transformation,
    and then adds the last value back. This helps mitigate the distribution shift
    problem common in time series data.
    """
    def __init__(self, seq_len: int, forecast_horizon: int, num_series: int, individual: bool = False, device: str = 'cpu'):
        """
        Initializes the N_Linear model.

        Args:
            seq_len (int): The length of the input sequence (window size).
            forecast_horizon (int): The number of time steps to predict into the future.
            num_series (int): The number of individual time series (channels).
            individual (bool): If True, create a separate linear layer for each time series.
                               If False, a single linear layer is shared across all series.
            device (str): The device to run the model on ('cpu' or 'cuda').
        """
        super(N_Linear, self).__init__()
        self.device = device
        self.seq_len = seq_len
        self.pred_len = forecast_horizon

        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        self.channels = num_series
        self.individual = individual

        # Create linear layers based on the 'individual' flag
        if self.individual:
            self.Linear = nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(nn.Linear(self.seq_len, self.pred_len))
        else:
            # A single linear layer for all time series
            self.Linear = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x: torch.Tensor, x_dec: torch.Tensor = None) -> torch.Tensor:
        """
        Defines the forward pass of the N-Linear model.

        Args:
            x (torch.Tensor): The input tensor of shape [Batch, Input length, Channel].
            x_dec (torch.Tensor, optional): Decoder input, not used in this model but
                                            included for API consistency with other models.

        Returns:
            torch.Tensor: The output prediction tensor of shape [Batch, Output length, Channel].
        """
        # Detach the last value of the sequence for normalization
        seq_last = x[:, -1:, :].detach()  # Shape: [Batch, 1, num_series]
        # Normalize the input by subtracting the last value
        x = x - seq_last

        # Apply the linear layer(s)
        if self.individual:
            # Apply a separate linear layer for each channel
            output = torch.zeros([x.size(0), self.pred_len, x.size(2)], dtype=x.dtype).to(self.device)
            for i in range(self.channels):
                output[:, :, i] = self.Linear[i](x[:, :, i])
            x = output
        else:
            # Apply the shared linear layer across all channels
            # Permute to shape [Batch, Channel, Input length] for the linear layer
            x = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)

        # Add the last value back to de-normalize the output
        x = x + seq_last
        return x  # [Batch, Output length, Channel]


# ---------- 1.  Direct mutlivariate multistep forecasting ----------
class GRU_Direct(nn.Module):
    def __init__(self, num_series, hidden_size, pred_len, output_size, num_layers=1):
        super(GRU_Direct, self).__init__()
        self.gru = nn.GRU(input_size=num_series,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len * output_size)
        self.pred_len = pred_len
        self.output_size = output_size

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, hn = self.gru(x)            # out: (batch, seq_len, hidden_size)
        last_hidden = out[:, -1, :]      # take hidden state from last timestep
        out = self.fc(last_hidden)       # (batch, pred_len * output_size)
        out = out.view(-1, self.pred_len, self.output_size)  # reshape to (batch, pred_len, output_size)
        return out

# ---------- 2. Autoregressive mutlivariate multistep forecasting ----------
class GRU_Autoregressive(nn.Module):
    def __init__(self, num_series, hidden_size, output_size,pred_len, num_layers=1):
        super(GRU_Autoregressive, self).__init__()
        self.gru = nn.GRU(input_size=num_series,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          batch_first=True)
        
        self.fc = nn.Linear(hidden_size, output_size)
        self.pred_len = pred_len


    def forward(self, x):
        """
        x: (batch, seq_len, num_series) -> past history
        pred_len: number of steps to predict
        """
        outputs = []
        # Initial hidden state from full input
        out, hn = self.gru(x)
        last_hidden = out[:, -1:, :]  # keep shape (batch, 1, hidden_size)
        # Autoregressive loop
        prev_input = x[:, -1:, :]  # last timestep of input as start
        h = hn
        for _ in range(self.pred_len):
            out, h = self.gru(prev_input, h)      # predict next hidden
            step_pred = self.fc(out[:, -1, :])    # map to output_size
            outputs.append(step_pred.unsqueeze(1))  # keep timestep dim
            # Feed prediction back as next input
            prev_input = step_pred.unsqueeze(1)

        return torch.cat(outputs, dim=1)  # (batch, pred_len, output_size)

class RegressionModels():
    """A factory class to provide instances of various regression models."""
    def __init__(self):
        pass

    def getXGBRegressor(self, objective, n_estimators, learning_rate, max_depth, subsample, colsample_bytree) -> XGBRegressor:
        """Returns an XGBoost Regressor model with specified hyperparameters."""
        return XGBRegressor(objective=objective, n_estimators=n_estimators, learning_rate=learning_rate,max_depth=max_depth,subsample=subsample, colsample_bytree=colsample_bytree )

    def getMLPRegressor(self, hidden_layer_sizes, activation, solver, max_iter, learning_rate_init, momentum, random_state) -> MLPRegressor:
        """Returns a Multi-layer Perceptron Regressor model with specified hyperparameters."""
        return MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,activation=activation,solver=solver,max_iter=max_iter,learning_rate_init=learning_rate_init,momentum=momentum,random_state=random_state)
    
    def getRandomForestRegressor(self, criterion, max_depth, n_estimators, min_samples_split, min_samples_leaf, max_features, bootstrap, random_state) -> RandomForestRegressor:
        """Returns a Random Forest Regressor model with specified hyperparameters."""
        return RandomForestRegressor(criterion=criterion ,  max_depth=max_depth, n_estimators=n_estimators, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, max_features=max_features, bootstrap=bootstrap, random_state=random_state)

    def getN_Linear(self, seq_len: int, forecast_horizon: int, num_series: int, individual: bool = False, device: str = 'cpu') -> N_Linear:
        """Returns an N-Linear model with specified hyperparameters."""
        # Corrected class name from nLinear to N_Linear
        nLinear = N_Linear(seq_len=seq_len, forecast_horizon=forecast_horizon, num_series=num_series, individual=individual, device=device)
        return nLinear
    def getGRU_Autoregressive(self, num_series:int,hidden_size=int,num_layers=int,pred_len=int,output_size=int):
            gru_auto =  GRU_Autoregressive(num_series=num_series,hidden_size=hidden_size,num_layers=num_layers,pred_len=pred_len,output_size=output_size)
            return gru_auto


    def get_rgr_models(self,args):
        """
        Collects and returns a dictionary of initialized scikit-learn regression models.
        
        Note: PyTorch models are commented out as they require specific parameters
        (e.g., sequence length, number of features) for instantiation.
        """       
        reg_models = {
        'XGBRegressor': self.getXGBRegressor(objective=args.m_params['XGBRegressor']['objective'], n_estimators=args.m_params['XGBRegressor']['n_estimators'], learning_rate=args.m_params['XGBRegressor']['learning_rate'],max_depth=args.m_params['XGBRegressor']['max_depth'],subsample=args.m_params['XGBRegressor']['subsample'], colsample_bytree=args.m_params['XGBRegressor']['colsample_bytree']),
        'MLPRegressor': self.getMLPRegressor(hidden_layer_sizes=args.m_params['MLPRegressor']['hidden_layer_sizes'], activation=args.m_params['MLPRegressor']['activation'], solver=args.m_params['MLPRegressor']['solver'], max_iter=args.m_params['MLPRegressor']['max_iter'], learning_rate_init=args.m_params['MLPRegressor']['learning_rate_init'], momentum=args.m_params['MLPRegressor']['momentum'], random_state=args.m_params['MLPRegressor']['random_state']),
        'RandomForestRegressor': self.getRandomForestRegressor(criterion=args.m_params['RandomForestRegressor']['criterion'], max_depth=args.m_params['RandomForestRegressor']['max_depth'], n_estimators=args.m_params['RandomForestRegressor']['n_estimators'], min_samples_split=args.m_params['RandomForestRegressor']['min_samples_split'], min_samples_leaf=args.m_params['RandomForestRegressor']['min_samples_leaf'], max_features=args.m_params['RandomForestRegressor']['max_features'], bootstrap=args.m_params['RandomForestRegressor']['bootstrap'], random_state=args.m_params['RandomForestRegressor']['random_state']),
        'GRU_Autoregressive' : self.getGRU_Autoregressive(num_series=args.m_params['shared_m_params']['num_series'],hidden_size=args.m_params['GRU_Autoregressive']['hidden_size'],num_layers=args.m_params['GRU_Autoregressive']['num_layers'],pred_len=args.m_params['shared_m_params']['pred_len'],output_size=args.m_params['GRU_Autoregressive']['output_size']),
        'N_Linear' : self.getN_Linear(seq_len=args.m_params['shared_m_params']['seq_len'], forecast_horizon=args.m_params['shared_m_params']['pred_len'], num_series=args.m_params['shared_m_params']['num_series'], individual=args.m_params['N_Linear']['individual'], device=args.device)
        }
        return reg_models


import pandas as pd
import numpy as np
import re
from typing import Tuple, Dict, Optional

def encode_target_if_categorical(
    df: pd.DataFrame,
    target_column: str,
    max_unique: int = 20,
    start_at: int = 0,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Optional[Dict[str, int]]]:
    """
    If the target column is object/category, encode it as ordinal integers.

    - Only encodes if dtype is object/category AND nb of unique values <= max_unique.
    - Returns (df, mapping) where `mapping` is {original_label -> int}
      or None if no encoding was applied.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the target.
    target_column : str
        Name of the target column.
    max_unique : int
        Max nb of categories to encode automatically.
    start_at : int
        First integer used in the mapping (0 or 1, etc.).
    verbose : bool
        If True, print info and mapping.

    """
    if target_column not in df.columns:
        raise KeyError(
            f"Target column '{target_column}' not found in dataframe. "
            f"Available columns: {list(df.columns)}"
        )

    col = df[target_column]
    dtype = col.dtype

    # Only act on object/category
    if not (pd.api.types.is_object_dtype(dtype) or pd.api.types.is_categorical_dtype(dtype)):
        if verbose:
            print(f"[ENCODER] Target '{target_column}' is not object/category (dtype={dtype}), no encoding applied.")
        return df, None

    # Drop NaNs for unique analysis
    uniques = col.dropna().unique()

    if len(uniques) == 0:
        if verbose:
            print(f"[ENCODER] Target '{target_column}' has only NaN values, no encoding applied.")
        return df, None

    if len(uniques) > max_unique:
        if verbose:
            print(
                f"[ENCODER] Target '{target_column}' has {len(uniques)} unique values "
                f"(> {max_unique}), skipping automatic ordinal encoding."
            )
        return df, None

    # Sort uniques for deterministic mapping
    # Cast to str for sorted order if mixed types
    uniques_sorted = sorted(uniques.astype(str))
    mapping = {cat: i + start_at for i, cat in enumerate(uniques_sorted)}

    if verbose:
        print(f"[ENCODER] Ordinal encoding for target '{target_column}':")
        for k, v in mapping.items():
            print(f"   {k!r} -> {v}")

    # Apply mapping
    df[target_column] = df[target_column].astype(str).map(mapping)

    # Check NaNs after mapping
    n_nans = df[target_column].isna().sum()
    if verbose:
        print(f"[ENCODER] NaNs in '{target_column}' after encoding: {n_nans}")

    return df, mapping

def build_generic_profile(
    df: pd.DataFrame,
    target_column: str,
    context_stack=None,
) -> Tuple[str, List[str], List[str]]:

    # NEW: try ordinal encoding if target is categorical
    df, target_mapping = encode_target_if_categorical(
        df,
        target_column=target_column,
        max_unique=20,
        start_at=0,
        verbose=True,
    )

    # Keep your existing numeric coercion for safety
    df[target_column] = pd.to_numeric(df[target_column], errors="coerce")

    # Numeric / bool features
    num_cols = df.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    features = [c for c in num_cols if c != target_column]

    # Categorical features
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    id_pattern = re.compile(r"id$|_id$|uuid|^id$", re.IGNORECASE)
    categorical_cols = [c for c in cat_cols if not id_pattern.search(c)]

    # Sync into context_stack
    if context_stack is not None and hasattr(context_stack, "data_profiling"):
        context_stack.data_profiling["features"] = features
        context_stack.data_profiling["target"] = target_column
        if target_mapping is not None:
            context_stack.data_profiling["target_encoding"] = target_mapping
        if hasattr(context_stack, "use_case") and context_stack.use_case is None:
            context_stack.use_case = "Generic Regression"

    return target_column, features, categorical_cols







def run_regression_model(
    df: pd.DataFrame,
    target_column: str,
    model_name=None,
    kb=None,
    context_stack=None,
    num_random_pipelines: int = 3,
    random_state: int = 42,
    kb_mode: str = "kb_high",  # "no_kb" | "kb_low" | "kb_high"
    random_pipelines : bool = False,
):
    rgr_models = RegressionModels()
    metrics = RegressionMetrics()
    preproc = Pipeline()

    # --- 1) Dataset-agnostic profile from df ---
    target_column, features, categorical_cols = build_generic_profile(
        df, target_column, context_stack=context_stack
    )

    # safety: drop target if it sneaks into features
    features = [f for f in features if f != target_column]

    print("[INFO] Target:", target_column)
    print("[INFO] Features:", features)
    print("[INFO] Categorical cols:", categorical_cols)

    # --- 2) Prepare df for pipelines (time col detection is generic) ---
    has_time, time_col = preproc._detect_time_column(df)
    df_for_pips = preproc._maybe_rename_time_col_to_Date(
        df.copy(),
        time_col if has_time else None
    )

    # --- 2bis) Init containers for pipelines + KB meta ---
    pipelines_to_test: dict = {}
    kb_best = None
    dist = None
    sim = None

    # --------- CAS AVEC KB ----------
    if kb is not None and kb_mode in ("kb_low", "kb_high"):
        kb_best, meta = kb.get_best_regression_pipeline_with_mode(
            use_case=getattr(context_stack, "use_case", "Generic Regression"),
            model_type=None if isinstance(model_name, (list, tuple)) else model_name,
            mode=kb_mode,   # "kb_low" ou "kb_high"
            query=None,
        )

        if meta is not None:
            dist, sim = meta

        print(f"[KB] mode={kb_mode} best={kb_best} dist={dist} sim={sim}")

        if kb_best is not None:
            pip_name = kb_best.get("best_pipeline")
            if pip_name is not None:
                pipelines_to_test[pip_name] = preproc.get_predefined_pips(pip_name)
                print(f"[KB] Using pipeline from KB: {pip_name}")
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
            pipelines_to_test = {
                name: preproc.get_predefined_pips(name) for name in ranked_pips
            }
        else:
            ranked_pips, signals = preproc.suggest_pipelines(
                df_for_pips,
                features,
                target_column,
            )
            print("[CHECKERS] Signals:", signals)




        print("[CHECKERS] Ranked pipelines:", ranked_pips)

        pipelines_to_test = {
            name: preproc.get_predefined_pips(name) for name in ranked_pips
        }

    # --- 3) Decide models (only 3 ML ones) ---
    allowed_models = ["XGBRegressor", "MLPRegressor", "RandomForestRegressor"]
    all_available_models = rgr_models.get_rgr_models(args)

    if model_name is None:
        model_names = [m for m in allowed_models if m in all_available_models]
    elif isinstance(model_name, (list, tuple)):
        model_names = [m for m in model_name if m in allowed_models and m in all_available_models]
    else:
        model_names = [model_name] if model_name in allowed_models and model_name in all_available_models else []

    if not model_names:
        raise ValueError(
            "No valid model names passed to run_regression_model(). "
            f"Allowed: {allowed_models}"
        )

    print("[MODELS] Will run:", model_names)

    # --- 4) Loop over pipelines & models ---
    all_results = []
    best_combo = {
        "rmse": float("inf"),
        "model": None,
        "pipeline": None,
        "mape": None,
        "nrmse_mean": None,
    }

    for pipeline_name, pip in pipelines_to_test.items():
        print(f"\n--- [RUNNING] Pipeline: {pipeline_name} ---")

        # SAFER CALL
        try:
            out = pip["pip_func"](
                df_for_pips.copy(),
                features,
                [target_column],
                categorical_cols,
            )
        except Exception as e:
            print(f"[SKIP] {pipeline_name} → pipeline raised an exception: {repr(e)}")
            continue

        if out is None:
            print(f"[SKIP] {pipeline_name} → pipeline returned None instead of (X, y).")
            continue

        if not isinstance(out, tuple) or len(out) != 2:
            print(
                f"[SKIP] {pipeline_name} → expected (X, y), got {type(out)} "
                f"with value: {out}"
            )
            continue

        X, y = out

        if isinstance(y, pd.DataFrame):
            if y.shape[1] != 1:
                print(f"[SKIP] {pipeline_name} → y has {y.shape[1]} columns; need 1.")
                continue
            y = y.iloc[:, 0]
        else:
            y = pd.Series(np.asarray(y).reshape(-1))

        if len(y) < 2 or X.shape[0] < 2:
            print(f"[SKIP] {pipeline_name} → too few samples.")
            continue

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state
        )
        datasets = {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
        }

        datasets_scaled = convert_to_numpy_arry(datasets)
        datasets_scaled = add_extra_dim(datasets_scaled, 2)

        loop_preproc = Pipeline()
        loop_preproc.initialize_scaler()
        datasets_scaled["X_train"], datasets_scaled["y_train"] = loop_preproc.scale_features(
            datasets_scaled["X_train"], datasets_scaled["y_train"]
        )
        datasets_scaled["X_test"], datasets_scaled["y_test"] = loop_preproc.scale_features(
            datasets_scaled["X_test"], datasets_scaled["y_test"]
        )

        y_test_denorm = loop_preproc.inverse_scale_features(datasets_scaled["y_test"])

        for m_name in model_names:
            print(f"\n   → [MODEL] {m_name}")
            model = rgr_models.get_rgr_models(args)[m_name]

            y_train_flat = datasets_scaled["y_train"].ravel()
            model.fit(datasets_scaled["X_train"], y_train_flat)

            y_pred_scaled = model.predict(datasets_scaled["X_test"])
            y_pred_scaled = add_extra_dim({"y_pred": y_pred_scaled}, 2)["y_pred"]
            y_pred = loop_preproc.inverse_scale_features(y_pred_scaled)

            rmse = metrics.rmse(y_test_denorm, y_pred)
            mape = metrics.mape(y_test_denorm, y_pred)
            nrmse_mean = metrics.nRMSE_mean(y_test_denorm, y_pred)

            print(
                f"   --- [DONE] RMSE: {rmse:.4f}, "
                f"MAPE: {mape:.4f}, nRMSE_mean: {nrmse_mean:.4f} ---"
            )

            result = {
                "model": m_name,
                "pipeline": pipeline_name,
                "RMSE": rmse,
                "MAPE": mape,
                "nRMSE_mean": nrmse_mean,
            }
            all_results.append(result)

            if rmse < best_combo["rmse"]:
                best_combo.update(
                    rmse=rmse,
                    model=m_name,
                    pipeline=pipeline_name,
                    mape=mape,
                    nrmse_mean=nrmse_mean,
                )

    # --- 5) Compare ---
    if all_results:
        df_res = pd.DataFrame(all_results).sort_values(by="RMSE")
        print("\n\n--- 📊 Final Comparison ---")
        print(df_res.to_markdown(index=False))
    else:
        print("No results were generated.")
        return  # rien à stocker, on sort

    # --- 6) Optionnel : mise à jour de la KB ---
    if kb is not None and best_combo["model"] is not None:
        existing_best, existing_meta = kb.get_best_regression_pipeline_with_mode(
            use_case=getattr(context_stack, "use_case", "Generic Regression"),
            model_type=best_combo["model"],   # même type de modèle
            mode=kb_mode,                     # même mode que pour la récupération
            query=None,
        )

        existing_rmse = None
        if existing_best is not None:
            eval_old = existing_best.get("evaluation", {}) or {}
            existing_rmse = eval_old.get("rmse")

        if existing_rmse is None or best_combo["rmse"] < existing_rmse:
            pipeline_data = {
                "best_model": best_combo["model"],
                "best_pipeline": best_combo["pipeline"],
                "use_case": getattr(context_stack, "use_case", "Generic Regression"),
                "target": getattr(context_stack, "data_profiling", {}).get("target", target_column),
                "evaluation": {
                    "rmse": best_combo["rmse"],
                    "mape": best_combo["mape"],
                    "nrmse_mean": best_combo["nrmse_mean"],
                },
                "model_type": best_combo["model"],
            }
            print(f"[INFO] Storing best combo in KB: {pipeline_data}")
            kb.store_best_regression_pipeline_in_kb(pipeline_data, context_stack)
            kb.store_best_regression_pipeline_to_file(pipeline_data, context_stack)
        else:
            print(
                f"[INFO] New RMSE={best_combo['rmse']:.4f} "
                f"is not better than KB RMSE={existing_rmse:.4f} → skip storing."
            )

                                



#
#


# def run_regression_model(
#     df: pd.DataFrame,
#     target_column: str,
#     model_name=None,          # str | list[str] | None
#     kb=None,
#     context_stack=None,
#     num_random_pipelines: int = 3,
#     random_state: int = 42
# ):
#     """Execute regression workflow on a given DataFrame and one or many models."""
    
#     rgr_models = RegressionModels()
#     metrics = RegressionMetrics()
#     preproc = Pipeline()

#     # --- Features & target from context_stack ---
#     # Force target to numeric so it survives numeric_only=True
#     df[target_column] = pd.to_numeric(df[target_column], errors="coerce")

#     features = context_stack.data_profiling["features"]
#     target = df[target_column]


#     print(f"--- Analyzing dataset to suggest pipelines... ---")

#     # Ensure time column name aligns with pipelines
#     has_time, time_col = preproc._detect_time_column(df)
#     df_for_pips = preproc._maybe_rename_time_col_to_Date(df.copy(), time_col if has_time else None)

#     ranked_pips, signals = preproc.suggest_pipelines(df_for_pips, features, target[0])
#     print("[CHECKERS] Signals:", signals)
#     print("[CHECKERS] Ranked pipelines:", ranked_pips)

#     pipelines_to_test = {name: preproc.get_predefined_pips(name) for name in ranked_pips}

#     # --- Decide which models to run ---
#     all_available_models = rgr_models.get_rgr_models(args)  # dict[name -> model instance]

#     if model_name is None:
#         # use all models
#         model_names = list(all_available_models.keys())
#     elif isinstance(model_name, (list, tuple)):
#         model_names = list(model_name)
#     else:
#         model_names = [model_name]

#     # keep only valid names
#     model_names = [m for m in model_names if m in all_available_models]
#     if not model_names:
#         raise ValueError("No valid model names passed to run_regression_model().")

#     print("[MODELS] Will run:", model_names)

#     all_results = []  # each entry: {"model", "pipeline", "RMSE", "MAPE", "nRMSE_mean"}

#     # Track global best combo (model + pipeline)
#     best_combo = {
#         "rmse": float("inf"),
#         "model": None,
#         "pipeline": None,
#         "mape": None,
#         "nrmse_mean": None,
#     }

#     # --- Outer loop over pipelines ---
#     for pipeline_name, pip in pipelines_to_test.items():
#         print(f"\n--- [RUNNING] Pipeline: {pipeline_name} ---")

#         # Apply pipeline once per pipeline (shared X,y for all models)
#         X, y = pip['pip_func'](df_for_pips.copy(), features, target, None)

#         # y → always 1D Series
#         if isinstance(y, pd.DataFrame):
#             if y.shape[1] != 1:
#                 print(f"[SKIP] {pipeline_name} → y has {y.shape[1]} columns; need 1.")
#                 continue
#             y = y.iloc[:, 0]
#         else:
#             y = pd.Series(np.asarray(y).reshape(-1))

#         if len(y) < 2 or X.shape[0] < 2:
#             print(f"[SKIP] {pipeline_name} → too few samples after preprocessing.")
#             continue

#         # Train/Test Split (same for all models)
#         X_train, X_test, y_train, y_test = train_test_split(
#             X, y, test_size=0.2, random_state=random_state
#         )
#         datasets = {
#             "X_train": X_train,
#             "y_train": y_train,
#             "X_test": X_test,
#             "y_test": y_test,
#         }

#         # Convert to numpy & add extra dim
#         datasets_scaled = convert_to_numpy_arry(datasets)
#         datasets_scaled = add_extra_dim(datasets_scaled, 2)

#         # Initialize scaler ONCE per pipeline
#         loop_preproc = Pipeline()
#         loop_preproc.initialize_scaler()

#         datasets_scaled["X_train"], datasets_scaled["y_train"] = loop_preproc.scale_features(
#             datasets_scaled["X_train"], datasets_scaled["y_train"]
#         )
#         datasets_scaled["X_test"], datasets_scaled["y_test"] = loop_preproc.scale_features(
#             datasets_scaled["X_test"], datasets_scaled["y_test"]
#         )

#         # Denorm y_test once (shared across models)
#         y_test_denorm = loop_preproc.inverse_scale_features(datasets_scaled["y_test"])

#         # --- Inner loop over models ---
#         for m_name in model_names:
#             print(f"\n   → [MODEL] {m_name}")

#             # Fresh instance for each (pipeline, model) combo
#             model = rgr_models.get_rgr_models(args)[m_name]

#             results = {
#                 "model": m_name,
#                 "pipeline": pipeline_name,
#             }

#             # Only ML models here (XGB, RF, MLP) – DL needs special handling
#             y_train_flat = datasets_scaled["y_train"].ravel()
#             model.fit(datasets_scaled["X_train"], y_train_flat)

#             y_pred_scaled = model.predict(datasets_scaled["X_test"])
#             y_pred_scaled = add_extra_dim({'y_pred': y_pred_scaled}, 2)['y_pred']

#             # Denormalize predictions
#             y_pred = loop_preproc.inverse_scale_features(y_pred_scaled)

#             # Metrics
#             rmse = metrics.rmse(y_test_denorm, y_pred)
#             mape = metrics.mape(y_test_denorm, y_pred)
#             nrmse_mean = metrics.nRMSE_mean(y_test_denorm, y_pred)

#             results["RMSE"] = rmse
#             results["MAPE"] = mape
#             results["nRMSE_mean"] = nrmse_mean

#             print(f"   --- [DONE] {m_name} | RMSE: {rmse:.4f}, MAPE: {mape:.4f}, nRMSE_mean: {nrmse_mean:.4f} ---")
#             all_results.append(results)

#             # Update global best combo
#             if rmse < best_combo["rmse"]:
#                 best_combo["rmse"] = rmse
#                 best_combo["model"] = m_name
#                 best_combo["pipeline"] = pipeline_name
#                 best_combo["mape"] = mape
#                 best_combo["nrmse_mean"] = nrmse_mean
#                 print(f"   [INFO] New best combo: model={m_name}, pipeline={pipeline_name}, RMSE={rmse:.4f}")

#     # --- Final comparison ---
#     print("\n\n--- 📊 Final Model + Pipeline Comparison ---")
#     if all_results:
#         results_df = pd.DataFrame(all_results)
#         results_df = results_df.sort_values(by="RMSE")
#         print(results_df.to_markdown(index=False))

#         if best_combo["model"] is not None and kb is not None:
#             pipeline_data = {
#                 "best_model": best_combo["model"],
#                 "best_pipeline": best_combo["pipeline"],
#                 "use_case": context_stack.use_case,
#                 "target": context_stack.data_profiling['target'],
#                 "evaluation": {
#                     "rmse": best_combo["rmse"],
#                     "mape": best_combo["mape"],
#                     "nrmse_mean": best_combo["nrmse_mean"],
#                 },
#                 "model_type": best_combo["model"],
#             }

#             print(f"[INFO] Storing best combo in KB: model={best_combo['model']}, pipeline={best_combo['pipeline']}")
#             kb.store_best_regression_pipeline_in_kb(pipeline_data, context_stack)
#     else:
#         print("No results were generated.")




























# if __name__ == "__main__":
#     rgr_models = RegressionModels()
#     metrics = RegressionMetrics()
#     preproc = Pipeline()

#     df=pd.read_csv('./datasets/df_normalized.csv')
#        # Sample data


#     features = ['duration_min', 'fatigue_day-1', 'mood_day-1',
#        'readiness_day-1', 'sleep_duration_h_day-1', 'sleep_quality_day-1',
#        'soreness_day-1', 'stress_day-1', 'fatigue_day-2', 'mood_day-2',
#        'readiness_day-2', 'sleep_duration_h_day-2', 'sleep_quality_day-2',
#        'soreness_day-2', 'stress_day-2', 'fatigue_day-3', 'mood_day-3',
#        'readiness_day-3', 'sleep_duration_h_day-3', 'sleep_quality_day-3',
#        'soreness_day-3', 'stress_day-3', 'injury_count_last_7d',
#        'injury_severity_last_7d']
        
#     target = ['RPE']
#     categorical_cols = None
    
#     num_rnd_pips  = 8
#     ## launch pipeline seperately   
#     # pip = preproc.get_predefined_pips("apply_first_pipeline")
#     # X , y = pip['pip_func'](df.copy(), features, target, categorical_cols)

#     # launch pipelines iteratively
#     rnd_pips = preproc.generate_random_pipelines(num_rnd_pips)
#     for pip_name, pip in rnd_pips.items():
#         print("pip_name: ", pip_name)
#         X, y = pip['pip_func'](df.copy(), features, target, categorical_cols)
            
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#         # print("-- ", args.ml_model_selected)
#         # print("args ", )
#         #if args.ml_model_selected == args.rgr_models
#         datasets = {
#             "X_train": X_train,
#             "y_train": y_train, 
#             "X_test": X_test,
#             "y_test": y_test
#         } 
#         #convert to numpy array
#         datasets  = convert_to_numpy_arry(datasets)
#         #add extra dim
#         datasets = add_extra_dim(datasets,2)
#         #initialize le scaler   
#         preproc.initialize_scaler()
#          #normalization/scaling 2D
#         datasets['X_train'], datasets['y_train'] = preproc.scale_features(datasets['X_train'],datasets['y_train'])
#         datasets['X_test'], datasets['y_test'] = preproc.scale_features(datasets['X_test'],datasets['y_test'])

#         if args.ml_model_selected == list(args.rgr_models.keys())[1]:  #DL_models
#             print("DL model selected: ", args.selected_model)
#             # ****when dealing with DL models - Apply preprocessing steps
                             

#             datasets['X_train'], _ = build_mutlivariate_training_data(datasets['X_train'], args.time_steps, 1)
#             _, datasets['y_train'] = build_mutlivariate_training_data(datasets['y_train'],  args.time_steps,1)
#             datasets['X_test'], _ = build_mutlivariate_training_data(datasets['X_test'],  args.time_steps, 1)
#             _, datasets['y_test'] = build_mutlivariate_training_data(datasets['y_test'],  args.time_steps,1)
            
                    
#             print("---")
#             print(datasets['X_train'].shape)
#             print(datasets['y_train'].shape)
#             print(datasets['X_test'].shape)
#             print(datasets['y_test'].shape)
        
#             datasets = add_extra_dim(datasets,3)
#             #get num_series, seq_len and pred_len before training
#             args.m_params['shared_m_params']['num_series'], args.m_params['shared_m_params']['seq_len'], args.m_params['shared_m_params']['pred_len'] = datasets['X_train'].shape[-1], datasets['X_train'].shape[-2], datasets['y_train'].shape[-2]
#             # print(f"{args.num_series} - {args.seq_len} - {args.pred_len} ")
            
#             datasets = create_custom_datasets(datasets)
#             dataloaders = get_dataloaders(datasets, args.batch_size, args.shuffle)
#             model = rgr_models.get_rgr_models(args)['GRU_Autoregressive']
            
#             mse_loss, mae_loss = nn.MSELoss(), nn.L1Loss()
#             lst_losses = [mse_loss, mae_loss]
        
#             # train model
#             train_epochs_mae_losses, val_epochs_mae_losses, optimizer = train_model(args, model, lst_losses, dataloaders['train_dataloader'],dataloaders['test_dataloader'], metrics)

#             pred_array, y_array = test_model(args, model, lst_losses[1], dataloaders['test_dataloader'], optimizer, metrics)

#             pred_array,y_array = convert_from_3D_to_2D(pred_array), convert_from_3D_to_2D(y_array)
            
            
#             # #denormalizing
#             pred_array = preproc.inverse_scale_features(pred_array)
#             y_array = preproc.inverse_scale_features(y_array)
            
#             metrics.displayMetrics(y_array, pred_array)

#             print("train_epochs_mae_losses", train_epochs_mae_losses)
#             print("val_epochs_mae_losses", val_epochs_mae_losses)
            
#         else:
#             print("ML model selected: ", args.selected_model)
#             model = rgr_models.get_rgr_models(args)[args.selected_model]
        
#             X_train, y_train, X_test, y_test = datasets['X_train'], datasets['y_train'], datasets['X_test'],datasets['y_test'],  
#             if args.cross_validation:
#                 train_model_with_cross_val(model = model, X= X.values,y=y.values,n_splits=args.cross_val_n_splits,metrics=metrics)
#             else:
#                 model.fit(X_train, y_train)
#                 y_pred = model.predict(X_test)
#                 y_pred =  add_extra_dim({'y_pred':y_pred},2)['y_pred']
#                 y_mean = y_pred.mean()
#                      # #denormalizing
#                 y_pred = preproc.inverse_scale_features(y_pred)
#                 y_test = preproc.inverse_scale_features(y_test)

#                 print("RMSE:{} nRMSE_mean:{} nRMSE_range:{}  MAPE:{}".format(metrics.rmse(y_test, y_pred),metrics.nRMSE_mean(y_test, y_pred),metrics.nRMSE_range(y_test, y_pred),metrics.mape(y_test, y_pred)))

