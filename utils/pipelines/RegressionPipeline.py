import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, FunctionTransformer
import random

import re
from typing import Dict, List, Tuple


class RegressionPipeline:
    def __init__(self):
        self.encoder = None
        self.imputer = None

    # ------------------------------------------------------
    # 1. Handle missing timestamps, deals with duplicates
    # ------------------------------------------------------

    def initialize_scaler(self):
        self.normalizer = self.TimeSeriesNormalizer(method = "standard")


    def fill_missing_timestamps(self, df: pd.DataFrame, time_col: str, freq: str = 'D') -> pd.DataFrame:
        # Ensure datetime type
        df[time_col] = pd.to_datetime(df[time_col])

        # Group, then set the index and resample
        df = (
            df.groupby(time_col)
            .mean(numeric_only=True)
            .asfreq(freq)
            .reset_index()
        )

        return df
    # ------------------------------------------------------
    # 2. Missing value imputation
    # ------------------------------------------------------
    # def impute_missing_values(self, df: pd.DataFrame, strategy: str = "mean") -> pd.DataFrame:
    #     self.imputer = SimpleImputer(strategy=strategy)
    #     imputed = self.imputer.fit_transform(df)
    #     return pd.DataFrame(imputed, columns=df.columns, index=df.index)
    
    def impute_missing_values(self, df: pd.DataFrame, strategy: str = "mean") -> pd.DataFrame:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        non_numeric_cols = df.columns.difference(numeric_cols)

        imputer = SimpleImputer(strategy=strategy)
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

        return df

    # ------------------------------------------------------
    # 3. Outlier handling
    # ------------------------------------------------------
    def clip_outliers(self, df: pd.DataFrame, lower_quantile: float = 0.01, upper_quantile: float = 0.99) -> pd.DataFrame:
        lower = df.quantile(lower_quantile)
        upper = df.quantile(upper_quantile)
        return df.clip(lower=lower, upper=upper, axis=1)

    
    # ------------------------------------------------------
    # 4. Scaling/Normalization
    # ------------------------------------------------------
    def scale_features(self, X:np.ndarray, y:np.ndarray) -> pd.DataFrame:
        if not self.normalizer.fitted:
            # print("fitting the X data once")
            self.normalizer.fit(X, y)
        scaled_x,scaled_y = self.normalizer.transform(X, y)
        return  scaled_x, scaled_y


    # ------------------------------------------------------
    # 5. Reverse Scaling/Normalization
    # ------------------------------------------------------
    def inverse_scale_features(self, y_scaled:np.ndarray) -> pd.DataFrame:
        """
        Undo the scaling applied by scale_features().
        Assumes self.scaler was already fitted during scaling.

        Parameters
        ----------
        y : np.ndarray
            The scaled y dataset to invert back to the original values.
        Returns
        -------
        y: np.ndarray
            Inversed scaled y dataset.
        """
        inversed_y = self.normalizer.inverse_transform_y(y_scaled)
        return inversed_y
    
    # ------------------------------------------------------
    # 6. Categorical encoding
    # ------------------------------------------------------
    def encode_categoricals(self, df: pd.DataFrame, categorical_cols: list = None) -> pd.DataFrame:
        if categorical_cols is None or len(categorical_cols) == 0:
            return df

        # Keep only columns that actually exist in df
        categorical_cols = [c for c in categorical_cols if c in df.columns]
        if not categorical_cols:
            return df

        # Handle sklearn version differences: sparse_output vs sparse
        try:
            # Newer versions (>= 1.2+)
            self.encoder = OneHotEncoder(
                sparse_output=False,
                drop="first",
                handle_unknown="ignore",
            )
        except TypeError:
            # Older versions
            self.encoder = OneHotEncoder(
                sparse=False,
                drop="first",
                handle_unknown="ignore",
            )

        encoded = self.encoder.fit_transform(df[categorical_cols])
        encoded_df = pd.DataFrame(
            encoded,
            index=df.index,
            columns=self.encoder.get_feature_names_out(categorical_cols),
        )

        df = df.drop(columns=categorical_cols)
        return pd.concat([df, encoded_df], axis=1)


    # ------------------------------------------------------
    # 7. Feature engineering: lag features
    # ------------------------------------------------------
    def create_lag_features(self, df, col: str, lags: int = 3):
        df = df.copy()
        for i in range(1, lags + 1):
            df[f"{col}_lag{i}"] = df[col].shift(i)
        cols_to_check = [col] + [f"{col}_lag{i}" for i in range(1, lags + 1)]
        df = df.dropna(subset=cols_to_check)
        y = df[col].copy()
        X = df.drop(columns=[col])
        # ensure contiguous matching indices
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        return X, y

    def create_rolling_features(self, df, col: str, window: int = 3):
        df = df.copy()
        df[f"{col}_roll_mean_{window}"] = df[col].rolling(window).mean()
        df[f"{col}_roll_std_{window}"]  = df[col].rolling(window).std()
        cols_to_check = [col, f"{col}_roll_mean_{window}", f"{col}_roll_std_{window}"]
        df = df.dropna(subset=cols_to_check)
        y = df[col].copy()
        X = df.drop(columns=[col])
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        return X, y


    # ------------------------------------------------------
    # 9. Windowing for supervised learning
    # ------------------------------------------------------
    def create_supervised_windows(self, df: pd.DataFrame, target_col: str, window_size: int = 3):
        X, y = [], []
        values = df[target_col].values
        for i in range(len(values) - window_size):
            X.append(values[i:i+window_size])
            y.append(values[i+window_size])
        return np.array(X), np.array(y)

   # ------------------------------------------------------
    # 10. Target transformation (optional)
    # ------------------------------------------------------
    def transform_target(self, y: pd.Series, method: str = "log") -> np.ndarray:
        if method == "log":
            self.target_transformer = FunctionTransformer(func=np.log1p, inverse_func=np.expm1)
            return self.target_transformer.fit_transform(y.values.reshape(-1,1)).flatten()
        return y.values
    
    # ------------------------------------------------------
    # 11. Target transformation (optional)
    # ------------------------------------------------------
    def inverse_transform_target(self, y_transformed: np.ndarray) -> np.ndarray:
            if self.target_transformer:
                return self.target_transformer.inverse_transform(y_transformed.reshape(-1,1)).flatten()
            return y_transformed
    
    # ------------------------------------------------------
    # 12. Drop outliers
    # ------------------------------------------------------
    def drop_outliers(self, df: pd.DataFrame, lower_quantile: float = 0.01, upper_quantile: float = 0.99) -> pd.DataFrame:
        """ 
        Drops rows with values outside the specified quantile range for any numeric column in the DataFrame. 
        Parameters: df (pd.DataFrame): Input DataFrame. lower_quantile (float): Lower quantile threshold (default=0.01). upper_quantile (float): Upper quantile threshold (default=0.99).
        Returns: pd.DataFrame: DataFrame with outlier rows removed.
        """
        lower = df.quantile(lower_quantile) 
        upper = df.quantile(upper_quantile) 
        # Keep only rows where all numeric values are within the quantile range 
        mask = (df >= lower) & (df <= upper) 
        return df[mask.all(axis=1)]
   

    def get_features_target(self, df:pd.DataFrame, lst_features:list, lst_target:list):
        X = df[lst_features]
        y = df[lst_target]
        return X, y
    
    def _tgt_name(self, target):
        return target[0] if isinstance(target, (list, tuple)) else target

#""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""#
    def apply_first_pipeline(
        self,
        df: pd.DataFrame,
        features: list,
        target: list,
        categorical_cols: list = None
    ):
        """
        Pipeline 1: fill_missing_timestamps -> impute -> (encode categoricals) -> clip_outliers
        """
        tgt = self._tgt_name(target)

        # Time handling + NaNs
        df = self.fill_missing_timestamps(df, time_col="Date", freq="D")
        df = self.impute_missing_values(df, strategy="mean")

        # --- Optional categorical encoding ---
        if categorical_cols:
            # keep only cats that actually exist
            valid_cats = [c for c in categorical_cols or [] if c in df.columns]
            df = self.encode_categoricals(df, categorical_cols=valid_cats)
      

        # Ensure we only keep features that still exist after preprocessing/encoding
        features = self._filter_existing_features(df, features)

        X = df[features]
        y = df[tgt]

        # Outlier clipping on feature matrix
        X = self.clip_outliers(X, lower_quantile=0.01, upper_quantile=0.99)

        return self._coerce_xy(X, y)


    def apply_second_pipeline(
        self,
        df: pd.DataFrame,
        features: list,
        target: list,
        categorical_cols: list = None
    ):
        """
        Pipeline 2: impute -> clip_outliers (no time, no categoricals)
        """
        tgt = self._tgt_name(target)

        df = self.impute_missing_values(df, strategy="mean")

        features = self._filter_existing_features(df, features)

        X = df[features]
        y = df[tgt]

        X = self.clip_outliers(X, lower_quantile=0.01, upper_quantile=0.99)

        return self._coerce_xy(X, y)


    def apply_third_pipeline(
        self,
        df: pd.DataFrame,
        features: list,
        target: list,
        categorical_cols: list = None
    ):
        """
        Pipeline 3: fill_missing_timestamps -> impute (no outlier clipping, no categoricals)
        """
        tgt = self._tgt_name(target)

        df = self.fill_missing_timestamps(df, time_col="Date", freq="D")
        df = self.impute_missing_values(df, strategy="mean")

        features = self._filter_existing_features(df, features)

        X = df[features]
        y = df[tgt]

        return self._coerce_xy(X, y)


    def apply_fourth_pipeline(
        self,
        df: pd.DataFrame,
        features: list,
        target: list,
        categorical_cols: list = None
    ):
        """
        Pipeline 4: impute -> clip_outliers (no time, no categoricals)
        """
        tgt = self._tgt_name(target)

        df = self.impute_missing_values(df, strategy="mean")

        features = self._filter_existing_features(df, features)

        X = df[features]
        y = df[tgt]

        X = self.clip_outliers(X, lower_quantile=0.01, upper_quantile=0.99)

        return self._coerce_xy(X, y)


    def apply_fifth_pipeline(
        self,
        df: pd.DataFrame,
        features,
        target,
        categorical_cols=None,
        lags: int = 2
    ):
        """
        Pipeline 5: time-series lag features on target only.
        """
        df = self.fill_missing_timestamps(df, time_col="Date", freq="D")
        df = self.impute_missing_values(df, strategy="mean")

        tgt = self._tgt_name(target)
        y_df = df[[tgt]].copy()

        X_from_y, y_vec = self.create_lag_features(y_df, col=tgt, lags=lags)

        return self._coerce_xy(X_from_y, y_vec)


    def apply_sixth_pipeline(
        self,
        df: pd.DataFrame,
        features,
        target,
        categorical_cols=None,
        window: int = 2
    ):
        """
        Pipeline 6: time-series rolling features on target only.
        """
        df = self.fill_missing_timestamps(df, time_col="Date", freq="D")
        df = self.impute_missing_values(df, strategy="mean")

        tgt = self._tgt_name(target)
        y_df = df[[tgt]].copy()

        X_from_y, y_vec = self.create_rolling_features(y_df, col=tgt, window=window)

        return self._coerce_xy(X_from_y, y_vec)


    def apply_seventh_pipeline(
        self,
        df: pd.DataFrame,
        features,
        target,
        categorical_cols=None,
        lags: int = 2,
        window: int = 2
    ):
        """
        Pipeline 7: combine lag + rolling features on target only (no fill_missing_timestamps here).
        """
        tgt = self._tgt_name(target)
        y_df = df[[tgt]].copy()

        X_lags, y_lags = self.create_lag_features(y_df.copy(), col=tgt, lags=lags)
        X_roll, y_roll = self.create_rolling_features(y_df.copy(), col=tgt, window=window)
        return self._coerce_xy(X_lags, y_lags)

    def apply_eighth_pipeline(
        self,
        df,
        features,
        target,
        categorical_cols=None,
        lags: int = 2,
        window: int = 2
    ):
        """
        Pipeline 8: fill_missing_timestamps -> impute -> then combined lag+rolling pipeline.
        Works on the target only (like 5/6/7).
        """
        df = self.fill_missing_timestamps(df, time_col="Date", freq="D")
        df = self.impute_missing_values(df, strategy="mean")

        return self.apply_seventh_pipeline(
        df.copy(),
        features,
        target,
        categorical_cols,
        lags=lags,
        window=window,)


    
    

    def get_predefined_pips(self, selected_pip=None)->dict:
        """
        Group all predifined pipelines in dict
        Args:
            None
        Returns:
            dict: A dict of all predefined pipelines
        """
        pipelines_dict = {
            "apply_first_pipeline": {'pip_steps': ['fill_missing_timestamps','impute_missing_values','clip_outliers','encode_categoricals'],
                'pip_func':self.apply_first_pipeline},

            "apply_second_pipeline" : {'pip_steps': ['fill_missing_timestamps','impute_missing_values','clip_outliers'],
                'pip_func':self.apply_second_pipeline},

            "apply_third_pipeline" : {'pip_steps': ['fill_missing_timestamps','impute_missing_values'],
                'pip_func':self.apply_third_pipeline},

            "apply_fourth_pipeline" : {'pip_steps': ['impute_missing_values','clip_outliers'],
                'pip_func':self.apply_fourth_pipeline},

            "apply_fifth_pipeline" : {'pip_steps':  ['fill_missing_timestamps', 'impute_missing_values','create_lag_features'],
                'pip_func':self.apply_fifth_pipeline},

            "apply_sixth_pipeline" : {'pip_steps': ['fill_missing_timestamps','impute_missing_values','create_rolling_features'],
                'pip_func':self.apply_sixth_pipeline},

            "apply_seventh_pipeline" : {'pip_steps': ['create_rolling_features', 'impute_missing_values'],
                'pip_func':self.apply_seventh_pipeline},

             "apply_eighth_pipeline" : {'pip_steps': ['fill_missing_timestamps','impute_missing_values','create_rolling_features', 'impute_missing_values'],
                'pip_func':self.apply_eighth_pipeline}   
        }
        return pipelines_dict if selected_pip is None else pipelines_dict[selected_pip]



    def generate_random_pipelines(self, num_rnd_pips:int)->dict:
        """
        Generate num_rnd_pips random pipelines from predefined pipelines
        Args:
            None
        Returns:
            dict: A dict of num_rnd_pips predefined pipelines
        """
        pips = self.get_predefined_pips()
        rnd_pips = dict(random.sample(list(pips.items()), num_rnd_pips))
        return rnd_pips 
    
   
    def get_random_pipelines(self, num_rnd_pips:int)->dict:
        return self.generate_random_pipelines(num_rnd_pips=num_rnd_pips)
        

# inside class RegressionPipeline
    def _coerce_xy(self, X_feat: pd.DataFrame, y_vec) -> tuple[pd.DataFrame, pd.Series]:
        # ---> force to 1-D series
        if isinstance(y_vec, pd.DataFrame):
            if y_vec.shape[1] != 1:
                raise ValueError("y must be a single column for regression.")
            y_vec = y_vec.iloc[:, 0]
        else:
            y_vec = np.asarray(y_vec).reshape(-1)  # ravel any (n,1) or (1,1)
            y_vec = pd.Series(y_vec)

        # drop non-finite in y
        y_vec = y_vec.astype(float)
        valid = np.isfinite(y_vec.values)
        y_vec = y_vec.loc[valid]

        # align X to y index/length
        if not isinstance(X_feat, pd.DataFrame):
            X_feat = pd.DataFrame(X_feat)
        if len(X_feat) != len(y_vec):
            # if X and y share index, align; else just trim to min length
            if hasattr(X_feat, "index") and hasattr(y_vec, "index") and X_feat.index.equals(y_vec.index):
                pass
            else:
                n = min(len(X_feat), len(y_vec))
                X_feat = X_feat.iloc[:n].copy()
                y_vec = y_vec.iloc[:n].copy()

        # clean X
        X_feat = X_feat.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")
        X_feat = X_feat.dropna(axis=0, how="any")
        y_vec = y_vec.loc[X_feat.index]

        if len(y_vec) < 2:
            raise ValueError("Not enough samples after preprocessing; need at least 2.")

        return X_feat, y_vec
    
    def _detect_time_column(self, df: pd.DataFrame) -> Tuple[bool, str | None]:
        """
        Try to detect a genuine time column.

        Rules:
        - Prefer explicit 'Date', 'date', etc.
        - Accept true datetime64 dtypes.
        - Optionally accept object/string columns that parse cleanly as dates.
        - DO NOT treat pure numeric columns (int/float/bool) as time.
        """
        # 1) Priority: explicit names
        candidates = ["Date", "date", "datetime", "timestamp", "time"]
        for c in candidates:
            if c in df.columns:
                try:
                    pd.to_datetime(df[c], errors="raise")
                    return True, c
                except Exception:
                    pass

        # 2) True datetime64 dtypes
        for c in df.columns:
            if np.issubdtype(df[c].dtype, np.datetime64):
                return True, c

        # 3) Object/string columns that look like dates
        for c in df.columns:
            dtype = df[c].dtype
            # only attempt parsing on non-numeric, non-bool
            if np.issubdtype(dtype, np.number) or np.issubdtype(dtype, np.bool_):
                continue

            try:
                parsed = pd.to_datetime(df[c], errors="raise")
                # heuristic: require at least a few unique non-na dates
                if parsed.notna().sum() >= 3:
                    return True, c
            except Exception:
                continue

        # 4) Nothing detected
        return False, None


    def _has_gaps_or_duplicates(self, df: pd.DataFrame, time_col: str, freq: str = "D") -> bool:
        try:
            s = pd.to_datetime(df[time_col]).sort_values()
            dupes = s.duplicated().any()
            # If resampling introduces new rows (gaps), len(asfreq) > unique timestamps
            idx = pd.DatetimeIndex(s.unique())
            gaps = len(pd.Series(1, index=idx).asfreq(freq)) > len(idx)
            return bool(dupes or gaps)
        except Exception:
            return False

    def _has_nans(self, df: pd.DataFrame) -> bool:
        return df.isna().any().any()

    def _has_categoricals(self, df: pd.DataFrame, features: List[str]) -> bool:
        f = [c for c in features if c in df.columns]
        if not f:
            return False
        dtypes = df[f].dtypes
        return any(dt == "object" or str(dt).startswith("category") for dt in dtypes)

    def _outlier_ratio_high(self, df: pd.DataFrame, features: List[str], ql=0.01, qu=0.99, thresh=0.05) -> bool:
        # percent of cells outside [ql, qu] across numeric feature matrix
        num = df[features].select_dtypes(include=[np.number])
        if num.empty:
            return False
        low = num.quantile(ql)
        up = num.quantile(qu)
        mask = (num.lt(low, axis=1) | num.gt(up, axis=1))
        ratio = mask.to_numpy().mean()
        return ratio >= thresh

    def _autocorr_strong(self, df: pd.DataFrame, target: str, max_lag: int = 24, thresh: float = 0.5) -> bool:
        if target not in df.columns:
            return False
        s = pd.to_numeric(df[target], errors="coerce").dropna()
        if len(s) < max_lag + 5:
            return False
        vals = []
        for lag in range(1, max_lag + 1):
            try:
                v = s.autocorr(lag)
                if pd.notna(v):
                    vals.append(abs(v))
            except Exception:
                pass
        return (max(vals) if vals else 0.0) >= thresh

    def _volatility_high(self, df: pd.DataFrame, target: str, roll: int = 7, thresh: float = 0.35) -> bool:
        if target not in df.columns:
            return False
        s = pd.to_numeric(df[target], errors="coerce").dropna()
        if len(s) < roll * 3:
            return False
        # coefficient of variation on rolling std vs level; or use short-term noise proxy
        roll_std = s.rolling(roll).std()
        roll_mean = s.rolling(roll).mean().abs()
        cv = (roll_std / (roll_mean.replace(0, np.nan))).dropna()
        return (cv.median() if not cv.empty else 0.0) >= thresh

    def _is_clean_timeseries(self, df: pd.DataFrame, time_col: str | None, target: str) -> bool:
        # "cas propre": has time col, few NaNs, low outliers, low vol
        has_time = time_col is not None
        few_nans = not self._has_nans(df)
        low_outliers = not self._outlier_ratio_high(df, [c for c in df.columns if c != target])
        low_vol = not self._volatility_high(df, target)
        return has_time and few_nans and low_outliers and low_vol

    def analyze_dataset_signals(
        self,
        df: pd.DataFrame,
        features: List[str],
        target: str
    ) -> Dict:
        # Keep only existing features, and don't duplicate target
        safe_features = [f for f in features if f in df.columns and f != target]

        has_time, time_col = self._detect_time_column(df)

        if target in df.columns:
            sub_df = df[safe_features + [target]].copy()
        else:
            sub_df = df[safe_features].copy()

        signals = {
            "has_time_col": has_time,
            "time_col": time_col,
            "holes_or_dupes": self._has_gaps_or_duplicates(df, time_col, "D") if has_time else False,
            "has_nans": self._has_nans(sub_df),
            "has_categoricals": self._has_categoricals(df, safe_features),
            "outlier_ratio_high": self._outlier_ratio_high(df, safe_features),
            "autocorr_strong": self._autocorr_strong(df, target),
            "volatility_high": self._volatility_high(df, target),
        }
        signals["clean_ts"] = self._is_clean_timeseries(
            df,
            time_col if has_time else None,
            target
        )
        return signals


    def _maybe_rename_time_col_to_Date(self, df: pd.DataFrame, time_col: str | None) -> pd.DataFrame:
        if time_col and time_col != "Date":
            df = df.rename(columns={time_col: "Date"})
        return df

    def suggest_pipelines(self, df: pd.DataFrame, features: List[str], target: str, top_k: int = 4):
        """Score pipelines based on dataset signals; return ranked list (desc)."""
        sig = self.analyze_dataset_signals(df, features, target)
        pips = list(self.get_predefined_pips().keys())
        score = {p: 0 for p in pips}
    
        # ---- Mapping from your rules ----
        # Fill_missing_timestamps
        if sig["has_time_col"] and sig["holes_or_dupes"]:
            for p in ["apply_first_pipeline","apply_third_pipeline","apply_fifth_pipeline",
                      "apply_sixth_pipeline","apply_eighth_pipeline"]:
                score[p] += 2
    
        # Impute_missing_values
        if sig["has_nans"]:
            for p in pips:
                score[p] += 1  # "presque toutes"
    
        # Encode_categoricals
        if sig["has_categoricals"]:
            score["apply_first_pipeline"] += 2
    
        # Clip_outliers
        if sig["outlier_ratio_high"]:
            for p in ["apply_first_pipeline","apply_second_pipeline","apply_fourth_pipeline"]:
                score[p] += 2
    
        # Lag_features
        if sig["autocorr_strong"]:
            for p in ["apply_fifth_pipeline","apply_seventh_pipeline","apply_eighth_pipeline"]:
                score[p] += 2
    
        # Rolling_features
        if sig["volatility_high"]:
            for p in ["apply_sixth_pipeline","apply_seventh_pipeline","apply_eighth_pipeline"]:
                score[p] += 2
    
        # cas “propre” série temporelle
        if sig["clean_ts"]:
            score["apply_third_pipeline"] += 3
    
        # pas de colonne temps
        if not sig["has_time_col"]:
            for p in ["apply_second_pipeline","apply_fourth_pipeline"]:
                score[p] += 3
    
        # ---- Base ranking by score ----
        ranked = sorted(score.items(), key=lambda kv: kv[1], reverse=True)
        ranked_names = [k for k, v in ranked if v > 0]
        if not ranked_names:
            ranked_names = ["apply_third_pipeline","apply_second_pipeline","apply_first_pipeline"]
    
        # ---- Safety inclusions (force coverage of regimes) ----
        must_have = []
        # Prefer explicit ordering for must-haves
        if sig["has_time_col"] and (sig["autocorr_strong"] or sig["volatility_high"]):
            # Include both lag & rolling families (and combos if available)
            for p in ["apply_fifth_pipeline","apply_sixth_pipeline",
                      "apply_seventh_pipeline","apply_eighth_pipeline"]:
                if p in pips and p not in must_have:
                    must_have.append(p)
        if not sig["has_time_col"]:
            for p in ["apply_second_pipeline","apply_fourth_pipeline"]:
                if p in pips and p not in must_have:
                    must_have.append(p)
    
        # ---- Merge: must-haves first, then ranked (dedup, preserve order) ----
        final = []
        for p in must_have + ranked_names:
            if p in pips and p not in final:
                final.append(p)
    
        # ---- Select top_k but guarantee both 5th & 6th when required ----
        selected = final[:top_k]
        need_both = sig["has_time_col"] and (sig["autocorr_strong"] or sig["volatility_high"])
        if need_both:
            needed = {"apply_fifth_pipeline","apply_sixth_pipeline"} & set(pips)
            missing = list(needed - set(selected))
            for m in missing:
                if len(selected) < top_k:
                    selected.append(m)
                else:
                    # replace from the end the first item that's not one of the needed ones
                    replaced = False
                    for i in range(top_k - 1, -1, -1):
                        if selected[i] not in needed:
                            selected[i] = m
                            replaced = True
                            break
                    if not replaced and m not in selected:
                        selected[-1] = m  # last resort
    
        return selected, sig
    


    def _filter_existing_features(self, df: pd.DataFrame, features: list) -> list:
            """
            Keep only features that actually exist in df; log the ones that disappeared
            after preprocessing (e.g. due to numeric_only or dropping).
            """
            existing = [f for f in features if f in df.columns]
            missing = [f for f in features if f not in df.columns]

            if missing:
                print(f"[WARN] These features are missing after preprocessing and will be ignored: {missing}")

            if not existing:
                raise ValueError(
                    "[ERROR] No valid features left after preprocessing. "
                    f"DataFrame columns: {list(df.columns)}"
                )
            return existing


    class TimeSeriesNormalizer:
        """
        Handles normalization and denormalization for multivariate time series
        data split into (X, y) sets.
        """
      
        def __init__(self, method="standard"):
            if method == "standard":
                self.x_scaler = StandardScaler()
                self.y_scaler = StandardScaler()
            else:
                self.x_scaler = MinMaxScaler()
                self.y_scaler = MinMaxScaler()
            self.fitted = False

        def fit(self, X_train, y_train):
            """Fit scalers only on the training data."""
            self.x_scaler.fit(X_train)
            self.y_scaler.fit(y_train)
            self.fitted = True

        def transform(self, X, y=None):
            """Transform inputs (and optionally outputs) using fitted scalers."""
            if not self.fitted:
                raise RuntimeError("You must call fit() before transform().")

            X_scaled = self.x_scaler.transform(X)
            y_scaled = None
            if y is not None:
                y_scaled = self.y_scaler.transform(y)
            return X_scaled, y_scaled

        def inverse_transform_y(self, y_scaled):
            """Convert normalized target(s) back to original scale."""
            if not self.fitted:
                raise RuntimeError("You must call fit() before inverse_transform_y().")
            return self.y_scaler.inverse_transform(y_scaled)
    




# if __name__ == "__main__":

    

#     # Sample data
#     df = pd.DataFrame({
#         "datetime": pd.date_range("2023-01-01", periods=10, freq="D"),
#         "value": [1, 2, np.nan, 4, 50, 6, 7, np.nan, 9, 10],
#         "height": [1, np.nan, 3, 4, 50, 6, 7, np.nan, 9, 10],
#         "category": ["A", "B", "A", "B", "A", "B", "A", "A", "B", "B"]
#     })


#     # Initialize preprocessor
#     preproc = RegressionPipeline()
#     # preproc.generate_random_pipelines(3)
#     # Apply preprocessing steps
#     df = preproc.fill_missing_timestamps(df, time_col="datetime", freq="D")


#     df = preproc.impute_missing_values(df, strategy="mean")
#     print("df ", df)
    

#     df = preproc.clip_outliers(df)
#     df = preproc.encode_categoricals(df, categorical_cols=["category"])
#     df = preproc.scale_features(df, method="standard")
#     print("jee usi s isd")
#     df = preproc.create_lag_features(df, col="value", lags=2)
#     print("fkekfef")
#     df = preproc.create_rolling_features(df, col="value", window=3)

#     print("df ", df)
    

#     # Train-test split
#     train, test = preproc.time_series_split(df, test_size=0.2)

#     # Supervised windowing for forecasting
#     X, y = preproc.create_supervised_windows(train, target_col="value", window_size=3)

#     print("Train shape:", train.shape)
#     print("Supervised dataset:", X.shape, y.shape)


    