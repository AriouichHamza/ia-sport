import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from typing import Optional, Union, Dict, Any, Tuple, List
from packaging.version import Version
import sklearn
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.utils import resample
import warnings


import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr
from typing import Optional




warnings.filterwarnings("ignore", category=UserWarning)


class DataAnalyzer:
    """
    Classe utilitaire pour l'analyse préliminaire de datasets :
    - Détection de valeurs manquantes
    - Détection de déséquilibre de classes
    - Vérification de la nécessité de sélection de variables
    - Vérification du besoin de réduction de dimensionnalité
    - Détection d'outliers
    """

    def __init__(self):
        pass

    # -------------------------------------------------------------------------
    def check_missing_values(self, dataset: pd.DataFrame) -> bool:
        """
        Vérifie s'il existe des valeurs manquantes dans le dataset.
        Retourne True si des valeurs manquantes sont détectées.
        """
        missing_values = dataset.isnull().sum()
        missing_values = missing_values[missing_values > 0]
        if not missing_values.empty:
            print("⚠️ Missing values found:")
            print(missing_values)
            return True
        print("✅ No missing values detected.")
        return False

    # -------------------------------------------------------------------------
    def check_unbalance_classes(self, dataset: pd.DataFrame, target: str, tolerance: float = 0.05) -> bool:
        """
        Vérifie si les classes de la variable cible sont déséquilibrées.
        Retourne True si le déséquilibre dépasse la tolérance spécifiée.
        """
        y = dataset[target]
        n_classes = y.nunique()
        dist = y.value_counts(normalize=True)
        ideal_ratio = 1 / n_classes
        deviations = abs(dist - ideal_ratio)
        max_deviation = deviations.max()

        print(f"📊 Class distribution:\n{dist}")
        print(f"⚙️  Max deviation from balance: {max_deviation:.3f}")

        if max_deviation > tolerance:
            print("⚠️ Classes are unbalanced.")
            return True
        else:
            print("✅ Classes are reasonably balanced.")
            return False

    # -------------------------------------------------------------------------
    def check_feature_selection_need(self, df: pd.DataFrame, target: str, threshold: float = 0.25) -> bool:
        """
        Détermine si une sélection de variables est nécessaire
        selon la corrélation entre les features et la cible.
        """
        y = df[target]
        X = df.drop(columns=[target]).select_dtypes(include=[np.number])
        correlations = []

        # Cas 1 : régression (target continue)
        if y.nunique() > 10:
            for col in X.columns:
                corr = X[col].corr(y)
                if not np.isnan(corr):
                    correlations.append(abs(corr))

        # Cas 2 : classification (target discrète)
        else:
            for col in X.columns:
                try:
                    corr, _ = pointbiserialr(y, X[col])
                    if not np.isnan(corr):
                        correlations.append(abs(corr))
                except Exception:
                    continue

        if not correlations:
            print("⚠️ No numeric features found for correlation analysis.")
            return False

        avg_corr = np.mean(correlations)
        print(f"📊 Average absolute correlation with target: {avg_corr:.3f}")

        if avg_corr < threshold:
            print("📉 Weak correlation detected → Feature selection recommended.")
            return True
        else:
            print("✅ Features are reasonably correlated → Feature selection not necessary.")
            return False

    # -------------------------------------------------------------------------
    def check_dimensionality_reduction_need(
        self, df: pd.DataFrame, target: Optional[str] = None, threshold: float = 0.01
    ) -> bool:
        """
        Vérifie la nécessité d'une réduction de dimensionnalité
        en fonction des corrélations entre features.
        """
        if target and target in df.columns:
            df = df.drop(columns=[target])

        corr_matrix = df.corr(numeric_only=True).abs()
        np.fill_diagonal(corr_matrix.values, 0)

        correlated_pairs = [
            (c1, c2, corr_matrix.loc[c1, c2])
            for c1 in corr_matrix.columns
            for c2 in corr_matrix.columns
            if c1 < c2 and corr_matrix.loc[c1, c2] > threshold
        ]

        if correlated_pairs:
            print(f"⚠️ Found {len(correlated_pairs)} highly correlated feature pairs:")
            for f1, f2, corr in correlated_pairs[:10]:
                print(f"   {f1} ↔ {f2}: {corr:.2f}")
            print("📉 Dimensionality reduction (e.g., PCA) might be beneficial.")
            return True
        else:
            print("✅ No strong feature correlations found. Dimensionality reduction likely unnecessary.")
            return False

    # -------------------------------------------------------------------------
    def check_outliers_need(self, df: pd.DataFrame, threshold: float = 0.05) -> bool:
        """
        Vérifie la présence d’outliers dans les variables numériques.
        Retourne True si la proportion d’outliers dépasse le seuil défini.
        """
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            print("⚠️ No numeric features to check for outliers.")
            return False

        total_values = numeric_df.size
        outlier_count = 0

        for col in numeric_df.columns:
            Q1 = numeric_df[col].quantile(0.25)
            Q3 = numeric_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = numeric_df[(numeric_df[col] < lower_bound) | (numeric_df[col] > upper_bound)]
            outlier_count += len(outliers)

        outlier_fraction = outlier_count / total_values
        print(f"📊 Fraction of values considered outliers: {outlier_fraction:.3f}")

        if outlier_fraction > threshold:
            print("📉 Significant outliers detected → Clipping or treatment recommended.")
            return True
        else:
            print("✅ Outliers not significant → Clipping not necessary.")
            return False
        
    #---------------------------------------------------------------------------------------
    def select_traitement(self, df: pd.DataFrame, target: str) -> list:
        """
        Propose automatiquement les traitements adéquats
        en fonction des besoins détectés sur le dataset.

        Retourne une liste de traitements recommandés (noms des méthodes).
        """
        print("🔍 Analyse automatique du dataset...\n")

        traitements = []

        # 1. Valeurs manquantes
        if self.check_missing_values(df):
            traitements.append("impute_missing_values")

        # 2. Corrélation features ↔ cible
        if self.check_feature_selection_need(df, target):
            traitements.append("select_features")

        # 3. Corrélations fortes entre features
        if self.check_dimensionality_reduction_need(df, target):
            traitements.append("dimensionality_reduction_lda")

        # 4. Outliers
        if self.check_outliers_need(df):
            traitements.append("clip_outliers")
        # 5. Déséquilibre des classes
        if self.check_unbalance_classes(df, target):
            traitements.append("downsample_dataframe")

        # Résumé final
        print("\n🧩 Recommandations de traitement :")
        if traitements:
            for t in traitements:
                print(f"   ✅ {t}")
        else:
            print("   👍 Aucun traitement particulier n’est nécessaire.")

        return traitements



class ClassificationPipeline:
    def __init__(self, analyzer: Optional[DataAnalyzer] = None):
        self.target = None
        self.scaler = None
        self.encoder = None
        self.label_encoder = None
        self.imputer = None
        self.feature_selector = None
        self.analyzer = analyzer if analyzer is not None else DataAnalyzer()
        
    # ------------------------------------------------------
    # 1. Handle missing values
    #    Strategies {"numerical": {'mean','median','constant'}, "categorical":{'most_frequent', 'constant'} }
    # ------------------------------------------------------
    def impute_missing_values(
       self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        *,
        phase: str = "train",
        test_params: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        if phase not in {"train", "test"}:
            raise ValueError("phase doit être 'train' ou 'test'.")

        df_copy = X.copy()

        numeric_cols = df_copy.select_dtypes(include=["number"]).columns.tolist()
        cat_cols    = df_copy.select_dtypes(include=["object", "category"]).columns.tolist()

        if phase == "train":
            learned_stats = {"num": {}, "cat": {}}

            if numeric_cols:
                num_imputer = SimpleImputer(strategy="mean")
                df_copy.loc[:, numeric_cols] = num_imputer.fit_transform(df_copy[numeric_cols])
                for col, val in zip(numeric_cols, num_imputer.statistics_):
                    learned_stats["num"][col] = val

            if cat_cols:
                cat_imputer = SimpleImputer(strategy="most_frequent")
                df_copy.loc[:, cat_cols] = cat_imputer.fit_transform(df_copy[cat_cols])
                for col, val in zip(cat_cols, cat_imputer.statistics_):
                    learned_stats["cat"][col] = val

            return df_copy, y, learned_stats

        # -------- phase test --------
        if test_params is None:
            raise ValueError("En phase 'test', fournir stats (renvoyées par la phase 'train').")

        # Remplissage vectorisé sans chained assignment
        num_stats = test_params.get("num", {})
        if num_stats:
            cols = [c for c in num_stats.keys() if c in df_copy.columns]
            if cols:
                mapping = {c: num_stats[c] for c in cols}
                df_copy.loc[:, cols] = df_copy.loc[:, cols].fillna(value=mapping)

        cat_stats = test_params.get("cat", {})
        if cat_stats:
            cols = [c for c in cat_stats.keys() if c in df_copy.columns]
            if cols:
                mapping = {c: cat_stats[c] for c in cols}
                df_copy.loc[:, cols] = df_copy.loc[:, cols].fillna(value=mapping)

        return df_copy
    # ------------------------------------------------------
    # 2. Outlier handling
    # ------------------------------------------------------
    def clip_outliers(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        *,
        phase: str = "train",
        test_params: Optional[Dict[str, pd.Series]] = None,   # {'lower': Series, 'upper': Series} appris au train
        lower_quantile: float = 0.01,
        upper_quantile: float = 0.99,
    ) :
        """
        phase='train' -> retourne (X_clipped, {'lower': q_low, 'upper': q_up})
        phase='test'  -> retourne X_clipped (en utilisant stats appris au train)
        """
        if phase not in {"train", "test"}:
            raise ValueError("phase doit être 'train' ou 'test'.")

        df_copy = X.copy()

        # colonnes numériques uniquement (on exclut bool explicitement)
        num_cols = [
            c for c in df_copy.select_dtypes(include=["number"]).columns
            if not pd.api.types.is_bool_dtype(df_copy[c])
        ]

        if phase == "train":
            if num_cols:
                q_low = df_copy.loc[:, num_cols].quantile(lower_quantile)
                q_up  = df_copy.loc[:, num_cols].quantile(upper_quantile)
                # assignation explicite, pas d'inplace sur Series -> pas de FutureWarning
                df_copy.loc[:, num_cols] = df_copy.loc[:, num_cols].clip(lower=q_low, upper=q_up, axis=1)
                learned = {"lower": q_low, "upper": q_up}
            else:
                learned = {"lower": pd.Series(dtype=float), "upper": pd.Series(dtype=float)}
            return df_copy, y, learned

        # ---- phase test ----
        if test_params is None or "lower" not in test_params or "upper" not in test_params:
            raise ValueError("En phase 'test', fournir stats={'lower': Series, 'upper': Series} issus du train.")

        if num_cols:
            # aligne sur l'intersection des colonnes présentes
            cols = [c for c in num_cols if c in test_params["lower"].index and c in test_params["upper"].index]
            if cols:
                q_low = test_params["lower"][cols]
                q_up  = test_params["upper"][cols]
                df_copy.loc[:, cols] = df_copy.loc[:, cols].clip(lower=q_low, upper=q_up, axis=1)

        return df_copy
    # ------------------------------------------------------
    # 3. Scaling/Normalization
    # ------------------------------------------------------
    def scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame, method: str = "standard") -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Scales numeric features of the training and test datasets using StandardScaler.
        
        Args:
            X_train (pd.DataFrame): Training feature set.
            X_test (pd.DataFrame): Test feature set.
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Scaled X_train and X_test as pandas DataFrames.
        """
        scaler = StandardScaler() if method == "standard" else MinMaxScaler()
        
        # Fit scaler only on training data
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Transform test data with the same scaler
        X_test_scaled = scaler.transform(X_test)
        
        # Convert back to DataFrame to preserve column names and indices
        X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
        
        return X_train_scaled_df, X_test_scaled_df
    # ------------------------------------------------------
    # 4. Categorical encoding
    # ------------------------------------------------------
    def encode_categoricals(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        *,
        phase: str = "train",
        test_params: Optional[OneHotEncoder] = None,        # fourni en phase='test'
        categorical_cols: Optional[List[str]] = None,   # si None -> détection auto au train
        drop: Union[str, None] = "first",               # 'first' pour éviter la colinéarité
    ) -> Union[Tuple[pd.DataFrame, OneHotEncoder], pd.DataFrame]:
        """
        phase='train' -> retourne (X_encoded, fitted_encoder)
        phase='test'  -> retourne X_encoded (utilise l'encoder appris au train)

        - Détecte automatiquement les colonnes catégorielles au train si non fournies.
        - Aligne le test sur les colonnes catégorielles apprises via encoder.feature_names_in_.
        - handle_unknown='ignore' pour éviter les erreurs si des catégories inédites apparaissent au test.
        """
        if phase not in {"train", "test"}:
            raise ValueError("phase doit être 'train' ou 'test'.")

        df = X.copy()

        # --- Détection des colonnes catégorielles (au TRAIN seulement si non spécifiées) ---
        if phase == "train":
            if categorical_cols is None:
                categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
            # Rien à encoder → renvoyer tel quel + encoder None
            if not categorical_cols:
                return df, y, 1

            # Compatibilité sklearn: sparse_output (>=1.2) vs sparse (anciennes versions)
            enc_kwargs: Dict[str, Any] = dict(handle_unknown="ignore", drop=drop)
            if Version(sklearn.__version__) >= Version("1.2"):
                enc_kwargs["sparse_output"] = False
            else:
                enc_kwargs["sparse"] = False

            ohe = OneHotEncoder(**enc_kwargs)
            encoded = ohe.fit_transform(df[categorical_cols])

            encoded_df = pd.DataFrame(
                encoded,
                columns=ohe.get_feature_names_out(categorical_cols),
                index=df.index,
            )

            # Supprimer les colonnes cat originales et concaténer
            df = df.drop(columns=categorical_cols)
            df = pd.concat([df, encoded_df], axis=1)

            return df, y, ohe

        # ---------------------- phase == 'test' ----------------------
        if test_params is None:
            raise ValueError("En phase 'test', fournir l'encoder appris au train.")

        # Colonnes catégorielles attendues par l'encoder (dans le même ordre)
        expected_cat_cols = list(getattr(test_params, "feature_names_in_", []))
        if not expected_cat_cols:
            # Le train n'avait aucune colonne catégorielle → on renvoie tel quel
            return df

        # Réindexer les colonnes catégorielles pour correspondre au train (colonnes manquantes -> NaN)
        X_cat = df.reindex(columns=expected_cat_cols, fill_value=np.nan)

        encoded = test_params.transform(X_cat)
        encoded_df = pd.DataFrame(
            encoded,
            columns=test_params.get_feature_names_out(expected_cat_cols),
            index=df.index,
        )

        # Retirer TOUTES les colonnes catégorielles originales présentes dans df
        drop_cols = [c for c in expected_cat_cols if c in df.columns]
        df = df.drop(columns=drop_cols)

        df = pd.concat([df, encoded_df], axis=1)
        return df
    # ------------------------------------------------------
    # 5. Encode target labels
    # ------------------------------------------------------
    def encode_labels(self, y: pd.Series) -> pd.Series:
        self.label_encoder = LabelEncoder()
        arr = self.label_encoder.fit_transform(y)
        return pd.Series(arr, index=y.index, name=y.name)
    # ------------------------------------------------------
    # 6. Feature selection (low variance)
    # ------------------------------------------------------





    def select_features_with_artifacts(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        *,
        phase: str = "train",
        test_params: Optional[Dict[str, Any]] = None,  # {'selector', 'cols_in', 'selected_cols'}
        threshold: float = 0.0,
    ):
        df = X.copy()
        num_cols = df.select_dtypes(include=["number", "bool"]).columns.tolist()

        if phase == "train":
            if not num_cols:
                return df, y, {"selector": None, "cols_in": [], "selected_cols": []}

            sel = VarianceThreshold(threshold=threshold)
            sel.fit(df[num_cols])

            support = sel.get_support()
            selected_cols = list(pd.Index(num_cols)[support])

            X_sel = pd.DataFrame(sel.transform(df[num_cols]), columns=selected_cols, index=df.index)
            return X_sel, y, {"selector": sel, "cols_in": num_cols, "selected_cols": selected_cols}

        # ---- test ----
        if test_params is None or test_params.get("selector", None) is None:
            # aucun fit : on renvoie tel quel
            return df

        sel: VarianceThreshold = test_params["selector"]
        cols_in: List[str] = test_params.get("cols_in", num_cols)
        selected_cols: List[str] = test_params.get("selected_cols", [])

        # réaligner le test sur les colonnes d'entrée du train
        X_num = df.reindex(columns=cols_in, fill_value=0.0)
        X_sel_arr = sel.transform(X_num)

        # reconstruire DataFrame avec les noms retenus au train
        X_sel = pd.DataFrame(X_sel_arr, columns=selected_cols, index=df.index)
        return X_sel
    # ------------------------------------------------------
    # 7. Train/test splitting
    # ------------------------------------------------------
    def classification_split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        stratify: bool = True,
        random_state: int | None = None,
    ):
        return train_test_split(
            X,
            y,
            test_size=test_size,
            stratify=y if stratify else None,
            random_state=random_state,
        )

    def get_categorical_columns(self, X: pd.DataFrame,y: pd.Series) -> list:
        """
        Returns a list of categorical column names in the DataFrame.
        
        Parameters:
            df (pd.DataFrame): Input DataFrame
        
        Returns:
            List of column names with dtype 'object' or 'category'
        """
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        return categorical_cols
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Hamza's Process from the first semester
    # See documentation for more details D4.1 S1
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #------------------------------------------------------
    #Data Balancing (Downsampling the majority class)
    #------------------------------------------------------
    def downsample_dataframe(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        *,
        phase: str = "train",
        test_params: Optional[Dict[str, Any]] = None,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]] | Tuple[pd.DataFrame, pd.Series]:
        """
        phase='train' -> (X_balanced, y_balanced, stats)
        - Sous-échantillonne chaque classe à min(counts).
        - 'stats' contient les comptes d'origine et n_per_class utilisé.
        phase='test'  -> (X, y) inchangés (pas de rééquilibrage du test).
        """
        if phase not in {"train", "test"}:
            raise ValueError("phase doit être 'train' ou 'test'.")

        if phase == "test":
            # Pas de rééquilibrage au test
            return X
        else:
            df = pd.concat([X, y], axis=1)
            # Separate majority and minority classes
            majority_class = df[y.name].value_counts().idxmax()
            minority_class = df[y.name].value_counts().idxmin()

            df_majority = df[df[y.name] == majority_class]
            df_minority = df[df[y.name] == minority_class]

            # Downsample majority class
            df_majority_downsampled = resample(
                df_majority,
                replace=False,  # sample without replacement
                n_samples=len(df_minority),  # to match minority class
                random_state=42  # for reproducibility
            )

            # Combine minority class with downsampled majority class
            df_balanced = pd.concat([df_majority_downsampled, df_minority])

            # Shuffle the dataset
            df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

            # Separate features and target
            y_balanced = df_balanced[y.name]
            X_balanced = df_balanced.drop(columns=[y.name])
            return X_balanced, y_balanced, 1
    #------------------------------------------------------
    #Dimensionality Reduction (PCA)
    #------------------------------------------------------
    def apply_pca(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        *,
        phase: str = "train",
        test_params: Optional[PCA] = None,      # fourni en phase="test"
        n_components: Optional[int] = 3 # si None -> borne auto = min(n_samples, n_features)
    ) -> Tuple[pd.DataFrame, PCA] | pd.DataFrame:
        """
        phase='train' -> (X_pca_df, fitted_pca)
        phase='test'  -> X_pca_df   (utilise le PCA appris au train)

        Notes:
        - Suppose que X est numérique (après encodage des catégorielles + imputation).
        - Borne auto: n_components <= min(n_samples, n_features).
        """
        if phase not in {"train", "test"}:
            raise ValueError("phase doit être 'train' ou 'test'.")

        Xn = X.copy()

        # Rien à faire si pas de features
        if Xn.shape[1] == 0:
            return (Xn, PCA(n_components=0)) if phase == "train" else Xn

        # Borne des composantes
        max_comp = min(Xn.shape[0], Xn.shape[1])  # contrainte scikit-learn pour PCA(int)
        if max_comp <= 0:
            return (Xn, PCA(n_components=0)) if phase == "train" else Xn
        if n_components is None or n_components > max_comp:
            n_comp = max_comp
        elif n_components < 1:
            n_comp = 1
        else:
            n_comp = n_components

        if phase == "train":
            pca_fitted = PCA(n_components=n_comp)
            X_pca = pca_fitted.fit_transform(Xn)
            cols = [f"PCA{i+1}" for i in range(pca_fitted.n_components_)]
            X_pca_df = pd.DataFrame(X_pca, index=Xn.index, columns=cols)
            return X_pca_df, y, pca_fitted

        # ----- phase test -----
        if test_params is None:
            raise ValueError("En phase 'test', fournir le PCA appris au train (paramètre pca=...).")

        X_pca = test_params.transform(Xn)
        cols = [f"PCA{i+1}" for i in range(test_params.n_components_)]
        X_pca_df = pd.DataFrame(X_pca, index=Xn.index, columns=cols)
        return X_pca_df
    #------------------------------------------------------
    #Dimensionality Reduction (LDA)
    #------------------------------------------------------
    def apply_lda(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, np.ndarray]] = None,  # requis en train
        *,
        phase: str = "train",
        test_params: Optional[LDA] = None,                         # requis en test
        n_components: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, Optional[LDA]] | pd.DataFrame:
        """
        phase='train' -> (X_lda_df, fitted_lda)   (fit sur X, y)
        phase='test'  ->  X_lda_df                (transform avec le LDA appris)

        - n_components est borné à min(n_features, n_classes-1). Si max_components <= 0, on no-op.
        - Suppose X déjà numérique (imputation + encodage faits avant).
        """
        if phase not in {"train", "test"}:
            raise ValueError("phase doit être 'train' ou 'test'.")

        Xn = X.copy()

        if phase == "train":
            if y is None:
                raise ValueError("En phase 'train', fournir y.")
            y_arr = pd.Series(y).to_numpy()

            n_features = Xn.shape[1]
            n_classes  = pd.Series(y_arr).dropna().nunique()
            max_components = max(0, min(n_features, n_classes - 1))

            # Rien à faire (pas assez de classes/features)
            if max_components <= 0:
                print("LDA no-op: pas assez de classes ou de features.")
                return Xn, y, None

            # borne auto
            if n_components is None or n_components > max_components:
                n_comp = max_components
            elif n_components < 1:
                n_comp = 1
            else:
                n_comp = n_components

            lda_fitted = LDA(n_components=n_comp)
            X_red = lda_fitted.fit_transform(Xn, y_arr)

            cols = [f"LDA{i+1}" for i in range(X_red.shape[1])] if X_red.ndim == 2 else ["LDA1"]
            X_red_df = pd.DataFrame(X_red, index=Xn.index, columns=cols)
            return X_red_df, y, lda_fitted

        # ---------- phase test ----------
        if test_params is None:
            return Xn

        X_red = test_params.transform(Xn)
        # nombre de composantes appris (fallback sur shape si attribut absent)
        n_comp = getattr(test_params, "n_components", None) or getattr(test_params, "n_components_", None) or X_red.shape[1]
        cols = [f"LDA{i+1}" for i in range(n_comp)]
        X_red_df = pd.DataFrame(X_red, index=Xn.index, columns=cols)
        return X_red_df
    
    #------------------------------------------------------
    # Pipeline Management
    #------------------------------------------------------
    def preproc_steps(self, X: pd.DataFrame, y: pd.Series, phase: str, test_params=None) -> dict:
        """
        Group all preprocessing steps in in dict
        Args:
            None
        Returns:
            dict: A dict of all possible preprocessing steps.
        """
        steps = {
            "impute_missing_values": self.impute_missing_values,
            "clip_outliers": self.clip_outliers,
            "scale_features": self.scale_features,
            "encode_categoricals": self.encode_categoricals,
            "encode_labels": self.encode_labels,
            "select_features": self.select_features_with_artifacts,
            "classification_split" : self.classification_split,
            "dimensionality_reduction_pca": self.apply_pca,
            "dimensionality_reduction_lda": self.apply_lda,
            "downsample_dataframe": self.downsample_dataframe,
        }
        return steps
    
    # ------------------------------------------------------
    # Pipeline Management & Random Selection
    # ------------------------------------------------------

    def get_predefined_pips(self, selected_pip: Optional[str] = None) -> Union[Dict[str, List[str]], List[str]]:
        """
        Returns a dictionary of named predefined pipelines or a specific one if selected_pip is provided.
        Each pipeline is a list of strings corresponding to keys in self.preproc_steps().
        """
        pipelines_dict = {
            "basic_cleaning": ["impute_missing_values", "encode_categoricals"],
            "full_preprocessing": ["impute_missing_values", "clip_outliers", "encode_categoricals", "select_features"],
            "dimensionality_reduction_pca": ["impute_missing_values", "encode_categoricals", "dimensionality_reduction_pca"],
            "dimensionality_reduction_lda": ["impute_missing_values", "encode_categoricals", "dimensionality_reduction_lda"],
            "balanced_pca": ["impute_missing_values", "encode_categoricals", "downsample_dataframe", "dimensionality_reduction_pca"],
            "balanced_lda": ["impute_missing_values", "encode_categoricals", "downsample_dataframe", "dimensionality_reduction_lda"],
            "robust_selection": ["clip_outliers", "impute_missing_values", "encode_categoricals", "select_features"],
            "minimalist": ["encode_categoricals"]
        }
        
        if selected_pip is not None:
            if selected_pip not in pipelines_dict:
                raise KeyError(f"Pipeline '{selected_pip}' not found in predefined list.")
            return pipelines_dict[selected_pip]
            
        return pipelines_dict

    def get_random_pipelines(self, num_rnd_pips: int) -> Dict[str, List[str]]:
        """
        Randomly selects N pipelines from the predefined dictionary.
        
        Args:
            num_rnd_pips (int): Number of unique pipelines to return.
            
        Returns:
            Dict[str, List[str]]: A dictionary of sampled pipelines.
        """
        import random
        all_pips = self.get_predefined_pips()
        
        # Ensure we don't try to sample more than we have
        actual_num = min(num_rnd_pips, len(all_pips))
        
        sampled_keys = random.sample(list(all_pips.keys()), actual_num)
        return {k: all_pips[k] for k in sampled_keys}

    def get_defined_pipelines(self, df: pd.DataFrame, target: str) -> List[List[str]]:
        """
        Updated to return a list of lists of steps, consistent with the loop in run_classification_model.
        """
        # Strategy: Get the analyzer recommendations
        recommended = self.analyzer.select_traitement(df, target)
        
        # Ensure mandatory encoding if not recommended
        if "encode_categoricals" not in recommended:
            recommended.append("encode_categoricals")
            
        # Also include some standard alternatives to ensure the loop has variety
        alternative = ["impute_missing_values", "encode_categoricals"]
        
        return [recommended, alternative]
    
     

    def get_defined_pipelines(self, df: pd.DataFrame, target: str) -> list:
        pipeline = self.analyzer.select_traitement(df, target)
        # If nothing recommended, at least run encoding
        if not pipeline:
            pipeline = ["encode_categoricals"]
        # Return a LIST of pipelines (because the caller loops over pipelines)
        return [pipeline]







