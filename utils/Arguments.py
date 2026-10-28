import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Arguments:
# -- Mapping entre ID de use case, colonne cible et titre du use case ---
    USECASE_MAP = {
    'UC-IP':  {'target': 'injury',   'title': 'Prédiction de blessures'},
    'UC-RPE': {'target': 'RPE',      'title': 'Prédiction de la perception de l’effort'},
    'UC-PRM': {'target': 'result',   'title': 'Prédiction des résultats de matchs'},
    'UC-REC': {'target': 'player_id','title': 'Recrutement de joueurs'}
    }


    
    api_key:str = os.getenv("OPENAI_API_KEY", "")
    config_path: str = "api.yaml"
    usecases_path: str = "./datasets/systmPrompt/usecases_yassine.md"
    model: str = "gpt-4-turbo"
    
    global_conxtexts = {"Classification":0,"Regression":1,"Time Series Forecasting":2,"Computer Vision" :3,"Annotation":4,"Clustering":6,"Anomaly Detection":6,"Audio Processing":7 }  

    usecases = {
        "Injury Prediction": {
            "title": "Prédiction de blessures",
            "goal": "Estimer quotidiennement la probabilité qu’un athlète se blesse.",
            "context": "Time Series Forecasting"
        },
        "RPE Prediction": {
            "title": "Prédiction de la perception de l’effort",
            "goal": "Anticiper le Rate of Perceived Exertion avant chaque séance.",
            "context": "Regression"
        },
        "Match Result Prediction": {
            "title": "Prédiction des résultats de matchs",
            "goal": "Estimer la probabilité de victoire, nul ou défaite avant un match.",
            "context": "Classification"
        },
        "Recruitment": {
            "title": "Recrutement de joueurs",
            "goal": "Identifier et évaluer automatiquement des profils adaptés aux besoins de l’équipe.",
            "context": "Clustering"
    }     
    }

                         

    # dataset_path = "./datasets/day_approach_maskedID_timeseries.csv"
    dataset_path = "./datasets/df_7days_mean.csv"
    # def print_args(self):
    #     print(
    #         "Args:\n val1: {0}\n val2:{1}\n val3:{2}\n "
    #         .format(
    #         self.a,
    #         self.b,
    #         self.c,            
    #         )
    #     )




    #regression 
    device = 'cpu'
    n_epochs = 10
    save_optimal_model = False
    save_model_PATH = False
    batch_size = 32
    shuffle = True
    lr=0.001
    selected_model = 'GRU'
    time_steps = 5

     #dataset params
    num_series = 4
    pred_len = 1
    seq_len = 5
    device = 'cpu'


    cross_validation = False
    cross_val_n_splits = 5
    ml_model_selected = False # cross valid function for none ML models to be written




    rgr_models = {
        'ML_models': [
            'XGBRegressor','MLPRegressor','RandomForestRegressor', 
        ],
        'DL_models': ['GRU_Autoregressive', 'N_Linear']
    }
    #selection modeling approach
    ml_model_selected = list(rgr_models.keys())[1] #choose ML/DP# cross valid function for none ML models to be written
    #select model
    selected_model = rgr_models[ml_model_selected][0] #choose ML/DP model

    m_params = {
        'shared_m_params':{
            'num_series' : 1,
            'seq_len' : 1, 
            'pred_len' : 1 
        },
        'XGBRegressor': {
            'objective':'reg:squarederror',
            'n_estimators': 20, 
            'learning_rate': 0.06,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.08,
        },

        'MLPRegressor': {
        'hidden_layer_sizes': (64, 64),
        'activation': 'relu',
        'solver': 'adam',
        'max_iter': 500,
        'learning_rate_init': 0.001,
        'momentum': 0.8,
        'random_state': 42,
        },

        'RandomForestRegressor': {
            'criterion': 'squared_error',
            'max_depth': 100,
            'n_estimators': 200,
            'min_samples_split': 4,
            'min_samples_leaf': 10,
            'max_features': 'sqrt',
            'bootstrap': True,
            'random_state': 42,
        },

        'GRU_Autoregressive': {
            'hidden_size': 64,
            'num_layers': 1,
            'output_size': 1
        },

        'N_Linear': {
            'individual': False,
            'device': device,
        },
    }
   




arguments =  Arguments()
# arguments.print_args()

