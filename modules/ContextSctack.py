
from dataclasses import dataclass
@dataclass
class ContextSctack:
    user_prompt = ''
    #user_dataset = ''
    use_case = ''
    global_context= ''
    data_profiling = {
        'features': '',  #identifier les colonnes pertinentes pour la prédiction
        'target' : '',
        'user_role' : ''
    }

    preproc_steps = {
        '' : {}
    }
    pipelines = {}
    optimal_pipelines = {'' : {}}
    preproc_models = {}
    optimal_preproc_models = {}
    results = {
        'num_experims' : '',

        'optimal_preproc' : {
            'pipeline': '',
            'model': ''
        },
        'metrics' : {}

    }



     