from utils.pipelines.RegressionPipeline import RegressionPipeline
from utils.metrics import regression_metrics 
from utils.metrics import classification_metrics 


class IntelligentModeling:
    def __init__(self):
        pass

    def query_knowledge_base():
        pass
    def get_pipelines(self):
        
        ts_processor = RegressionPipeline()
        #generate random pipelines
        random_pipelines = ts_processor.generate_random_pipelines( num_pipelines=3, min_steps=1,num_prec_steps_excluded=0) 
        #print("random pipelines", random_pipelines)
    
    def implement_pipeline(self):
        pass

    def get_models(self):
        pass
    
    def implement_models(self):
        pass
    


object = IntelligentModeling()
object.get_pipelines()

