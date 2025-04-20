     """
     Model configuration for different pre-trained models
     """
     
     # Financial models configuration
     FINANCIAL_MODELS = {
         # Standard financial models
         "finbert": {
             "name": "yiyanghkust/finbert-pretrain",
             "description": "BERT model pre-trained on financial text",
             "size": "base",
             "quantization": None
         },
         "fingpt": {
             "name": "bloomberg/bloomberggpt-llama-7b",
             "description": "Bloomberg's financial model based on LLaMA",
             "size": "7B",
             "quantization": 8
         },
         "bloomberg": {
             "name": "bloomberg/bloomberggpt",
             "description": "Bloomberg's 50B parameter financial LLM",
             "size": "50B",
             "quantization": 4
         },
         "fingpt-small": {
             "name": "ProsusAI/finbert",
             "description": "Smaller FinBERT variant",
             "size": "base",
             "quantization": None
         }
     }
     
     # Medical models configuration
     MEDICAL_MODELS = {
         "clinicalbert": {
             "name": "medicalai/ClinicalBERT",
             "description": "BERT model trained on clinical text",
             "size": "base",
             "quantization": None
         },
         "bio_clinicalbert": {
             "name": "emilyalsentzer/Bio_ClinicalBERT",
             "description": "ClinicalBERT variant trained on biomedical text",
             "size": "base",
             "quantization": None
         },
         "meditron": {
             "name": "epfl-llm/meditron-7b",
             "description": "7B parameter medical LLM",
             "size": "7B",
             "quantization": 8
         },
         "med-llama": {
             "name": "medalpaca/medalpaca-13b",
             "description": "Medical variant of LLaMA",
             "size": "13B",
             "quantization": 4
         }
     }
     
     # Mapping of model names to configurations
     MODEL_CONFIGS = {**FINANCIAL_MODELS, **MEDICAL_MODELS}
     
     def get_model_config(model_name):
         """Get configuration for a specific model"""
         return MODEL_CONFIGS.get(model_name, None)