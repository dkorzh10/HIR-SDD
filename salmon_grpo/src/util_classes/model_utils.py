class ModelUtils:
    """Model utility functions"""
    
    def __init__(self, use_distributed):
        self.use_distributed = use_distributed
        
    def unwrap_dist_model(self, model):
        if self.use_distributed:
            return model.module
        else:
            return model
