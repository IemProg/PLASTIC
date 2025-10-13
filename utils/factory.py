def get_model(model_name, args):
    name = model_name.lower()
    if name=="simplecil":
        from models.simplecil import Learner
        return Learner(args)
    elif name=="slca":
        from models.SLCA import SLCA
        return SLCA(args)
    elif name=="adapt_tta":
        from models.adapt_tta import Learner
        return Learner(args)
    elif name=="layeradapt":
        from models.layeradapt import Learner
        return Learner(args)
    elif name=="finetune":
        from models.finetune import Learner
        return Learner(args)
    elif name=="adapter":
        from models.finetune import Learner
        return Learner(args)
    elif name == "fsil":
        from models.FSIL import Learner
        return Learner(args)
    elif name == 'ranpac':
        from models.ranpac import Learner
        return Learner(args)
    elif name == 'adam_adapter':
        from models.adam_adapter import Learner
        return Learner(args)
    elif name == "coda_prompt":
        from models.coda_prompt import Learner
        return Learner(args)
    elif name == 'fecam':
        from models.fecam import Learner
        return Learner(args)
    else:
        print("No mode defined")
        assert 0
