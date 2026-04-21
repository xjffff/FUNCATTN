from model.simple_transformer import SimpleTransformer


def get_model(args):
    model_dict = {
        'SimpleTransformer': SimpleTransformer,
    }
    return model_dict[args.model]
