from model import  Functional_Attention_Structured_Mesh_2D, Functional_Attnetion_Structured_Mesh_2D_Shared, Functional_Attention_Irregular_Mesh_Shared


def get_model(args):
    model_dict = {
        'Functional_Attention_Structured_Mesh_2D': Functional_Attention_Structured_Mesh_2D,
        'Functional_Attention_Structured_Mesh_2D_Shared': Functional_Attnetion_Structured_Mesh_2D_Shared,
        'Functional_Attention_Irregular_Mesh_Shared': Functional_Attention_Irregular_Mesh_Shared,
    }
    return model_dict[args.model]
