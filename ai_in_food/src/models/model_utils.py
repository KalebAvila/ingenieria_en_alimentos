from src.data.utils import (
    get_general_path,
    join_paths,
    read_pickle_with_pandas,
)

non_predictor_columns = ['similar']
id_columns = ['fdc_id_source', 'fdc_id_target']
MODELS_PATH = "models"


def get_model(filename='model.pkl'):
    general_path = get_general_path()
    model_path = join_paths(general_path, MODELS_PATH, filename)
    model = read_pickle_with_pandas(model_path)
    return model


def predict(model, dataset):
    pred = model.predict_proba(dataset)[:, 1]
    return pred
