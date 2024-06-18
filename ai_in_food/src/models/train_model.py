from sklearn.ensemble import RandomForestClassifier
from src.data.utils import (
    get_general_path,
    join_paths,
    save_as_pickle,
    read_data
)


PROCESSED_DATA_PATH = "data/processed"
MODELS_PATH = "models"
MODEL_FILE = "model.pkl"


X_TRAIN_PATH = 'x_train.parquet'
Y_TRAIN_PATH = 'y_train.parquet'
X_TEST_PATH = 'x_test.parquet'
Y_TEST_PATH = 'y_test.parquet'


def train_model(model, x_train, y_train):
    model.fit(x_train, y_train)
    return model


if __name__ == '__main__':
    general_path = get_general_path()
    processed_path = join_paths(general_path, PROCESSED_DATA_PATH)
    x_train_path = join_paths(processed_path, X_TRAIN_PATH)
    y_train_path = join_paths(processed_path, Y_TRAIN_PATH)
    model_path = join_paths(general_path, MODELS_PATH, MODEL_FILE)

    x_train = read_data(x_train_path)
    y_train = read_data(y_train_path)

    model = RandomForestClassifier()
    trained_model = train_model(model, x_train, y_train)
    save_as_pickle(what=trained_model, where=model_path)
