from src.features.feature_utils import RANDOM_SEED

from model_utils import (
    id_columns
)

from src.data.utils import (
    get_general_path,
    join_paths,
    save_as_parquet,
    read_data
)

from sklearn.model_selection import train_test_split

TRAIN_SIZE = 0.7

PROCESSED_DATA_PATH = "data/processed"

PROCESSED_PREDICTORS = "processed_predictors.parquet"
PROCESSED_TARGET = "processed_target.parquet"

X_TRAIN_PATH = 'x_train.parquet'
Y_TRAIN_PATH = 'y_train.parquet'
X_TEST_PATH = 'x_test.parquet'
Y_TEST_PATH = 'y_test.parquet'


def get_train_test_split(predictors, target):
    train_index, test_index = train_test_split(
        predictors['fdc_id_source'].unique(),
        train_size=TRAIN_SIZE,
        random_state=RANDOM_SEED,
    )
    real_train_index = predictors[
        (predictors['fdc_id_source'].isin(train_index)) &
        (~predictors['fdc_id_target'].isin(test_index))
        ].index

    real_test_index = predictors[
        (predictors['fdc_id_source'].isin(test_index)) &
        (~predictors['fdc_id_target'].isin(train_index))
        ].index

    x_train = predictors.loc[real_train_index].drop(id_columns, axis=1)
    y_train = target.loc[real_train_index]
    x_test = predictors.loc[real_test_index].drop(id_columns, axis=1)
    y_test = target.loc[real_test_index]
    return x_train, y_train, x_test, y_test


if __name__ == "__main__":
    general_path = get_general_path()
    processed_path = join_paths(general_path, PROCESSED_DATA_PATH)

    processed_predictors_path = join_paths(
        processed_path, PROCESSED_PREDICTORS
    )
    processed_target_path = join_paths(processed_path, PROCESSED_TARGET)

    x_train_path = join_paths(processed_path, X_TRAIN_PATH)
    y_train_path = join_paths(processed_path, Y_TRAIN_PATH)
    x_test_path = join_paths(processed_path, X_TEST_PATH)
    y_test_path = join_paths(processed_path, Y_TEST_PATH)

    predictors = read_data(path=processed_predictors_path)
    target = read_data(path=processed_target_path)

    x_train, y_train, x_test, y_test = get_train_test_split(predictors, target)
    save_as_parquet(what=x_train, where=x_train_path)
    save_as_parquet(what=y_train, where=y_train_path)
    save_as_parquet(what=x_test, where=x_test_path)
    save_as_parquet(what=y_test, where=y_test_path)
