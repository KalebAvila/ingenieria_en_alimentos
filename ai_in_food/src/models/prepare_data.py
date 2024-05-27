from src.features.feature_utils import RANDOM_SEED

from model_utils import (
    non_predictor_columns,
)

from src.data.utils import (
    get_general_path,
    join_paths,
    make_desired_folder,
    check_if_filepath_exists,
    save_as_parquet,
    read_data
)

SAMPLES_PATH = "samples.pkl"
SAMPLE_FEATURES_PATH = "sample_features.parquet"
SAMPLE_TARGET_PATH = "sample_target.parquet"
INTERIM_DATA_PATH = "data/interim"
PROCESSED_DATA_PATH = "data/processed"

PROCESSED_PREDICTORS = "processed_predictors.parquet"
PROCESSED_TARGET = "processed_target.parquet"


# target
# features

def transform_multiclass_target_into_binary_by_pairs(target):
    multiclass_target = target.reset_index()
    multiclass_target['fdc_id2'] = (
            [target.reset_index().fdc_id.to_list()] *
            target.reset_index().shape[0]
    )
    binary_target = multiclass_target.drop("fdc_id2", axis=1).merge(
        multiclass_target.explode("fdc_id2"),
        left_on='fdc_id',
        right_on='fdc_id2',
        how='inner'
    )
    # Category comparison
    cc = binary_target.filtered_category_x == binary_target.filtered_category_y
    binary_target['similar'] = (cc).astype('int')
    binary_target = binary_target.rename(
        columns={'fdc_id_x': 'fdc_id_source', 'fdc_id_y': 'fdc_id_target'})[
        ['fdc_id_source', 'fdc_id_target', 'similar']]
    return binary_target


def get_target_sample(target, frac=0.1):
    target_sample = target.sample(frac=frac, random_state=RANDOM_SEED)
    return target_sample


def create_dataset(target, features):
    dataset = target\
        .merge(
            features.reset_index(),
            left_on="fdc_id_source",
            right_on='fdc_id',
            how='inner'
        )\
        .drop('fdc_id', axis=1)\
        .merge(
            features.reset_index(),
            left_on='fdc_id_target',
            right_on='fdc_id',
            how='inner'
        )\
        .drop('fdc_id', axis=1)
    return dataset


if __name__ == "__main__":
    general_path = get_general_path()
    interim_path = join_paths(general_path, INTERIM_DATA_PATH)
    processed_path = join_paths(general_path, PROCESSED_DATA_PATH)

    target_path = join_paths(interim_path, SAMPLE_TARGET_PATH)
    features_path = join_paths(interim_path, SAMPLE_FEATURES_PATH)

    processed_predictors_path = join_paths(
        processed_path, PROCESSED_PREDICTORS
    )
    processed_target_path = join_paths(processed_path, PROCESSED_TARGET)

    if not check_if_filepath_exists(processed_path):
        make_desired_folder(processed_path)

    target = read_data(target_path)
    features = read_data(features_path)

    binary_target = transform_multiclass_target_into_binary_by_pairs(target)
    binary_target_sample = get_target_sample(binary_target)

    dataset = create_dataset(binary_target_sample, features)

    processed_predictors = dataset.drop(non_predictor_columns, axis=1)
    processed_target = dataset[['similar']]

    save_as_parquet(what=processed_predictors, where=processed_predictors_path)
    save_as_parquet(what=processed_target, where=processed_target_path)
