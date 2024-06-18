from sklearn.metrics import roc_auc_score, average_precision_score

from src.data.utils import (
    get_general_path,
    join_paths,
    read_data,
)
from src.models.model_utils import (
    get_model,
    predict
)
from src.visualization.visualization_utils import (
    evaluate_metric,
    plot_histogram,
    plot_proportion_of_similarity_in_bins
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
    x_test_path = join_paths(processed_path, X_TEST_PATH)
    y_test_path = join_paths(processed_path, Y_TEST_PATH)

    x_train = read_data(x_train_path)
    y_train = read_data(y_train_path)
    x_test = read_data(x_test_path)
    y_test = read_data(y_test_path)

    model = get_model()
    pred_test = predict(model, x_test)
    pred_train = predict(model, x_train)

    print('Train:')
    evaluate_metric(
        y_true=y_train,
        y_pred=pred_train,
        metric=roc_auc_score,
        metric_name='ROC_AUC',
    )
    evaluate_metric(
        y_true=y_train,
        y_pred=pred_train,
        metric=average_precision_score,
        metric_name='Average Precision',
    )

    print('Test:')
    evaluate_metric(
        y_true=y_test,
        y_pred=pred_test,
        metric=roc_auc_score,
        metric_name='ROC_AUC',
    )
    evaluate_metric(
        y_true=y_test,
        y_pred=pred_test,
        metric=average_precision_score,
        metric_name='Average Precision',
    )
    # Plots
    plot_histogram(
        y_true=y_test,
        y_pred=pred_test,
        fig_name='test_histogram_score_distribution.png'
    )
    plot_histogram(
        y_true=y_train,
        y_pred=pred_train,
        fig_name='train_histogram_score_distribution.png'
    )

    plot_proportion_of_similarity_in_bins(
        y_true=y_test,
        y_pred=pred_test,
        fig_name='proportion_of_similarity.png'
    )
