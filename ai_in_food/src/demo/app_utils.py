import gradio as gr
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import shap

from src.features.build_features import (
    get_features_vector,
    get_base_ingredients
)
from src.models.model_utils import (
    get_model,
    predict
)

DATASET_COLORS = {
    'nutrients': 'indianred',
    'flavor_profile': 'steelblue',
    'functional_group': 'cadetblue',
    'common_name': 'mediumslateblue'
}


def matplotlib_color_config():
    COLOR = '#FBFBF9'
    mpl.rcParams['text.color'] = COLOR
    mpl.rcParams['axes.labelcolor'] = COLOR
    mpl.rcParams['xtick.color'] = COLOR
    mpl.rcParams['ytick.color'] = COLOR
    # plt.rcParams["font.family"] = "serif"


def make_features_model_consumable(features1, features2, handle='Series'):
    features1['dummy_index'] = 'index'
    features2['dummy_index'] = 'index'
    product1_df = pd.DataFrame(features1).T
    if handle == 'Series':
        product2_df = pd.DataFrame(features2).T
    else:
        product2_df = features2
    model_consumable = product1_df.merge(
        product2_df, on='dummy_index'
    ).drop('dummy_index', axis=1)
    return model_consumable


def get_explanations(model, x_vals, max_display=20):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_vals)[1]
    shap_values_df = pd.DataFrame(
        shap_values,
        columns=x_vals.columns
    ).T
    processed_shap_values = shap_values_df\
        .abs()\
        .sort_values(0, ascending=False)
    processed_shap_values['shap_vals'] = shap_values_df\
        .loc[processed_shap_values.index][0]
    top_features = pd.Series(processed_shap_values.head(max_display).index)
    relevant_f = top_features.str.split('__').apply(lambda x: x[0])
    condition__flavor_profile = relevant_f.str.startswith('flavor_profile')
    condition__common_name = relevant_f.str.startswith('common_name')
    condition__functional_group = relevant_f.str.startswith('functional_group')
    condition__nutrients = ~(
            condition__flavor_profile |
            condition__common_name |
            condition__functional_group
    )
    relevant_features_df = pd.DataFrame(relevant_f)
    relevant_features_df.loc[
        condition__flavor_profile, 'tag'] = 'flavor_profile'
    relevant_features_df.loc[condition__common_name, 'tag'] = 'common_name'
    relevant_features_df.loc[
        condition__functional_group, 'tag'] = 'functional_group'
    relevant_features_df.loc[condition__nutrients, 'tag'] = 'nutrients'
    relevant_datasets = relevant_features_df.tag.value_counts(normalize=True)

    processed_shap_values = processed_shap_values.head(max_display)
    pos_shap = processed_shap_values.shap_vals >= 0
    neg_shap = processed_shap_values.shap_vals < 0
    processed_shap_values.loc[pos_shap, 'color'] = 'deeppink'
    processed_shap_values.loc[neg_shap, 'color'] = 'dodgerblue'

    fig = plt.figure(facecolor='#1f2937')
    plt.suptitle('Product Explanations')
    plt.subplot(121)
    plt.title('Shap importance')
    sns.barplot(
        y=processed_shap_values.index,
        x=processed_shap_values.shap_vals,
        palette=processed_shap_values.color,
        orient='h'
    )
    plt.box(False)
    plt.xlabel('Shap Values', )
    plt.ylabel('Most relevant features')
    plt.subplot(122)
    plt.title('General relevance')
    pie_results = relevant_datasets.to_dict()
    colors = [DATASET_COLORS.get(key) for key in pie_results.keys()]
    plt.pie(
        x=pie_results.values(),
        labels=pie_results.keys(),
        startangle=90,
        autopct='%1.1f%%',
        colors=colors
    )
    plt.tight_layout()
    return fig


def product_comparison(product_a, product_b, need_results=False):
    product_a_features = get_features_vector(product_a)
    product_b_features = get_features_vector(product_b)
    x_vals = make_features_model_consumable(
        product_a_features,
        product_b_features,
        handle='Series'
    )
    model = get_model()
    prediction = predict(model, x_vals)[0]
    if need_results:
        return prediction, (model, x_vals)
    return prediction


def product_comparison_explain(product_a, product_b):
    prediction, results = product_comparison(
        product_a, product_b, need_results=True
    )
    model, x_vals = results
    fig = get_explanations(model, x_vals)
    return prediction, fig


def gradio_init(explanation=False):
    matplotlib_color_config()
    ingredient_list = get_base_ingredients()
    product_description = (
        "Please, enter the ingredients as a list of values. You can select "
        "more than one ingredient."
    )
    general_description = (
        'Add the list of ingredients of "Product X" and "Product Y" in the '
        'corresponding fields, then click "Submit". The model will compare'
        ' them and obtain a coeficient that goes from 0 to 1, where 1 means '
        'that the products are indeed similar. And 0 means they are '
        'definitely not similar.'
    )
    inputs = [
        gr.Dropdown(
            ingredient_list,
            value=['Asparagus', 'Salt', 'Water'],
            multiselect=True,
            label="Product X",
            info=product_description
        ),
        gr.Dropdown(
            ingredient_list,
            value=['Beans', 'Carrot', 'Corn', 'Green Beans', 'Peas'],
            multiselect=True,
            label="Product Y",
            info=product_description
        ),
    ]
    if explanation:
        fn = product_comparison_explain
        outputs = [gr.Text(label='Model Score'), gr.Plot(label="Explanation")]
    else:
        fn = product_comparison
        print('the output should be just text')
        outputs = gr.Text(label='Model Score')

    demo = gr.Interface(
        fn=fn,
        inputs=inputs,
        outputs=outputs,
        examples=[
            [['Asparagus', 'Salt', 'Water'],
             ['Beans', 'Carrot', 'Corn', 'Green Beans', 'Peas']],
            [['Amaranth', 'Banana', 'Chia', 'Flour', 'Kale', 'Lemon', 'Milk',
              'Oats', 'Strawberry', 'Water', 'Yogurt'],
             ['Beans', 'Coconut', 'Cream', 'Milk', 'Pectin', 'Sugar']],
            [['Crab', 'Salt'], ['Salt', 'Shrimp']],
            [['Anchovy', 'Olive', 'Salt'],
             ['Salmon', 'Salt', 'Sockeye salmon']],
            [['Salmon', 'Salt', 'Sockeye salmon'],
             ['Anchovy', 'Olive', 'Salt']],
            [['Barley', 'Lentils', 'Malt', 'Millet', 'Salt', 'Soybean',
              'Spelt', 'Water', 'Wheat'],
             ['Barley', 'Buttermilk', 'Cream', 'Flour', 'Garlic', 'Malt',
              'Onion', 'Salt', 'Soybean', 'Soybean Oil', 'Spread', 'Sugar',
              'Water', 'Wheat']],
        ],
        title='Model Product Comparison',
        description=general_description,
    )
    return demo
