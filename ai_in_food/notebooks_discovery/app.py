import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import seaborn as sns


entity_flavor_molecules_name_reduced = pd.read_pickle('../data_discovery/processed/entity_flavor_molecules_name_reduced.pkl')
entity_flavor_profile_reduced = pd.read_pickle('../data_discovery/processed/entity_flavor_profile_reduced.pkl')
entity_functional_group_reduced = pd.read_pickle('../data_discovery/processed/entity_functional_group_reduced.pkl')
entity_nutrition_facts = pd.read_pickle('../data_discovery/processed/entity_nutrition_facts.pkl')


base_ingredients = pd.Series(pd.read_pickle('../data_discovery/processed/available_ingredients.pkl').index)
model = pd.read_pickle('../models/first_model.pkl')

explainer = shap.TreeExplainer(model)

def get_features_vector(ingredient_list):
    return pd.concat([
        entity_flavor_molecules_name_reduced.loc[ingredient_list].mean(),
        entity_flavor_profile_reduced.loc[ingredient_list].mean(),
        entity_functional_group_reduced.loc[ingredient_list].mean(),
        entity_nutrition_facts.loc[ingredient_list].mean()
    ])


def make_features_model_consumable_forSeries(features1, features2):
    features1['dummy_index'] = 'index'
    features2['dummy_index'] = 'index'
    product1_df = pd.DataFrame(features1).T
    product2_df = pd.DataFrame(features2).T
    model_consumable = product1_df.merge(product2_df, on='dummy_index').drop('dummy_index', axis=1)
    return model_consumable

def make_features_model_consumable_forDf(features1, features2):
    features1['dummy_index'] = 'index'
    features2['dummy_index'] = 'index'
    product1_df = pd.DataFrame(features1).T
    product2_df = features2
    model_consumable = product1_df.merge(product2_df, on='dummy_index').drop('dummy_index', axis=1)
    return model_consumable

def get_score(x_vals):
    result = model.predict_proba(x_vals)[:,1]
    return result

def get_ingredient_list_from_length(length, undesired_ingredients=[]):
    ingredient_list = base_ingredients[~(base_ingredients.isin(undesired_ingredients))].sample(length).to_list()
    return ingredient_list


import gradio as gr

def product_comparison(product_a, product_b):
    product_a_features = get_features_vector(product_a)
    product_b_features = get_features_vector(product_b)
    x_vals = make_features_model_consumable_forSeries(product_a_features, product_b_features)
    return get_score(x_vals)[0]

ingredient_list = base_ingredients.to_list()


product_description = "Please, enter the ingredients as a list of values. You can select more than one ingredient."
general_description='Add the list of ingredients of "Product A" and "Product B" in the corrponding fields, then click "Submit". The model will compare them and obtain a coeficient that goes from 0 to 1, where 1 means that the products are indeed similar. And 0 means they are definitely not similar.  '

demo = gr.Interface(
    product_comparison,
    [
        gr.Dropdown(
            ingredient_list,
            value=['Asparagus', 'Salt', 'Water'],
            multiselect=True,
            label="Product A",
            info=product_description
        ),
        gr.Dropdown(
            ingredient_list,
            value=['Beans', 'Carrot', 'Corn', 'Green Beans', 'Peas'],
            multiselect=True,
            label="Product B",
            info=product_description
        ),
    ],
    "text",
    examples=[
        [['Asparagus', 'Salt', 'Water'],['Beans', 'Carrot', 'Corn', 'Green Beans', 'Peas']],
        [['Amaranth', 'Banana', 'Chia', 'Flour', 'Kale', 'Lemon', 'Milk', 'Oats', 'Strawberry', 'Water', 'Yogurt'], ['Beans', 'Coconut', 'Cream', 'Milk', 'Pectin', 'Sugar']],
        [['Crab', 'Salt'], ['Salt', 'Shrimp']],
        [['Anchovy', 'Olive', 'Salt'], ['Salmon', 'Salt', 'Sockeye salmon']],
        [['Salmon', 'Salt', 'Sockeye salmon'],['Anchovy', 'Olive', 'Salt']],
        [['Barley', 'Lentils', 'Malt', 'Millet', 'Salt', 'Soybean', 'Spelt', 'Water', 'Wheat'], ['Barley', 'Buttermilk', 'Cream', 'Flour', 'Garlic', 'Malt', 'Onion', 'Salt', 'Soybean', 'Soybean Oil', 'Spread', 'Sugar', 'Water', 'Wheat']],

    ],
    title='Model Product Comparison',
    description = general_description,
)


if __name__ == "__main__":
    demo.launch(share=True)
    #demo.close()