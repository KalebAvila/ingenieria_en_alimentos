import pandas as pd
import numpy as np

from nltk.stem import PorterStemmer
from sklearn.decomposition import PCA

REGEX_PATTERN = r'[^\w\s]'
RANDOM_SEED = 49


def stem_list(word_list):
    stem_result = [PorterStemmer().stem(word) for word in word_list]
    return stem_result


def clean_usda_foods(usda_db):
    usda_db = usda_db[usda_db.ingredients.notna()]
    return usda_db


def process_base_ingredients(base_ingredients_list):
    base_ingredients = pd.DataFrame()
    base_ingredients['ingredients'] = pd.Series(base_ingredients_list)
    base_ingredients['ingredients_cleaned_list'] = base_ingredients\
        .ingredients.str.lower()\
        .replace(REGEX_PATTERN, '', regex=True)\
        .str.split(' ')
    base_ingredients['stemmed_ingredients'] = base_ingredients\
        .ingredients_cleaned_list.apply(stem_list)
    base_ingredients['ingredients_cleaned'] = base_ingredients\
        .stemmed_ingredients.str.join(' ')
    base_ingredients['ingredients_for_search'] = (
            ' ' +
            base_ingredients.ingredients_cleaned +
            ' '
    )
    base_ingredients.set_index('ingredients', inplace=True)
    return base_ingredients


def get_entity_simplification_for_feature(
    entity_feature_df, feature, fill_value=np.nan
):
    relation_df = entity_feature_df.explode(feature)
    relation_df['dummy'] = 1
    entity_feature = pd.pivot_table(
        relation_df,
        index='entity_alias_readable',
        columns=feature,
        aggfunc='count',
        values="dummy",
        fill_value=fill_value
    )
    return entity_feature


def get_dim_reduction_from_simplification(
    simplified_df, feature_name, components=100
):
    pca_obj = PCA(
        n_components=components,
        copy=True,
        whiten=False,
        svd_solver='auto',
        tol=0.0,
        iterated_power='auto',
        n_oversamples=10,
        power_iteration_normalizer='auto',
        random_state=RANDOM_SEED
    )
    columns = [f'{feature_name}__{col}' for col in range(components)]
    dim_reduction = pd.DataFrame(
        pca_obj.fit_transform(simplified_df),
        index=simplified_df.index,
        columns=columns
    )
    return dim_reduction, pca_obj
