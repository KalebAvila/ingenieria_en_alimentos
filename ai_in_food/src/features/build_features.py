import pandas as pd

from feature_utils import (
    stem_list,
    process_base_ingredients,
    get_dim_reduction_from_simplification,
    get_entity_simplification_for_feature,
    REGEX_PATTERN,
    RANDOM_SEED,
)

from feature_information import categs_dict, flavor_df_columns_defined

from src.data.utils import (
    get_general_path,
    join_paths,
    read_data,
    read_pickle_with_pandas,
    make_desired_folder,
    check_if_filepath_exists,
    save_as_csv,
    save_as_parquet,
    save_as_pickle,
    concat_dataframes_from_list,
)


EDAMAM_FLAVOR_DB = "data/raw/edamam_flavor_dbs_nutrients.pkl"
USDA_FOOD_DB = "data/raw/branded_food.csv"
USDA_SEARCHABLE_DATA = "all_branded_foods_usda_searchable.csv"
INTERIM_DATA_PATH = "data/interim"
AVAILABLE_INGREDIENTS_PATH = "available_ingredients.pkl"
USDA_INGREDIENTS_LIST_PATH = "usda_ingredients.pkl"
SAMPLES_PATH = "samples.pkl"
SAMPLE_FEATURES_PATH = "sample_features.parquet"
SAMPLE_TARGET_PATH = "sample_target.parquet"

ENTITY_NUTRITION_FACTS_PATH = "enf.pkl"
ENTITY_FLAVOR_PROFILE_PATH = "efp.pkl"
ENTITY_FUNCTIONAL_GROUP_PATH = "efg.pkl"
ENTITY_FLAVOR_MOLECULES_PATH = "efm.pkl"

# entity_nutrition facts (enf)
# entity_flavor_profile (efp)
# entity_functional_group (efg)
# entity_flavor_molecules_name (efm)


def get_usda_foods():
    general_path = get_general_path()
    usda_db_path = join_paths(general_path, USDA_FOOD_DB)
    usda_foods = read_data(usda_db_path)
    usda_foods = usda_foods[usda_foods.ingredients.notna()]
    return usda_foods


def get_edamam_flavor_db():
    general_path = get_general_path()
    edamam_flavor_db_path = join_paths(general_path, EDAMAM_FLAVOR_DB)
    edamam_flavor = read_pickle_with_pandas(edamam_flavor_db_path)
    return edamam_flavor


def get_base_ingredients():
    edamam_flavor = get_edamam_flavor_db()
    base_ingredients = edamam_flavor.entity_alias_readable.to_list()
    return base_ingredients


def process_usda_ingredients(usda_data):
    print('Executing process_usda_ingredients function.')
    general_path = get_general_path()
    interim_path = join_paths(general_path, INTERIM_DATA_PATH)
    if not check_if_filepath_exists(interim_path):
        make_desired_folder(interim_path)
    usda_searchable_file = join_paths(interim_path, USDA_SEARCHABLE_DATA)
    if check_if_filepath_exists(usda_searchable_file):
        print(f'No need to process... File exists at: {usda_searchable_file}')
        return read_data(usda_searchable_file)
    new_usda_data = usda_data.copy()
    new_usda_data['ingredient_searchable'] = (
       ' ' +
       new_usda_data.ingredients
       .str.lower()
       .replace(REGEX_PATTERN, '', regex=True)
       .str.split(' ')
       .apply(stem_list)
       .str.join(' ') +
       ' '
    )
    new_usda_data = new_usda_data[
        ["ingredients",
         "ingredient_searchable",
         "branded_food_category",
         ]
    ]
    save_as_csv(what=new_usda_data, where=usda_searchable_file)
    return new_usda_data


def get_list_of_usda_ingredients(base_ingredients, usda_foods):
    print('Executing get_list_of_usda_ingredients function.')
    general_path = get_general_path()
    usda_ingredients_path = join_paths(
        general_path, INTERIM_DATA_PATH, USDA_INGREDIENTS_LIST_PATH
    )
    if check_if_filepath_exists(usda_ingredients_path):
        print(f'No need to process... File exists at: {usda_ingredients_path}')
        return read_pickle_with_pandas(usda_ingredients_path)
    ingredients = base_ingredients.ingredients_for_search.to_dict()
    usda_ingredients_results = []
    for real_ingredient, search_ingredient in ingredients.items():
        usda_ingredient_results = usda_foods[
            usda_foods.ingredient_searchable.str.contains(search_ingredient)]
        usda_ingredient_results['ing'] = real_ingredient
        usda_ingredients_results.append(usda_ingredient_results)

    usda_ingredients_repeated = concat_dataframes_from_list(
        usda_ingredients_results
    )
    usda_ingredients = usda_ingredients_repeated.groupby('fdc_id').agg(
        ingredients=('ingredients', 'first'),
        flavor_ingredients=('ing', lambda x: list(x)),
        category=('branded_food_category', 'first'),
        flavor_ingredients_count=('ing', 'count'),
    )
    usda_ingredients['filtered_category'] = usda_ingredients.category.map(
        categs_dict
    )
    save_as_pickle(what=usda_ingredients, where=usda_ingredients_path)
    return usda_ingredients


def get_sample_for_categories(data_set, size=200):
    general_path = get_general_path()
    samples_path = join_paths(
        general_path, INTERIM_DATA_PATH, SAMPLES_PATH
    )
    if check_if_filepath_exists(samples_path):
        return read_pickle_with_pandas(samples_path)

    information_samples = []
    unique_categories = list(set(categs_dict.values()))
    for categ in unique_categories:
        subcateg_df = data_set[data_set.filtered_category == categ]
        subcategs = subcateg_df.category.unique()
        for subcateg in subcategs:
            print(f"Getting sample for category {categ} "
                  f"in the subcategory {subcateg}")
            subcateg_condition = data_set.category == subcateg
            ingredients_condition = data_set.flavor_ingredients_count != 1
            try:
                sample_df = data_set[
                    ((subcateg_condition) & (ingredients_condition))
                ].sample(size, replace=True, random_state=RANDOM_SEED)
            except:
                sample_df = data_set[
                    subcateg_condition
                ].sample(size, replace=True, random_state=RANDOM_SEED)
            final_sample = sample_df\
                .reset_index()\
                .drop_duplicates('fdc_id')\
                .set_index('fdc_id')
            information_samples.append(final_sample)
    all_samples_df = concat_dataframes_from_list(information_samples)
    all_samples_df['str_flavor_ingredients'] = all_samples_df.\
        flavor_ingredients.apply(sorted).astype('str')

    save_as_pickle(what=all_samples_df, where=samples_path)
    return all_samples_df


def get_entity_traits(edamam_flavor):
    flavor_molecules_df_list = []
    nutrition_facts_df_list = []
    for row_index in range(edamam_flavor.shape[0]):
        df_tmp_idx = edamam_flavor.iloc[row_index]
        flvr_mol_df = pd.DataFrame(df_tmp_idx.molecules)
        flvr_mol_df['entity_alias_readable'] = df_tmp_idx.entity_alias_readable
        flavor_molecules_df_list.append(flvr_mol_df)
        nutrition_facts_df = pd.DataFrame(
            df_tmp_idx.nutritional_info).iloc[-1:]
        nutrition_facts_df[
            'entity_alias_readable'] = df_tmp_idx.entity_alias_readable
        nutrition_facts_df_list.append(nutrition_facts_df)

    nutrition_facts = concat_dataframes_from_list(
        nutrition_facts_df_list
    ).set_index("entity_alias_readable").fillna(0)

    flavor_molecules = concat_dataframes_from_list(flavor_molecules_df_list)[
        flavor_df_columns_defined + ['entity_alias_readable']]

    flavor_molecules[
        'flavor_profile_list'] = flavor_molecules\
        .fooddb_flavor_profile.str.split('@')
    flavor_molecules[
        'functional_group_list'] = flavor_molecules\
        .functional_groups.str.split('@')

    entity_nutrition_facts = nutrition_facts.copy()

    entity_flavor_profile = pd.DataFrame(
        flavor_molecules.explode('flavor_profile_list').groupby(
            'entity_alias_readable').flavor_profile_list.apply(set).apply(
            list))
    entity_functional_group = pd.DataFrame(
        flavor_molecules.explode('functional_group_list').groupby(
            'entity_alias_readable').functional_group_list.apply(set).apply(
            list))
    entity_flavor_molecules_name = pd.DataFrame(
        flavor_molecules.groupby('entity_alias_readable').common_name.apply(
            set).apply(list))
    return (
        entity_nutrition_facts,
        entity_flavor_profile,
        entity_functional_group,
        entity_flavor_molecules_name,
    )


def compute_flavor_entity_reduction(
    entity_flavor_profile,
    entity_functional_group,
    entity_flavor_molecules_name
):
    efp_reduced, efp_transformer = get_dim_reduction_from_simplification(
        get_entity_simplification_for_feature(
            entity_flavor_profile,
            "flavor_profile_list",
            fill_value=0
        ),
        feature_name='flavor_profile',
        components=50
    )

    efg_reduced, efg_transformer = get_dim_reduction_from_simplification(
        get_entity_simplification_for_feature(
            entity_functional_group,
            "functional_group_list",
            fill_value=0
        ),
        feature_name='functional_group',
        components=20
    )

    efm_reduced, efm_transformer = get_dim_reduction_from_simplification(
        get_entity_simplification_for_feature(
            entity_flavor_molecules_name,
            "common_name",
            fill_value=0
        ),
        feature_name='common_name',
        components=100
    )
    reductions = efp_reduced, efg_reduced, efm_reduced
    transformers = efp_transformer, efg_transformer, efm_transformer
    return reductions, transformers


def get_features_vector(ingredient_list):
    general_path = get_general_path()
    interim_path = join_paths(general_path, INTERIM_DATA_PATH)
    enf_path = join_paths(interim_path, ENTITY_NUTRITION_FACTS_PATH)
    efp_path = join_paths(interim_path, ENTITY_FLAVOR_PROFILE_PATH)
    efg_path = join_paths(interim_path, ENTITY_FUNCTIONAL_GROUP_PATH)
    efm_path = join_paths(interim_path, ENTITY_FLAVOR_MOLECULES_PATH)

    efm = read_pickle_with_pandas(efm_path)
    efp = read_pickle_with_pandas(efp_path)
    efg = read_pickle_with_pandas(efg_path)
    enf = read_pickle_with_pandas(enf_path)
    return pd.concat([
        efm.loc[ingredient_list].mean(),
        efp.loc[ingredient_list].mean(),
        efg.loc[ingredient_list].mean(),
        enf.loc[ingredient_list].mean()
    ])


def feature_generation():
    general_path = get_general_path()
    interim_path = join_paths(general_path, INTERIM_DATA_PATH)
    base_ingredients_path = join_paths(
        interim_path, AVAILABLE_INGREDIENTS_PATH
    )
    enf_path = join_paths(interim_path, ENTITY_NUTRITION_FACTS_PATH)
    efp_path = join_paths(interim_path, ENTITY_FLAVOR_PROFILE_PATH)
    efg_path = join_paths(interim_path, ENTITY_FUNCTIONAL_GROUP_PATH)
    efm_path = join_paths(interim_path, ENTITY_FLAVOR_MOLECULES_PATH)
    features_path = join_paths(interim_path, SAMPLE_FEATURES_PATH)
    target_path = join_paths(interim_path, SAMPLE_TARGET_PATH)

    usda_foods = get_usda_foods()
    searchable_usda_foods = process_usda_ingredients(usda_foods)
    edamam_flavor_db = get_edamam_flavor_db()
    base_ingredients = process_base_ingredients(get_base_ingredients())
    save_as_pickle(what=base_ingredients, where=base_ingredients_path)

    usda_ingredients = get_list_of_usda_ingredients(
        base_ingredients=base_ingredients,
        usda_foods=searchable_usda_foods,
    )
    all_samples = get_sample_for_categories(
        data_set=usda_ingredients, size=150
    )
    # entity_nutrition_facts (enf)
    # entity_flavor_profile (efp)
    # entity_functional_group (efg)
    # entity_flavor_molecules_name (efm)
    enf, efp, efg, efm = get_entity_traits(edamam_flavor_db)
    reductions, transformers = compute_flavor_entity_reduction(
        entity_flavor_profile=efp,
        entity_functional_group=efg,
        entity_flavor_molecules_name=efm,
    )
    efp_r, efg_r, efm_r = reductions

    print("Saving entities for ingredients...")

    print(f"Saving entity_nutrition_facts at {enf_path}")
    save_as_pickle(what=enf, where=enf_path)

    print(f"Saving entity_flavor_profile at {efp_path}")
    save_as_pickle(what=efp_r, where=efp_path)

    print(f"Saving entity_functional_group at {efg_path}")
    save_as_pickle(what=efg_r, where=efg_path)

    print(f"Saving entity_flavor_molecules_name at {efm_path}")
    save_as_pickle(what=efm_r, where=efm_path)
    features = all_samples.flavor_ingredients.apply(get_features_vector)

    print(f'Saving features, at {features_path}')
    save_as_parquet(what=features, where=features_path)
    target = all_samples[['filtered_category']]

    print(f'Saving target, at {target_path}')
    save_as_parquet(what=target, where=target_path)


if __name__ == "__main__":
    feature_generation()
