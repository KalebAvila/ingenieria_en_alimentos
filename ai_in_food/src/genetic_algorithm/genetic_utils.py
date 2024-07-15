import numpy as np
import pandas as pd

from src.demo.app_utils import make_features_model_consumable
from src.features.build_features import (
    get_entity_features,
    get_features_vector,
    get_features_vector_wo_download,
    get_base_ingredients,
)
from src.models.model_utils import get_model, predict

from src.data.utils import (
    get_general_path,
    join_paths,
    read_data,
    read_pickle_with_pandas,
    make_desired_folder,
    check_if_filepath_exists,
    save_as_csv,
)

NEW_PATH = 'data/New_Datasets'

MATING_PROPORTION = 0.9


def get_ingredient_list_from_length(
    length,
    undesired_ingredients=[]
):
    base_ingredients = get_base_ingredients()
    base_ingredients_series = pd.Series(base_ingredients)
    ingredient_list = base_ingredients_series[
        ~(base_ingredients_series.isin(undesired_ingredients))
    ].sample(length).to_list()
    return ingredient_list


def mutation2_procedure(children_list):
    if len(children_list) > 1:
        r_num = np.random.randint(1, len(children_list))
        new_children_list = list(children_list).copy()
        new_children_list.pop(r_num)
    else:
        new_children_list = children_list.copy()
    return new_children_list


def mutations3_procedure(children_list, additional_undesired_ingredients):
    children_list = list(children_list)
    new_children_list = children_list + get_ingredient_list_from_length(
        1, children_list + additional_undesired_ingredients
    )
    return new_children_list


def mutation4_procedure(children_list, additional_undesired_ingredients):
    children_list = list(children_list)
    new_ingredient = get_ingredient_list_from_length(
        1,
        children_list + additional_undesired_ingredients
    )
    if len(children_list) > 2:
        r_num = np.random.randint(1, len(children_list))
        new_children_list = list(children_list).copy()
        new_children_list.pop(r_num)
    else:
        new_children_list = children_list
    new_children_list = new_children_list + new_ingredient
    return new_children_list


def genetic_algorithm(
    product_to_develop=[],
    undesired_ingredients=[],
    initial_population_size=100,
    generations=600,
    verbose=1,
    show_gen_samples_nb=20,
    return_last_available_generation=False,
    population_sample=None
):
    efm, efp, efg, enf = get_entity_features()
    model = get_model()

    if population_sample is None:
        if verbose >= 2:
            print(
                f'Initializing population with an initial size: '
                f'{initial_population_size}'
            )
        population = pd.DataFrame()
        if verbose >= 2:
            print(
                'Getting random_number_ingredient_list.'
            )
        random_number_ingredient_list = np.random.randint(
            2, 100, initial_population_size
        )

        population['lenght'] = random_number_ingredient_list
        if verbose >= 2:
            print(
                'Getting an ingredient list based on the lenght.'
            )
        population['ingredient_list'] = population['lenght'].apply(
            lambda x: get_ingredient_list_from_length(
                length=x,
                undesired_ingredients=undesired_ingredients
            )
        )
    else:
        print(
            f'Initializing population from the population_sample dataframe,'
            f' an initial size of: {population_sample.shape[0]} is considered'
        )
        population = population_sample
    if verbose >= 2:
        print('Done with initialization... Showing the initial population:')
        print(population.head(5))

    if verbose >= 2:
        print(
            'Getting the features from the corresponding data matrices...'
        )
    proposal_features = population.ingredient_list.apply(
        get_features_vector_wo_download,
        efm=efm,
        efp=efp,
        efg=efg,
        enf=enf
    )
    if verbose >= 2:
        print('Showing the features...')
        print(proposal_features.head())
        print('Getting features for the product to develop...')
    product_to_develop_features = get_features_vector_wo_download(
        product_to_develop,
        efm=efm,
        efp=efp,
        efg=efg,
        enf=enf
    )
    if verbose >= 2:
        print(product_to_develop_features.head())
        print('Creating the values to use model...')
    x_vals = make_features_model_consumable(
        features1=product_to_develop_features,
        features2=proposal_features,
        handle='DataFrame'
    )
    if verbose >= 2:
        print('Showing the values to use model:')
        print(x_vals.head())
        print('Making initial prediction...')
    population['prediction'] = predict(model, x_vals)
    if verbose >= 2:
        print('Showing predictions:')
        print(population['prediction'].head())
        print('Done with predictions.')

    # Elitismo
    population_gen = population.sort_values('prediction', ascending=False)
    result_list = []
    for gen in range(generations):
        try:
            # Which get to live (fittest) but some unfit aswell
            if verbose >= 2:
                print('Evaluating fitness...')
            living_number = int(initial_population_size * 0.1)
            additional_living_number = int(living_number * .010)
            survivors_head_and_tail = [
                population_gen.head(living_number),
                population_gen.tail(additional_living_number)
            ]
            survivors = pd.concat(survivors_head_and_tail)

            # Which get to mate, (90% of the fittest)
            if verbose >= 2:
                print('Selecting mates...')
            mating_number = int(living_number * MATING_PROPORTION)
            survivors.loc[:, 'identifier'] = np.random.randint(
                0,
                living_number,
                survivors.shape[0]
            )
            maters = survivors[
                survivors.identifier < mating_number
            ].drop('identifier', axis=1)
            maters.loc[:, 'mating_loc'] = np.random.randint(
                0, maters.shape[0], maters.shape[0]
            )
            maters['mate_ingredient_list'] = maters.mating_loc.apply(
                lambda x: list(maters.iloc[x].ingredient_list)
            )
            maters['children_number'] = np.random.randint(
                1, 10, maters.shape[0]
            )
            maters['ingredient_list'] = maters.ingredient_list.apply(list)
            print(maters.head())

            # Recombination
            # Add both ingredient lists (as a set)
            if verbose >= 2:
                print('Reproduction phase...')
                print(
                    'Crossover is in progress: children are being created...')

            childrens_t1 = (
                    maters.ingredient_list + maters.mate_ingredient_list
            ).apply(lambda x: list(set(x)))

            # Keep first half and exchange second half
            childrens_t2 = maters.ingredient_list.apply(
                lambda x: x[:len(x) // 2]
            ) + maters.mate_ingredient_list.apply(
                lambda x: x[len(x) // 2:]
            )

            # Keep second half and exchange first half
            childrens_t3 = maters.ingredient_list.apply(
                lambda x: x[len(x) // 2:]
            ) + maters.mate_ingredient_list.apply(
                lambda x: x[:len(x) // 2]
            )

            # Add both ingredient lists (as a set)
            childrens_t4 = (
                maters.ingredient_list +
                maters.mate_ingredient_list).apply(
                    lambda x: np.random.choice(
                        list(set(x)),
                        np.random.randint(1, len(list(set(x))))
                    )
                )
            childrens = pd.concat(
                [childrens_t1, childrens_t2, childrens_t3, childrens_t4]
            ).reset_index(drop=True)
            print(childrens.head())

            # Mutations
            # No mutation
            if verbose >= 2:
                print('Including mutations into childrens...')

            mutations1 = childrens.copy()
            print(mutations1.head())

            # Randomly delete an element from the childrens
            mutations2 = childrens.apply(lambda x: mutation2_procedure(x))
            print(mutations2.head())

            # Randomly add an element from the childrens
            mutations3 = childrens.apply(
                mutations3_procedure,
                additional_undesired_ingredients=undesired_ingredients
            )
            print(mutations3.head())

            # Randomly delete and add an element (different of the deletion)
            mutations4 = childrens.apply(
                mutation4_procedure,
                additional_undesired_ingredients=undesired_ingredients
            )
            print(mutations4.head())

            children_with_mutations = pd.concat(
                [mutations1, mutations2, mutations3, mutations4]).reset_index(
                drop=True)
            children_with_mutations = children_with_mutations.apply(
                lambda x: list(set(x))
            )
            print(children_with_mutations.head())
            if verbose >= 2:
                print(
                    f'Total of {children_with_mutations.shape[0]} distinct'
                    f' children were born.'
                )
            new_population = pd.DataFrame()
            new_population['ingredient_list'] = children_with_mutations
            new_population['str_ingredient_list'] = new_population[
                'ingredient_list'].astype('str')
            new_population = new_population.drop_duplicates(
                'str_ingredient_list').drop('str_ingredient_list', axis=1)

            new_proposal_features = new_population.ingredient_list.apply(
                get_features_vector)

            new_x_vals = make_features_model_consumable(
                features1=product_to_develop_features,
                features2=new_proposal_features,
                handle='DataFrame'
            )
            new_population['prediction'] = predict(model, new_x_vals)

            results = new_population.sort_values(
                'prediction', ascending=False
            ).head(3)
            results['generation'] = gen + 1
            result_list.append(results)

            if verbose >= 2:
                print('Results were saved')
            population_gen = new_population.sort_values(
                'prediction',
                ascending=False
            )
            if verbose >= 1:
                print(f'Generation: {gen + 1}')
        except:
            print(
                'Something happened Process was manually (or by mistake)'
                ' stopped... Fetching result list.'
            )
            break
    final_results = pd.concat(result_list).sort_values(
        'prediction',
        ascending=False
    )
    if return_last_available_generation:
        return final_results, population_gen
    return final_results


def process_results(results):
    general_path = get_general_path()
    csv_path = join_paths(general_path, NEW_PATH)
    r = results.copy()
    r.sort_values('prediction', inplace=True)
    max_pred_value = r.prediction.max()
    max_results_candidates = r[r.prediction == max_pred_value]
    max_results_candidates.loc[:, 'entity_lenght'] = (
        max_results_candidates.ingredient_list.apply(len)
    )
    min_entity_lenght = (
            max_results_candidates.entity_lenght ==
            max_results_candidates.entity_lenght.min()
    )
    max_results_candidates[min_entity_lenght]
    min_generation = (
            max_results_candidates.generation ==
            max_results_candidates.generation.min()
    )
    max_results_candidates[min_generation]
    final_result = max_results_candidates.iloc[0]
    resume = {
        'ingredient_list': final_result.ingredient_list,
        'model_score': final_result.prediction
    }
    print(f'Saving target, at {csv_path}')
    final_result.to_csv('/Users/kalebavila/Documents/ingenieria_en_alimentos/ai_in_food/data/New_Datasets/Resultados_1.csv', index=False)
    #save_as_csv(what=final_result, where=csv_path)

    return resume
