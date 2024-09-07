from src.features.build_features import get_base_ingredients
from src.genetic_algorithm.genetic_utils import (
    genetic_algorithm,
    process_results
)


if __name__ == "__main__":
    ingredients = get_base_ingredients()
    not_ing = [ing for ing in ingredients if ('milk' in ing.lower())]
    results, gen = genetic_algorithm(
        product_to_develop=['Milk'],
        undesired_ingredients=not_ing,
        generations=600,
        verbose=3,
        show_gen_samples_nb=3,
        return_last_available_generation=True
    )
    processed_results = process_results(results)
    print(processed_results)
