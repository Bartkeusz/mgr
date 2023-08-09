from itertools import product
import json

def generate_resnet50_combinations():
    input_dict = {
        "resize": ["warp", "center_crop", "random_crop"],
        "rotate": [True, False],
        "flip": [True, False],
        "fldsdi23p": [True, False],
        "fldsdip": [True, False],
        }
    
    keys = list(input_dict.keys())
    values = [input_dict[key] for key in keys]

    all_combinations = list(product(*values))
    combinations_with_keys = [dict(zip(keys, combination)) for combination in all_combinations]

    output_file = 'configs/resnet50_combinations.json'
    with open(output_file, 'w+') as f:
        json.dump(combinations_with_keys, f, indent=4)

    return combinations_with_keys

combinations = generate_resnet50_combinations()
for combination in combinations:
    print(combination)


