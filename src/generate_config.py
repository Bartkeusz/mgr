def generate_configs():
    input_dict = {
        "resize": ["warp", "center_crop", "random_crop"],
        "rotate": [True, False],
    }

    result = []

    def helper(keys, current_combination):
        if not keys:
            result.append(current_combination.copy())
            return

        key = keys[0]
        values = input_dict[key]

        for value in values:
            current_combination[key] = value
            helper(keys[1:], current_combination, result)
            del current_combination[key]

    helper(list(input_dict.keys()), {})
    return result