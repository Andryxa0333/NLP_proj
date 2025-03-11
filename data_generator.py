import numpy as np
import pandas as pd
import config
import json

with open('config.json', 'r') as file:
    data = json.load(file)
config = config.Config(**data)

np.random.seed(config.data.random_seed)

def generate_complex_data():
    data = []

    for i in range(0, 10, 5):  # i = 0 to 9
        for j in range(1, 10):  # j = 1 to 9
            i_random = i + np.random.randint(0, 5)
            j_random = j + np.random.randint(0, 5)

            expr = f"{i_random} + {j_random}"
            result = i_random + j_random
            if result >= 0:
                data.append({'expression': expr, 'result': result})

            expr = f"{i_random} - {j_random}"
            result = i_random - j_random
            if result >= 0:
                data.append({'expression': expr, 'result': result})

            expr = f"{i_random} * {j_random}"
            result = i_random * j_random
            data.append({'expression': expr, 'result': result})

            if j_random != 0 and i_random % j_random == 0:
                expr = f"{i_random} / {j_random}"
                result = i_random // j_random
                data.append({'expression': expr, 'result': result})

            expr = f"{i_random}^2"
            result = i_random ** 2
            data.append({'expression': expr, 'result': result})

    for i in range(10, 100, 5):
        for j in range(10, 100):
            i_random = i + np.random.randint(0, 5)
            j_random = j + np.random.randint(0, 5)

            expr = f"{i_random} + {j_random}"
            result = i_random + j_random
            if result >= 0:
                data.append({'expression': expr, 'result': result})

            expr = f"{i_random} - {j_random}"
            result = i_random - j_random
            if result >= 0:
                data.append({'expression': expr, 'result': result})

            expr = f"{i_random} * {j_random}"
            result = i_random * j_random
            data.append({'expression': expr, 'result': result})

            if j_random != 0 and i_random % j_random == 0:
                expr = f"{i_random} / {j_random}"
                result = i_random // j_random
                data.append({'expression': expr, 'result': result})

            expr = f"{i_random}^2"
            result = i_random ** 2
            data.append({'expression': expr, 'result': result})

    for _ in range(1000):
        numbers = np.random.randint(1, 100, 3)
        expr = f"({numbers[0]} + {numbers[1]}) * {numbers[2]}"
        result = (numbers[0] + numbers[1]) * numbers[2]
        if result >= 0:
            data.append({'expression': expr, 'result': result})

    for _ in range(1000):
        numbers = np.random.randint(1, 100, 4)
        expr = f"({numbers[0]} + {numbers[1]}) * ({numbers[2]} - {numbers[3]})"
        result = (numbers[0] + numbers[1]) * (numbers[2] - numbers[3])
        if result >= 0:
            data.append({'expression': expr, 'result': result})

    return pd.DataFrame(data)

complex_data = generate_complex_data()

shuffled_data = complex_data.sample(frac=1).reset_index(drop=True)

def generate_test_data():
    test_data = []
    for _ in range(100):
        a = np.random.randint(1, 1000)
        b = np.random.randint(1, 1000)
        c = np.random.randint(1, 1000)
        d = np.random.randint(1, 1000)
        expr = f"({a} * {b}) / ({c} + {d})"
        result = (a * b) / (c + d)
        test_data.append({'expression': expr, 'result': str(result)})
    return pd.DataFrame(test_data)

test_data = generate_test_data()

complex_data.to_csv('сurriculum_data.csv', index=False)
shuffled_data.to_csv('shuffled_data.csv', index=False)
test_data.to_csv('test_data.csv', index=False)
