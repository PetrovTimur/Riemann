import numpy as np
import pandas as pd
# from test2 import generate_log_uniform_sample

def generate_training_data(riemann_solver_newton, num_samples=1000,
                           h_min=0.01, h_max=10.0,
                           hu_min=-10.0, hu_max=10.0):

    # Initialize lists to store input and output samples
    inputs = []
    outputs = []

    for _ in range(num_samples):
        # Randomly sample input values within the specified domain
        hL = np.random.uniform(h_min, h_max)
        huL = np.random.uniform(hu_min, hu_max)
        hR = np.random.uniform(h_min, h_max)
        huR = np.random.uniform(hu_min, hu_max)
        # hL = generate_log_uniform_sample(h_min, h_max)
        # huL = (-1) ** (np.random.random() > 0.5) * generate_log_uniform_sample(0.01, hu_max)
        # hR = generate_log_uniform_sample(h_min, h_max)
        # huR = (-1) ** (np.random.random() > 0.5) * generate_log_uniform_sample(0.01, hu_max)

        # Compute the Riemann solution
        data = riemann_solver_newton(hL, huL, hR, huR, tol=1e-12)
        # print(data)
        h_star, u_star = data['star']
        k = data['data'].item()


        # Store the inputs and outputs
        inputs.append([hL, huL, hR, huR])
        outputs.append([h_star, u_star, k])

    # Convert to pandas DataFrame for easy handling
    df = pd.DataFrame(
        data=np.hstack([inputs, outputs]),
        columns=['hL', 'huL', 'hR', 'huR', 'h_star', 'u_star', 'iter']
    )

    return df


def generate_balanced_training_data(riemann_solver_newton, num_samples_per_category=100,
                                        h_min=0.1, h_max=10.0,
                                        hu_min=-10.0, hu_max=10.0, tol=1e-12):

    g = 9.8066
    # Mapping from condition pair to category
    mapping = {
        (0, 0): 0,
        (0, 1): 1,
        (0, 2): 2,
        (1, 0): 3,
        (1, 2): 4,
        (2, 0): 5,
        (2, 1): 6,
        (2, 2): 7
    }

    # Storage for samples, one list per category
    samples = {cat: [] for cat in range(8)}

    # Function to compute condition index
    def get_cond(u, c):
        if u < -c:
            return 0
        elif u > c:
            return 2
        else:
            return 1

    # Loop until all categories have at least num_samples_per_category
    while any(len(samples[cat]) < num_samples_per_category for cat in samples):
        # Randomly sample input values
        hL = np.random.uniform(h_min, h_max)
        huL = np.random.uniform(hu_min, hu_max)
        hR = np.random.uniform(h_min, h_max)
        huR = np.random.uniform(hu_min, hu_max)

        # Compute u and c values
        uL = huL / hL
        uR = huR / hR
        cL = np.sqrt(g * hL)
        cR = np.sqrt(g * hR)

        left_cond = get_cond(uL, cL)
        right_cond = get_cond(uR, cR)

        # Skip sample if both conditions are the middle case
        if left_cond == 1 and right_cond == 1:
            continue

        # Map conditions to a category, if mapping exists
        key = (left_cond, right_cond)
        if key not in mapping:
            continue
        cat = mapping[key]

        # If target for this category is reached, skip sample
        if len(samples[cat]) >= num_samples_per_category:
            continue

        # Compute the Riemann solution using the provided solver
        data = riemann_solver_newton(hL, huL, hR, huR, tol=tol)
        h_star, u_star = data['star']
        iter_val = data['data'].item()

        # Store the sample with corresponding inputs, outputs, and category
        sample = [hL, huL, hR, huR, h_star, u_star, iter_val, cat]
        samples[cat].append(sample)

    # Concatenate all samples from each category and create a DataFrame
    data_list = []
    for cat_list in samples.values():
        data_list.extend(cat_list)

    df = pd.DataFrame(data_list, columns=['hL', 'huL', 'hR', 'huR', 'h_star', 'u_star', 'iter', 'category'])
    return df