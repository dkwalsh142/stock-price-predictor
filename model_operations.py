import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

def logs(model, df):
    new_col = []
    iterable_model = model[:]
    for term in iterable_model:
        model.append(f"log({term})")
        if f"log({term}" not in df.columns:
            new_term = df[f'{term}'].apply(lambda x: np.log(abs(x)) * (-1 if x < 0 else 1) if x!= 0 else 0)
            new_term.name = f"log({term})"
            new_col.append(new_term)
    df = pd.concat([df] + new_col, axis=1)
    return model, df

def polynomials(model, df):
    new_col = []
    iterable_model = model[:]
    for term in iterable_model:
        model.append(f"({term})^2")
        if f"{term}^2" not in df.columns:
            new_term = df[f'{term}'].apply(lambda x: np.square(x))
            new_term.name = f"({term})^2"
            new_col.append(new_term)
    df = pd.concat([df] + new_col, axis=1)
    return model, df

def sqrts(model, df):
    new_col = []
    iterable_model = model[:]
    for term in iterable_model:
        model.append(f"sqrt({term})")
        if f"sqrt({term})" not in df.columns:
            new_term = df[term].apply(lambda x: np.sign(x) * np.sqrt(abs(x)))
            new_term.name = f"sqrt({term})"
            new_col.append(new_term)
    df = pd.concat([df] + new_col, axis=1)
    return model, df


def interactions(model, df):
    new_col = []
    iterable_model = model[:]
    for i in range (0,len(iterable_model)-1):
        model.append(f"{model[i]}*{model[i+1]}")
        if f"{model[i]}*{model[i+1]}" not in df.columns:
            new_term = df[model[i]]*df[model[i+1]]
            new_term.name = f"{model[i]}*{model[i+1]}"
            new_col.append(new_term)
    df = pd.concat([df] + new_col, axis=1)
    return model, df

def random_interactions(model, df):
    new_col = []
    scramble_model = model[:]
    random.shuffle(scramble_model)
    for i in range (0,len(scramble_model)-1):
        model.append(f"{scramble_model[i]}*{scramble_model[i+1]}")
        if f"{scramble_model[i]}*{scramble_model[i+1]}" not in df.columns:
            new_term = df[scramble_model[i]]*df[scramble_model[i+1]]
            new_term.name = f"{scramble_model[i]}*{scramble_model[i+1]}"
            new_col.append(new_term)
    df = pd.concat([df] + new_col, axis=1)
    return model, df

def estimate(intercept, coefficients, model, model_name, df):
    guess = pd.Series(0,index = df.index, name = f"{model_name}_guess")
    for var, co in zip(model, coefficients):
        guess += df[var]*co
    guess += intercept
    df = pd.concat([df, guess], axis = 1)
    return df

def mse(guess, target):
    residual = guess - target
    residual_squared = residual**2
    msr = residual_squared.sum() / len(target)
    return msr

def polarity(guess, target):
    guess = np.array(guess)
    target = np.array(target)
    correct_polarity = np.sign(guess) == np.sign(target)
    return (correct_polarity.sum() / len(target)) * 100

def get_best(models, y_data, df):
    model = linear_model.LinearRegression()

    top_mse = {}
    top_polarity = {}

    for outer_key, value in models.items():
        model.fit(value['data'],y_data)
        
        df = estimate(model.intercept_, model.coef_, value['variables'], outer_key, df)
        cos = []
        for coefficient in model.coef_:
            cos.append(coefficient)

        print(f'Model {outer_key}')
        print("Variables:", value['variables'])
        print('Coefficients:', model.coef_)
        print('Intercept:', model.intercept_)

        MSE = mse(df['target'], df[f'{outer_key}_guess'])
        POLARITY =  polarity(df['target'], df[f'{outer_key}_guess'] )

        print('MSE', MSE)
        print('Correct polarity', POLARITY)
        print('\n')

        mse_keys = list(top_mse.keys())
        mse_values = list(top_mse.values())
        polarity_keys = list(top_polarity.keys())
        polarity_values = list(top_polarity.values())

        if len(top_mse) < 2:
            top_mse[outer_key] = MSE
        else:
            combined = list(top_mse.items()) + [(outer_key, MSE)]
            combined.sort(key=lambda x: x[1])  # ascending MSE is better
            combined = combined[:2]
            top_mse = dict(combined)

        # Add to top_polarity if less than 2 models are stored
        if len(top_polarity) < 2:
            top_polarity[outer_key] = POLARITY
        else:
            # Combine current top models with the new candidate
            combined = list(top_polarity.items()) + [(outer_key, POLARITY)]
            # Sort by polarity score descending
            combined.sort(key=lambda x: x[1], reverse=True)
            # Take top 2
            combined = combined[:2]
            # Rebuild the dictionary
            top_polarity = dict(combined)

    return top_mse, top_polarity