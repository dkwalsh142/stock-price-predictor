import requests
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model_operations import logs, polynomials, sqrts, interactions, get_best, random_interactions
from sklearn import linear_model

iterations = 5

url = "https://api.polygon.io/v2/aggs/ticker/GOOGL/range/1/minute/2024-01-02/2024-04-01?apiKey=[INSETR OWN KEY]&limit=50000"

response = requests.get(url)

if response.status_code == 200:
    data = response.json()

    if "results" in data:
        results = data["results"]
        df = pd.DataFrame(results)
        df['date'] = pd.to_datetime(df['t'], unit='ms').dt.date
        #print(df.head())
        #print(df.tail())
        #print(len(df))

else:
    print(f"Failed to fetch data: {response.status_code}")  

#create target values of percent change 30 minutes from now
df['target'] = df['c'].pct_change(periods=-30)*100

#generate percent change variables
model_1 = []
new_collumns = []
for i in range(1, 384,20): #percent change for past two hours at 10 minute increments
    pct_change_collumn = df['c'].pct_change(periods=i)*100
    pct_change_collumn.name = f"pct_{i}min"
    new_collumns.append(pct_change_collumn)
    model_1.append(f"pct_{i}min")

df = pd.concat([df] + new_collumns, axis=1)
df = df.dropna() 

model_2,df = logs(model_1[:],df)
model_3,df = polynomials(model_1[:],df)
model_4,df = sqrts(model_1[:],df)
model_5,df = interactions(model_1[:],df)
model_6,df = random_interactions(model_1[:],df)

y_data = df['target']
x1_data = df[model_1]
x2_data = df[model_2]
x3_data = df[model_3]
x4_data = df[model_4]
x5_data = df[model_5]
x6_data = df[model_6]

models = {
    '1' : {'variables': model_1, 'data' : x1_data},
    '2' : {'variables': model_2, 'data' : x2_data},
    '3' : {'variables': model_3, 'data' : x3_data},
    '4' : {'variables': model_4, 'data' : x4_data},
    '5' : {'variables': model_5, 'data' : x5_data},
    '6' : {'variables': model_6, 'data' : x6_data}
    }

i = 0
while i < iterations:

    top_mse, top_polarity = get_best(models, y_data, df)
        
    print("Top MSE:", top_mse)
    print("Top polarity correctness:", top_polarity)


    best_models = {}
    for key in top_polarity:
        if key not in best_models:
            best_models[key] = models[key]
            
    print(best_models.keys())

    new_models = {}
    for key, inner_dict in best_models.items():
        placeholder = copy.deepcopy(best_models[key]["variables"])
        
        #Add original Model
        new_models[key] = {}
        new_models[key]["variables"] = placeholder[:]
        new_models[key]["data"] = df[placeholder[:]]
        #Create logs mutation
        new_models[f"{key}.2"] = {}
        df = df.loc[:, ~df.columns.duplicated()]
        new_models[f"{key}.2"]["variables"], df = logs(placeholder[:], df)
        new_models[f"{key}.2"]['data'] = df[new_models[f"{key}.2"]["variables"]]
        #Create polynomial mutation
        new_models[f"{key}.3"] = {}
        df = df.loc[:, ~df.columns.duplicated()]
        new_models[f"{key}.3"]["variables"], df = polynomials(placeholder[:], df)
        new_models[f"{key}.3"]['data'] = df[new_models[f"{key}.3"]["variables"]]
        #Create sqrts mutation
        new_models[f"{key}.4"] = {}
        df = df.loc[:, ~df.columns.duplicated()]
        new_models[f"{key}.4"]["variables"], df = sqrts(placeholder[:], df)
        new_models[f"{key}.4"]['data'] = df[new_models[f"{key}.4"]["variables"]]
        #Create side by side interaction mutation
        new_models[f"{key}.5"] = {}
        df = df.loc[:, ~df.columns.duplicated()]
        new_models[f"{key}.5"]["variables"], df = interactions(placeholder[:], df)
        new_models[f"{key}.5"]['data'] = df[new_models[f"{key}.5"]["variables"]]
        #Create random interaction mutation
        new_models[f"{key}.6"] = {}
        df = df.loc[:, ~df.columns.duplicated()]
        new_models[f"{key}.6"]["variables"], df = random_interactions(placeholder[:], df)
        new_models[f"{key}.6"]['data'] = df[new_models[f"{key}.6"]["variables"]]
        


    print(new_models.keys())

    models = new_models
    
    i += 1 

