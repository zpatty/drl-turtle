import numpy as np
import json

num_params = 21
# mu = np.random.rand((num_params)) 
# sigma = np.random.rand((num_params)) + 0.9
mu = np.array([5.5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
sigma = np.ones((num_params)) * 0.3 
sigma[0] = 0.1

print(f"starting mu: {mu}\n")
print(f"starting sigma: {sigma}\n")

with open('config.json') as config:
    params = json.load(config)
print(f"params: {params}")
params["mu"] = list(mu)
params["sigma"] = list(sigma)

config_params = json.dumps(params, indent=10)
print(f"config params: {config_params}")
# print(config_params)


with open('config.json', "w") as outfile:
    outfile.write(config_params)

