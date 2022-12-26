import random

Q = ["Phở", "Cơm tấm", "Bánh mì", "Súp cua"]

n_obs = 365

data = [random.choice(Q) for i in range(n_obs)]

data = "\n".join(data)

with open("breakfast.csv", "w", encoding="utf-8") as f:
    f.write("Food\n")
    f.write(data)
