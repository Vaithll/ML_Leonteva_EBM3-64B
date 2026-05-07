
from sklearn.datasets import load_wine
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df["target"] = wine.target

# cоздание категориального признака
df["alcohol_level"] = pd.cut(
    df["alcohol"],
    bins=3,
    labels=["low", "medium", "high"]
)

# добавление пропусков
np.random.seed(42)

num_missing_idx = np.random.choice(df.index, size=int(0.1 * len(df)), replace=False)
df.loc[num_missing_idx, "alcohol"] = np.nan
cat_missing_idx = np.random.choice(df.index, size=int(0.1 * len(df)), replace=False)
df.loc[cat_missing_idx, "alcohol_level"] = np.nan
print("Пропуски ДО обработки:\n", df.isnull().sum())

df["alcohol"] = df["alcohol"].fillna(df["alcohol"].median())
df["alcohol_level"] = df["alcohol_level"].fillna(df["alcohol_level"].mode()[0])
print("\nПропуски ПОСЛЕ обработки:\n", df.isnull().sum())

# диаграмма рассеяния
plt.figure(figsize=(8, 5))
scatter = plt.scatter(df["alcohol"], df["malic_acid"], c=df["target"], cmap="viridis", alpha=0.7, edgecolors="k")
plt.xlabel("Alcohol")
plt.ylabel("Malic Acid")
plt.title("Диаграмма рассеяния: Alcohol vs Malic Acid (цвет = класс вина)")
plt.colorbar(scatter, label="Wine Class")
plt.grid(True)
plt.show()


