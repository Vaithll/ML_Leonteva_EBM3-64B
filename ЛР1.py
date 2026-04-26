import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris

iris = load_iris()
df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)
df["target"] = iris.target

print("Первые 5 строк:")
print(df.head())

print("\nРазмер таблицы:")
print(df.shape)

print("\nИнформация:")
print(df.info())

print("\nСтатистика:")
print(df.describe())

df.hist(figsize=(10,8))
plt.suptitle("Гистограммы")
plt.show()

plt.figure(figsize=(10,6))
sns.boxplot(data=df.iloc[:,0:4])
plt.title("Boxplot")
plt.show()

sns.scatterplot(
    data=df,
    x="sepal length (cm)",
    y="petal length (cm)",
    hue="target"
)
plt.title("Диаграмма рассеяния")
plt.show()

corr = df.corr()

plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Корреляционная матрица")
plt.show()