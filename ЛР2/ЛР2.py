import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

df = pd.read_csv("job_salary_prediction_dataset.csv")

print("Первые 5 строк:")
print(df.head())

print("\nРазмер таблицы:")
print(df.shape)

print("\nПропуски по столбцам:")
print(df.isnull().sum())

# создадим пропуски
df.loc[0, "education_level"] = None
df.loc[3, "skills_count"] = None

print("\nПосле добавления пропусков:")
print(df.isnull().sum())

df["skills_count"].fillna(df["skills_count"].mean(), inplace=True)

df["education_level"].fillna(df["education_level"].mode()[0], inplace=True)

print("\nПосле заполнения пропусков:")
print(df.isnull().sum())

cat_cols = [
    "job_title",
    "education_level",
    "industry",
    "company_size",
    "location",
    "remote_work"
]

encoder = LabelEncoder()

for col in cat_cols:
    df[col] = encoder.fit_transform(df[col])

print("\nПосле кодирования:")
print(df.head())

num_cols = [
    "experience_years",
    "skills_count",
    "certifications",
    "salary"
]

scaler = StandardScaler()

df[num_cols] = scaler.fit_transform(df[num_cols])

print("\nПосле масштабирования:")
print(df.head())