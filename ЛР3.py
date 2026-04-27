import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

df = pd.read_csv("/Users/veraleonteva/Desktop/ML/ЛР3/job_salary_prediction_dataset.csv")

# Чистим experience_years
df["experience_years"] = pd.to_numeric(
    df["experience_years"],
    errors="coerce"
)

df["experience_years"].fillna(
    df["experience_years"].mean(),
    inplace=True
)

# Кодирование категорий
cat_cols = [
    "job_title",
    "education_level",
    "industry",
    "company_size",
    "location",
    "remote_work"
]

le = LabelEncoder()

for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# X и y
X = df.drop("salary", axis=1)
y = df["salary"]

# масштабирование
scaler = StandardScaler()
X = scaler.fit_transform(X)

# train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Исходная модель
model = KNeighborsRegressor(n_neighbors=5)
model.fit(X_train, y_train)

pred = model.predict(X_test)

print("ИСХОДНАЯ МОДЕЛЬ")
print("MAE:", mean_absolute_error(y_test, pred))
print("MSE:", mean_squared_error(y_test, pred))
print("R2 :", r2_score(y_test, pred))

# GridSearchCV
params = {"n_neighbors": [1,3,5,7,9,11,13,15]}

cv1 = KFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    KNeighborsRegressor(),
    params,
    cv=cv1,
    scoring="r2"
)

grid.fit(X_train, y_train)

print("\nGRIDSEARCHCV")
print("Лучший K:", grid.best_params_)
print("Лучший score:", grid.best_score_)

best_model = grid.best_estimator_
pred2 = best_model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, pred2))
print("MSE:", mean_squared_error(y_test, pred2))
print("R2 :", r2_score(y_test, pred2))

# RandomizedSearchCV
cv2 = ShuffleSplit(
    n_splits=5,
    test_size=0.2,
    random_state=42
)

rand = RandomizedSearchCV(
    KNeighborsRegressor(),
    params,
    n_iter=5,
    cv=cv2,
    random_state=42,
    scoring="r2"
)

rand.fit(X_train, y_train)

print("\nRANDOMIZEDSEARCHCV")
print("Лучший K:", rand.best_params_)
print("Лучший score:", rand.best_score_)