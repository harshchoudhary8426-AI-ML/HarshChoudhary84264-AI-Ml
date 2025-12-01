import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline

# LOAD DATA
# Make sure you upload/keep students.csv in this repository.
data = pd.read_csv("students.csv")

# Expected columns:
# 'hours_studied', 'previous_score', 'attendance', 'gender', 'parent_education', 'final_score'

target = "final_score"
X = data.drop(columns=[target])
y = data[target]

numeric_features = X.select_dtypes(include=['float64', 'int64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

model = RandomForestRegressor(n_estimators=200, random_state=42)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])

# SPLIT
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TRAIN
pipeline.fit(X_train, y_train)

# EVALUATE
y_pred = pipeline.predict(X_test)
score = r2_score(y_test, y_pred)

print("R² Score:", round(score, 3))

# SAMPLE PREDICTIONS
print("\nSample predictions:")
print(pd.DataFrame({
    "Actual": y_test[:10].values,
    "Predicted": y_pred[:10]
}))
