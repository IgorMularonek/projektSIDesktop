import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
import pickle

# Wczytanie danych
df = pd.read_csv('train.csv')

# Zamiana nazw kolumn na małe litery
df.columns = [col.lower() for col in df.columns]

# Imputacja braków
for col in df.columns:
    if df[col].isnull().any():
        if df[col].dtype in ['float64', 'int64']:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

# Enkodowanie kolumn kategorycznych
cat_cols = ['gender', 'customer type', 'type of travel', 'class', 'satisfaction']
label_encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Wybór tylko tych cech, które są używane w GUI
X = df[['gender', 'customer type', 'type of travel', 'class', 'age', 'flight distance']]
y = df['satisfaction']

# Podział danych
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Modele
models = {
    "knn": KNeighborsClassifier(n_neighbors=5),
    "decision_tree": DecisionTreeClassifier(),
    "random_forest": RandomForestClassifier(n_estimators=100),
}

# Trenowanie i zapis
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} accuracy: {acc:.4f}")
    with open(f"{name}_model.pkl", "wb") as f:
        pickle.dump(model, f)

# Regresja liniowa (zaokrąglona)
linreg = LinearRegression()
linreg.fit(X_train, y_train)
y_pred_lr = linreg.predict(X_test)
y_pred_lr_rounded = y_pred_lr.round().clip(0, y.nunique() - 1).astype(int)
acc_lr = accuracy_score(y_test, y_pred_lr_rounded)
print(f"linear_regression accuracy (rounded): {acc_lr:.4f}")
with open("linear_regression_model.pkl", "wb") as f:
    pickle.dump(linreg, f)

# Zapis enkoderów
with open("label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)
