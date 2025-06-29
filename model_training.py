# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report
# from sklearn.preprocessing import StandardScaler
# import joblib

# data = pd.read_csv('creditcard.csv')

# X = data.drop('Class', axis=1)
# y = data['Class']

# scaler = StandardScaler()
# X[['Amount', 'Time']] = scaler.fit_transform(X[['Amount', 'Time']])

# joblib.dump(X.columns.tolist(), 'columns.pkl')

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y
# )

# model = RandomForestClassifier(
#     n_estimators=100, class_weight='balanced', random_state=42
# )
# model.fit(X_train, y_train)

# y_pred = model.predict(X_test)
# print(classification_report(y_test, y_pred))

# joblib.dump(model, 'fraud_model.pkl')
# joblib.dump(scaler, 'scaler.pkl')

# print("✅ Model, scaler, and column list saved successfully.")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import joblib

data = pd.read_csv('creditcard.csv')

X = data.drop('Class', axis=1)
y = data['Class']

scaler = StandardScaler()
X[['Amount', 'Time']] = scaler.fit_transform(X[['Amount', 'Time']])

joblib.dump(X.columns.tolist(), 'columns.pkl')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

joblib.dump(model, 'fraud_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("✅ Model training completed and files saved successfully.")
