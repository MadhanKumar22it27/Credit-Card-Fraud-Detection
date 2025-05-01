import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
# Optional (for handling imbalanced data)
from imblearn.over_sampling import SMOTE

df = pd.read_csv('creditcard.csv')
print(df.head())

print(df.info())
print(df['Class'].value_counts())  # 0 = Not Fraud, 1 = Fraud

# Check imbalance
sns.countplot(x='Class', data=df)
plt.title("Class Distribution")
plt.show()

# Features and labels
X = df.drop('Class', axis=1)
y = df['Class']

# Scale the 'Amount' and 'Time'
scaler = StandardScaler()
X['Amount'] = scaler.fit_transform(X['Amount'].values.reshape(-1, 1))
X['Time'] = scaler.fit_transform(X['Time'].values.reshape(-1, 1))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)

# Check new class balance
print(pd.Series(y_res).value_counts())

model = LogisticRegression()
model.fit(X_res, y_res)

y_pred = model.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
plt.show()

print("ROC AUC Score:", roc_auc_score(y_test, y_pred))