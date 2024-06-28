import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
import matplotlib.pyplot as plt


data = pd.read_csv('loan_data.csv')
print(data.head())
data.fillna(data.mean(), inplace=True)
features = ['credit_lines_outstanding', 'loan_amt_outstanding', 'total_debt_outstanding', 'income', 'years_employed', 'fico_score']
X = data[features]
y = data['default']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = LogisticRegression()
model.fit(X_train_scaled, y_train)


y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]


y_pred = model.predict(X_test_scaled)


roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f'ROC AUC Score: {roc_auc}')

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')


from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()



def predict_expected_loss(model, scaler, features, loan_amount, recovery_rate=0.1):

    features_scaled = scaler.transform([features])
    pd_default = model.predict_proba(features_scaled)[:, 1][0]
    expected_loss = pd_default * (1 - recovery_rate) * loan_amount
    
    return expected_loss

borrower_features = [5, 1958.93, 8228.75, 26648.44, 2, 572]
loan_amount = 250000
loss = predict_expected_loss(model, scaler, borrower_features, loan_amount)
print(f'Expected Loss: ${loss:.2f}')
