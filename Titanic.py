import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the data
train_df = pd.read_csv('C:\\Users\\HP\\OneDrive\\Desktop\\Python\\Titanic Classification\\train.csv')
test_df = pd.read_csv('C:\\Users\\HP\\OneDrive\\Desktop\\Python\\Titanic Classification\\test.csv')

# Preprocess the data
data = [train_df, test_df]
for dataset in data:
    mean_age = dataset["Age"].mean()
    std_age = dataset["Age"].std()
    is_null_age = dataset["Age"].isnull().sum()
    rand_age = np.random.randint(mean_age - std_age, mean_age + std_age, size=is_null_age)
    age_slice = dataset["Age"].copy()
    age_slice[np.isnan(age_slice)] = rand_age
    dataset["Age"] = age_slice
    dataset["Age"] = dataset["Age"].astype(int)

train_df['Embarked'] = train_df['Embarked'].fillna(train_df['Embarked'].mode()[0])
test_df['Embarked'] = test_df['Embarked'].fillna(train_df['Embarked'].mode()[0])

X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test = test_df.drop("PassengerId", axis=1).copy()

# Feature engineering
X_train['CabinBool'] = (X_train['Cabin'].notnull()).astype(int)
X_test['CabinBool'] = (X_test['Cabin'].notnull()).astype(int)

# Scale the data
scaler = StandardScaler()
X_train[['Age', 'Fare']] = scaler.fit_transform(X_train[['Age', 'Fare']])
X_test[['Age', 'Fare']] = scaler.transform(X_test[['Age', 'Fare']])

# Train the models
sgd = SGDClassifier(max_iter=5, tol=None)
sgd.fit(X_train, Y_train)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
acc_rf = round(random_forest.score(X_train, Y_train) * 100, 2)

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, Y_train)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)

# Evaluate the models
print("SGD Accuracy:", acc_sgd)
print("Random Forest Accuracy:", acc_rf)
print("Logistic Regression Accuracy:", acc_log)
print("KNN Accuracy:", acc_knn)

# Plot the ROC curve
false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_train, random_forest.predict_proba(X_train)[:, 1])
roc_auc = auc(false_positive_rate, true_positive_rate)
plt.figure(figsize=(14, 7))
plt.plot(false_positive_rate, true_positive_rate, color='b', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='r', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

# Save the predictions to a CSV file
submission_df = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Survived': random_forest.predict(X_test)})
submission_df.to_csv('submission.csv', index=False)
