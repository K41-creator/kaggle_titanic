import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
##print(train.head())
##print(train.info())
##print(test.head())

##print(train.groupby("Sex")["Survived"].mean())
##print(train.groupby("Pclass")["Survived"].mean())
'''
train["Age"] = train["Age"].fillna(
    train["Age"].median()
)
test["Age"] = test["Age"].fillna(
    train["Age"].median()
)

test["Fare"] = test["Fare"].fillna(
    train["Fare"].median()
)

train["FamilySize"] = (
    train["SibSp"]
    + train["Parch"]
    + 1
)

test["FamilySize"] = (
    test["SibSp"]
    + test["Parch"]
    + 1
)
'''
train["Age"] = train["Age"].fillna(
    train["Age"].median()
)
test["Age"] = test["Age"].fillna(
    train["Age"].median()
)

test["Fare"] = test["Fare"].fillna(
    train["Fare"].median()
)

train["FamilySize"] = (
    train["SibSp"]
    + train["Parch"]
    + 1
)

test["FamilySize"] = (
    test["SibSp"]
    + test["Parch"]
    + 1
)

train["Embarked"] = train["Embarked"].fillna(
    train["Embarked"].mode()[0]
)

test["Embarked"] = test["Embarked"].fillna(
    train["Embarked"].mode()[0]
)

##特徴量
features = [
    "Pclass",
    "Sex",
    "Age",
    "Fare",
    "FamilySize",
    "Embarked"
]

x = pd.get_dummies(train[features])
x_test = pd.get_dummies(test[features])

y = train["Survived"]

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=5,
    random_state=42
)


X_train, X_valid, y_train, y_valid = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)
model.fit(
    x,
    y
    )

predictions = model.predict(x_test)

submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": predictions
})

submission.to_csv(
    "submission.csv",
    index=False
)


##検証する
pred = model.predict(X_valid)
score = accuracy_score(
    y_valid,
    pred
)

print(score)
for name, importance in zip(
    x.columns,
    model.feature_importances_
):
    print(name, importance)
