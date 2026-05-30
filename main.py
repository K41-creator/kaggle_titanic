import pandas as pd
from sklearn.ensemble import RandomForestClassifier
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
##print(train.head())
##print(train.info())
##print(test.head())

##print(train.groupby("Sex")["Survived"].mean())
##print(train.groupby("Pclass")["Survived"].mean())


##特徴量
features = [
    "Pclass",
    "Sex",
    "SibSp",
    "Parch"
]

x = pd.get_dummies(train[features])
x_test = pd.get_dummies(test[features])

y = train["Survived"]

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(x,y)

predictions = model.predict(x_test)

submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": predictions
})

submission.to_csv(
    "submission.csv",
    index=False
)