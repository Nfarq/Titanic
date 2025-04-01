import pandas as pd
import sklearn

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score





df = pd.read_csv('Titanic-Dataset.csv')

# drop un-needed data
df.drop(['Embarked', 'Cabin', 'Ticket'], axis=1, inplace=True)
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# change sex to 1 or 0
df['Sex'] = df['Sex'].apply(lambda x: 1 if x == 'female' else 0)

# change name to title
le = LabelEncoder()

df['Name'] = df['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())

df['Name'] = le.fit_transform(df['Name'])


# change age and fare to ints
df['Fare'] = le.fit_transform(df['Fare'])
df['Age'] = le.fit_transform(df['Age'])


# split into test and train
y = df['Survived']
X = df.drop(['Survived'], axis =1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Decision tree classifier
#model = DecisionTreeClassifier(random_state=42)
#model.fit(X_train, y_train)

# logistic regression model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
# check accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

