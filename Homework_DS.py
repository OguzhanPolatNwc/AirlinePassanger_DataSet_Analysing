import shutil
import numpy as np
import os
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
# Sklearn and decision tree
from sklearn.tree import export_graphviz
from io import StringIO
from IPython.display import Image
import pydotplus
# Import Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
# Import train_test_split function
from sklearn.model_selection import train_test_split
# Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

AirlinePassenger = pd.read_excel('AirlinePassenger.xlsm')

del AirlinePassenger['id']

Control = pd.DataFrame(columns=['Name', 'Mean', 'Median', 'Mode', 'STD', 'Q1', 'Q3', 'IQR', 'UpperBound', 'LowerBound'])

for x in AirlinePassenger:
    if str(AirlinePassenger.dtypes[x]) in ['float64', 'int64']:
        Control.loc[len(Control)] = [
            x,
            AirlinePassenger[x].mean(),
            AirlinePassenger[x].median(),
            AirlinePassenger[x].max(),
            AirlinePassenger[x].std(),
            AirlinePassenger[x].quantile(0.25),
            AirlinePassenger[x].quantile(0.75),
            AirlinePassenger[x].quantile(0.75) - AirlinePassenger[x].quantile(0.25),
            AirlinePassenger[x].quantile(0.75) + 1.5 * (
                    AirlinePassenger[x].quantile(0.75) - AirlinePassenger[x].quantile(0.25)),
            AirlinePassenger[x].quantile(0.25) - 1.5 * (
                    AirlinePassenger[x].quantile(0.75) - AirlinePassenger[x].quantile(0.25))
        ]
print(Control)
Control.to_csv('Control.csv')

AirlinePassenger.isna().sum()

# Boxplot
f, axes = plt.subplots(9, 2, figsize=(10, 15))
f.tight_layout(pad=5)
f.suptitle("Distrubition before processing")
cols = AirlinePassenger.select_dtypes(exclude='object').columns

x_axis = 0
y_axis = 0

for col in cols:
    sns.boxplot(data=AirlinePassenger, x=col, ax=axes[x_axis, y_axis])
    axes[x_axis, y_axis].set_xlabel(col)
    axes[x_axis, y_axis].set_ylabel("Count")
    axes[x_axis, y_axis].set_title(f"{col.title()} Box Count")

    if y_axis == 1:
        y_axis = 0
        x_axis += 1
    else:
        y_axis += 1

plt.savefig("boxplot.png")
plt.show()

# Classification based on age
Age_Group_avg = AirlinePassenger.groupby("Age").mean()
Age_Group = AirlinePassenger.groupby("Age").sum()

# Age and Flight Distance
Age_Group['Flight Distance'].plot()
plt.savefig('FlightDistance.png')
plt.show()

# Age and Wifi Service
Age_Group['Inflight wifi service'].plot()
plt.savefig('Inflightwifiservice.png')
plt.show()

# Age and Food/Drink
Age_Group['Food and drink'].plot()
plt.savefig('FoodandDrink.png')
plt.show()

# Age and Dep/Arr
Age_Group['Departure/Arrival time convenient'].plot()
plt.savefig('DepartureArrivaltimeconvenient.png')
plt.show()

# Convenient of flight and Food and drink
line1 = plt.plot(Age_Group_avg['Departure/Arrival time convenient'], label='Dep/Arr')
line2 = plt.plot(Age_Group_avg['Food and drink'], label='Food/Drink')
plt.legend(loc='upper center')
plt.savefig('ConvenientwithFood.png')
plt.show()

# Age and flight relation
plt.hist(AirlinePassenger['Age'])
plt.savefig('AgeandFlyrelation.png')
plt.show()

# Age and cleanliness
ax1 = sns.histplot(x=Age_Group_avg.index, y=Age_Group_avg['Cleanliness'].values, bins=15)
ax1.tick_params(axis='x', rotation=90)
plt.savefig("Cleanliness.png")
plt.show()

# Age and ease of booking
ax1 = sns.jointplot(x=Age_Group_avg.index, y=Age_Group_avg['Ease of Online booking'].values)
plt.savefig("EaseofOnlineBooking.png")
plt.show()

# Correlation matrix
AirlinePassenger['SatisfactionCheck'] = AirlinePassenger['satisfaction'].apply(lambda x: 1 if x == 'satisfied' else 0)
sns.set(rc={'figure.figsize': (15, 10)})
sns.heatmap(AirlinePassenger.corr(), annot=True)
plt.title("Correlation Matrix")
plt.savefig("CorrelationMatrix.png")
plt.show()

# Satisfaction Check
AirlinePassenger['SatisfactionCheck'] = AirlinePassenger['satisfaction'].apply(lambda x: 1 if x == 'satisfied' else 0)
Age_Group_avg = AirlinePassenger.groupby("Age").mean()
ax2 = sns.jointplot(x=Age_Group_avg.index, y=Age_Group_avg['SatisfactionCheck'].values)
plt.savefig("SatisfactionCheck.png")
plt.show()

# Decision tree
AirlinePassenger.head()
AirlinePassenger['Gender'] = AirlinePassenger['Gender'].replace({'Female': 1, 'Male': 0})
AirlinePassenger['Customer Type'] = AirlinePassenger['Customer Type'].replace(
    {'Loyal Customer': 1, 'disloyal Customer': 0})
AirlinePassenger['satisfaction'] = AirlinePassenger['satisfaction'].replace(
    {'satisfied': 1, 'neutral or dissatisfied': 0})
AirlinePassenger['Type of Travel'] = AirlinePassenger['Type of Travel'].replace(
    {'Personal Travel': 1, 'Business travel': 0})
del AirlinePassenger['Class']

AirlinePassenger.columns
Selected_cols = ['Gender', 'Customer Type', 'Age', 'Type of Travel',
                 'Flight Distance', 'Inflight wifi service',
                 'Departure/Arrival time convenient', 'Ease of Online booking',
                 'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort',
                 'Inflight entertainment', 'On-board service', 'Leg room service',
                 'Baggage handling', 'Checkin service', 'Inflight service',
                 'Cleanliness', 'Departure Delay in Minutes', 'Arrival Delay in Minutes',
                 ]
# AirlinePassenger.to_excel('Airline_Passenger.xlsm')
# Airline_Passanger = pd.read_excel('Airline_Passenger.xlsm')

X = AirlinePassenger[Selected_cols]  # Features
y = AirlinePassenger.satisfaction  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, )  # 70% training and 30% test

# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train, y_train, )

# Predict the response for test dataset
y_pred = clf.predict(X_test)

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True, feature_names=Selected_cols, class_names=['0', '1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('FullTreeClassifier.png')
Image(graph.create_png())


# Deep 3 ,tree classifier
clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)

# Train Decision Tree Classifer
clf = clf.fit(X_train, y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True, feature_names=Selected_cols, class_names=['0', '1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('FullTreeClassifier_Deep3.png')
Image(graph.create_png())


# Decision tree _ Least effective columns deleted
AirlinePassenger.head()
AirlinePassenger['Gender'] = AirlinePassenger['Gender'].replace({'Female': 1, 'Male': 0})
AirlinePassenger['Customer Type'] = AirlinePassenger['Customer Type'].replace(
    {'Loyal Customer': 1, 'disloyal Customer': 0})
AirlinePassenger['satisfaction'] = AirlinePassenger['satisfaction'].replace(
    {'satisfied': 1, 'neutral or dissatisfied': 0})
AirlinePassenger['Type of Travel'] = AirlinePassenger['Type of Travel'].replace(
    {'Personal Travel': 1, 'Business travel': 0})
del AirlinePassenger['Departure Delay in Minutes']
del AirlinePassenger['Arrival Delay in Minutes']
del AirlinePassenger['Departure/Arrival time convenient']
del AirlinePassenger['Gate location']

"""
hayat = list(range(10,0,-1))
print(hayat)


hayat = list(reversed(range(1,10)))
print(hayat)
"""

AirlinePassenger.columns
Selected_cols = ['Gender', 'Customer Type', 'Age', 'Type of Travel',
                 'Flight Distance', 'Inflight wifi service','Ease of Online booking',
                 'Food and drink', 'Online boarding', 'Seat comfort',
                 'Inflight entertainment', 'On-board service', 'Leg room service',
                 'Baggage handling', 'Checkin service', 'Inflight service',
                 'Cleanliness',
                 ]
# AirlinePassenger.to_excel('Airline_Passenger.xlsm')
# Airline_Passanger = pd.read_excel('Airline_Passenger.xlsm')

X = AirlinePassenger[Selected_cols]  # Features
y = AirlinePassenger.satisfaction  # Target variable

# We used random state one to have our data shuffled each time we run but that would prevent objective assesment
# of changes. If we change random state value to something but 1 then we would have our data shuffled same way
# every time
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, )  # 70% training and 30% test

# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)

# Train Decision Tree Classifer
clf = clf.fit(X_train, y_train, )

# Predict the response for test dataset
y_pred = clf.predict(X_test)

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True, feature_names=Selected_cols, class_names=['0', '1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('FullTreeClassifier_Deep3_2.png')
Image(graph.create_png())

