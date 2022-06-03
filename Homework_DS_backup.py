import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import os

AirlinePassenger = pd.read_excel('AirlinePassenger.xlsm')

Control = pd.DataFrame(columns=['name', 'mean', 'median', 'std', 'kurtosis', 'skewness', 'upper', 'lower'])
AirlinePassenger_Control = pd.read_excel('AirlinePassenger.xlsm')

for x in AirlinePassenger_Control.columns:
    if str(AirlinePassenger_Control.dtypes[x]) in ['int64', 'float64']:
        Mean = AirlinePassenger_Control[x].mean()
        Medium = AirlinePassenger_Control[x].median()
        std = AirlinePassenger_Control[x].std()
        Kurtosis = AirlinePassenger_Control[x].kurtosis()
        Skew = AirlinePassenger_Control[x].skew()
        q1 = AirlinePassenger_Control[x].quantile(0.25)
        q3 = AirlinePassenger_Control[x].quantile(0.75)
        iqr = q3 - q1
        upper_bound = q3 + 1.5 * iqr
        lower_bound = q1 - 1.5 * iqr
        Control.loc[len(Control)] = [x, Mean, Medium, std, Kurtosis, Skew, upper_bound, lower_bound]

print(Control)

# To show main statistics of dataset
Description = AirlinePassenger.describe()

# Compare 2 column
AirlinePassenger.plot(x='Flight Distance', y='Inflight wifi service', style='o')
plt.show()

# Age with flight quality
AirlinePassenger.plot(x="Age", y=["Flight Distance", "Inflight wifi service", ])
plt.show()

# PDF
all_files = os.listdir("Project/")
report = [f"plots{file}" for file in all_files]
print(report)

from fpdf import FPDF

WIDTH = 210
HEIGHT = 297

pdf = FPDF()
pdf.set_font("Arial", "B", 56)

pdf.add_page()
pdf.cell(180, 20, txt='REPORT', align='C')

for report in report:
    pdf.add_page()

    pdf.set_font("Arial", "B", 56)
    pdf.cell(180, 20, txt='REPORT', align='C')

    pdf.image(report, 5, 30, WIDTH - 5)

pdf.output("Countries_Report.pdf", "F")


# or

from fpdf import FPDF

pdf = FPDF()

pdf.add_page()

pdf.set_font("Arial", size=25)

# create a cell
pdf.cell(200, 10, txt="JournalDev",
         ln=1, align='C')

pdf.image(img="boxplot.png")
pdf.cell(200, 10, txt="Welcome to the world of technologies!",
         ln=2, align='C')

pdf.output("data.pdf")


# Classification based on age
Age_Group_avg = AirlinePassenger.groupby("Age").mean()
Age_Group = AirlinePassenger.groupby("Age").sum()

# Convenient of flight and Food and drink
Age_Group_avg['Departure/Arrival time convenient'].plot()
Age_Group_avg['Food and drink'].plot()
# plt.savefig('ConvenientwithFood.png')
plt.show()

# f, axe explanation
'''plt.subplots() is a function that returns a tuple containing a figure and axes object(s). 
Thus when using fig, ax = plt.subplots() you unpack this tuple into the variables fig and ax. 
Having fig is useful if you want to change figure-level attributes or save the figure as an image 
file later (e.g. with fig.savefig('yourfilename.png')). You certainly don't have to use the returned 
figure object but many people do use it later so it's common to see. Also, all axes objects 
(the objects that have plotting methods), have a parent figure object anyway, thus:'''

fig, ax = plt.subplots()
'''is more concise than this: '''

fig = plt.figure()
ax = fig.add_subplot()

# Histogram
f, axes = plt.subplots(10, 2, figsize=(10, 15))
f.tight_layout(pad=3)
f.suptitle("Distrubition before processing")
cols = AirlinePassenger.select_dtypes(exclude='object').columns

x_axis = 0
y_axis = 0

for col in cols:
    sns.histplot(data=AirlinePassenger, x=col, kde=True, ax=axes[x_axis,y_axis])
    axes[x_axis,y_axis].set_xlabel(col.title())
    axes[x_axis,y_axis].set_ylabel(col.title())
    axes[x_axis,y_axis].set_title(f"{col.title()} Count")

    if y_axis ==1:
        y_axis=0
        x_axis +=1
    else:
        y_axis +=1

plt.savefig("histogram.png")
plt.show()


from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Decision Tree
AirlinePassenger = AirlinePassenger.sample(frac = 1.0)

columns = list(AirlinePassenger.columns)
columns.remove('SatisfactionCheck')


#: SPLIT INTO TRAIN TEST
number_of_rows = len(AirlinePassenger) #129880
train_count = int(number_of_rows * 0.60)

train = AirlinePassenger[:train_count]
test = AirlinePassenger[train_count:]

train_y = train['Age']
train_x = train[columns]

test_y = test['Age']
test_x = test[columns]


#: CREATE THE CLASSIFIERS

algorithms = [
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GaussianNB(),
    svm.SVC(),
    #NearestNeighbors(),
    MLPClassifier(random_state=1, max_iter=300)
]


for a in algorithms:
    a.fit( train_x, train_y )
    pred = a.predict( test_x )
    print( a, accuracy_score( test_y, pred ) )

for o in AirlinePassenger:
    print( o, AirlinePassenger[o].corr( AirlinePassenger['Age']))




"""
for leaf_size in [5, 10, 20, 40, 50, 80, 100, 125, 175, 250]:    
    clf = RandomForestClassifier(max_depth=6, min_samples_leaf=leaf_size) # , criterion = 'entropy'
    clf.fit( train_x, train_y )
    #! pred = clf.predict( test_x )
    print("Score", clf.score( test_x, test_y ), leaf_size)
"""

for d in [2,3,4,5,6,7,8,9,10]:
    clf = RandomForestClassifier(max_depth=d, min_samples_leaf=20) # , criterion = 'entropy'
    clf.fit( train_x, train_y )
    #! pred = clf.predict( test_x )
    print("Score", clf.score( test_x, test_y ), d)







# Decision tree 2

AirlinePassenger = pd.read_excel("AirlinePassenger.xlsm")
AirlinePassenger['Gender'] = AirlinePassenger['Gender'].replace({'Female': 1, 'Male': 0})
AirlinePassenger['Customer Type'] = AirlinePassenger['Customer Type'].replace(
    {'Loyal Customer': 1, 'disloyal Customer': 0})
AirlinePassenger['Type of Travel'] = AirlinePassenger['Type of Travel'].replace(
    {'Personal Travel': 1, 'Business travel': 0})
AirlinePassenger['satisfaction'] = AirlinePassenger['satisfaction'].replace(
    {'satisfied': 1, 'neutral or dissatisfied': 0})
del AirlinePassenger['Class']
del AirlinePassenger['id']
del AirlinePassenger['SatisfactionCheck']

#: SHUFFLE
AirlinePassenger = AirlinePassenger.sample(frac=1.0)

columns = list(AirlinePassenger.columns)

#: SPLIT INTO TRAIN TEST
number_of_rows = len(AirlinePassenger)  # 129880
train_count = int(number_of_rows * 0.60)

train = AirlinePassenger[:train_count]
test = AirlinePassenger[train_count:]

train_y = train['satisfaction']
train_x = train[columns]

test_y = test['satisfaction']
test_x = test[columns]

#: CREATE THE CLASSIFIERS

algorithms = [
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GaussianNB(),
    svm.SVC(),
    # NearestNeighbors(),
    MLPClassifier(random_state=1, max_iter=300)
]

for a in algorithms:
    a.fit(train_x, train_y)
    pred = a.predict(test_x)
    print(a, accuracy_score(test_y, pred))

for o in AirlinePassenger:
    print(o, AirlinePassenger[o].corr(AirlinePassenger['satisfaction']))

"""
Gender -0.011236116189643694
Customer Type 0.1860171906955102
Age 0.13409123867754635
Type of Travel -0.4498612546656334
Flight Distance 0.29808489854594195
Inflight wifi service 0.28346023010121646
Departure/Arrival time convenient -0.05426971049373724
Ease of Online booking 0.1688771390528305
Gate location -0.0027932746524710794
Food and drink 0.2113402076250672
Online boarding 0.5017494207376233
Seat comfort 0.3488293461025928
Inflight entertainment 0.39823365061187715
On-board service 0.3222048233927061
Leg room service 0.3124238194944813
Baggage handling 0.2486799187751388
Checkin service 0.23725236030900043
Inflight service 0.24491783574569506
Cleanliness 0.30703467056329875
Departure Delay in Minutes -0.05073986595225647
Arrival Delay in Minutes -0.058275092682866646
satisfaction 0.9999999999999999
"""

for leaf_size in [5, 10, 20, 40, 50, 80, 100, 125, 175, 250]:
    clf = RandomForestClassifier(max_depth=6, min_samples_leaf=leaf_size)  # , criterion = 'entropy'
    clf.fit(train_x, train_y)
    # ! pred = clf.predict( test_x )
    print("Score", clf.score(test_x, test_y), leaf_size)


# Get dummies

import pandas as pd

AirlinePassenger = pd.read_excel('AirlinePassenger.xlsm')

df_dummies = pd.get_dummies(AirlinePassenger)

df_dummies.to_csv('df_dummies.csv')




