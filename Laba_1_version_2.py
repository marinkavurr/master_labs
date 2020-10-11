
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier





missing_values = ["n/a", "na", "?"]

bands = pd.read_csv(r"D:\stat\master_final\bands.txt", sep=',',na_values = missing_values)
#print(bands)
data=bands[['proof_cut',
       'viscosity', 'caliper', 'ink_temperature', 'humifity', 'roughness',
       'blade_pressure', 'varnish_pct', 'press_speed', 'ink_pct',
       'solvent_pct', 'ESA_Voltage', 'ESA_Amperage', 'wax', 'hardener',
       'roller_durometer', 'current_density', 'anode_space_ratio',
       'chrome_content','band_type']]

### delete missing values
data=data.dropna()


X = data.iloc[:, :-1].values
Y = data.iloc[:, 19].values

########################################################################################
########################--------KNN------------#########################################
########################################################################################

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20, random_state=1)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 10)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
result = confusion_matrix(y_test, y_pred)
print("Confusion Matrix KNN:")
print(result)
#result1 = classification_report(y_test, y_pred)
#print("Classification Report:",)
#print (result1)
result2 = accuracy_score(y_test,y_pred)
print("Accuracy KNN:",result2)


#SVM
from sklearn.svm import SVC
SVC_model = SVC(kernel='linear',gamma='scale')
SVC_model.fit(X_train, y_train)
SVC_prediction = SVC_model.predict(X_test)

print("Accuracy SWM:",accuracy_score(SVC_prediction, y_test))
print("Confusion Matrix SWM:")
print(confusion_matrix(SVC_prediction, y_test))



tree = DecisionTreeClassifier(random_state=1)
tree.fit(X_train, y_train)


#Decision_trees
DT_predictions = tree.predict(X_test)
print("Accuracy DT:",accuracy_score(DT_predictions, y_test))
print("Confusion Matrix DT:")
print(confusion_matrix(DT_predictions, y_test))
# Calculate the absolute errors

#### adding factor data

data2=bands[['cylinder_number','proof_cut',
       'viscosity', 'caliper', 'ink_temperature', 'humifity', 'roughness',
       'blade_pressure', 'varnish_pct', 'press_speed', 'ink_pct',
       'solvent_pct', 'ESA_Voltage', 'ESA_Amperage', 'wax', 'hardener',
       'roller_durometer', 'current_density', 'anode_space_ratio',
       'chrome_content','band_type']]
data2=data2.dropna()
number= ['A', 'B', 'X', 'R', 'O', 'G', 'M', 'I', 'W', 'F', 'E', 'J', 'S', 'T', 'V', 'Y']

data2['cylinder_number']=data2['cylinder_number'].str[0]

#modify factor data
i=float(0)
for i in range(15):
       data2.cylinder_number[data2["cylinder_number"]==number[i]]=i
       i=+1

data2['cylinder_number']=pd.factorize(data2['cylinder_number'])[0]


X2 = data2.iloc[:, :-1].values
Y2 = data2.iloc[:, 20].values


########################################################################################
########################--------KNN------------#########################################
########################################################################################
print("added factor data")
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, Y2, test_size = 0.20, random_state=1)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train2)
X_train2 = scaler.transform(X_train2)
X_test2 = scaler.transform(X_test2)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 8)
classifier.fit(X_train2, y_train2)

y_pred = classifier.predict(X_test2)


result = confusion_matrix(y_test2, y_pred)
print("Confusion Matrix KNN:")
print(result)
result2 = accuracy_score(y_test2,y_pred)
print("Accuracy KNN:",result2)


SVC_model = SVC(kernel='linear',gamma='scale')
SVC_model.fit(X_train2, y_train2)
SVC_prediction = SVC_model.predict(X_test2)

print("Accuracy SWM:",accuracy_score(SVC_prediction, y_test2))
print("Confusion Matrix SWM:")
print(confusion_matrix(SVC_prediction, y_test2))



tree = DecisionTreeClassifier(random_state=1)
tree.fit(X_train2, y_train2)


#Decision_trees
DT_predictions = tree.predict(X_test2)
print("Accuracy DT:",accuracy_score(DT_predictions, y_test2))
print("Confusion Matrix DT:")
print(confusion_matrix(DT_predictions, y_test2))