from pydataset import data
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

melanoma_data = data('Melanoma')

## Removed rows for which status = 3 (Died due to caused unrelated to Melanoma)
melanoma_data_fmt = melanoma_data.loc[melanoma_data['status'] != 3]

X = melanoma_data_fmt.loc[:,['time', 'sex', 'age', 'thickness', 'ulcer']]
y = melanoma_data_fmt.loc[:,['status']]

accuracy_list = []
precision_list = []
recall_list = []

for i in list(range(5)):
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
	clf = RandomForestClassifier(n_estimators=100, max_depth=8)
	clf.fit(X_train, y_train)
	y_pred = clf.predict(X_test)
	accuracy_list.append(accuracy_score(y_test,y_pred))
	precision_list.append(precision_score(y_test,y_pred))
	recall_list.append(recall_score(y_test,y_pred))

## Calculate Mean and SD of Accuracy, Precision, and Recall
mean_acc = np.mean(accuracy_list)
std_acc = np.std(accuracy_list)
mean_precision = np.mean(precision_list)
std_precision = np.std(precision_list)
mean_recall = np.mean(recall_list)
std_recall = np.std(recall_list)

print("Mean Accuracy Score on the Test data: %f" % mean_acc)
print("Standard Deviation of Accuracy Score on the Test data: %f" % std_acc)
print("Mean Precision Score on the Test data: %f" % mean_precision)
print("Standard Deviation of Precision Score on the Test data: %f" % std_precision)
print("Mean Recall Score on the Test data: %f" % mean_recall)
print("Standard Deviation of Recall Score on the Test data: %f" % std_recall)

plt.xticks([1,2,3,4,5])
plt.plot(x_val, accuracy_list,color='blue',label='Accuracy')
plt.plot(x_val, precision_list,color='red',label='Precision')
plt.plot(x_val, recall_list,color='green',label='Recall')

plt.plot(x_val, [mean_acc]*5,'b:',label='Mean Accuracy')
plt.plot(x_val, [mean_precision]*5,'r:',label='Mean Precision')
plt.plot(x_val, [mean_recall]*5,'g:',label='Mean Recall')

plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()