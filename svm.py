from numpy import genfromtxt, savetxt
from sklearn import cross_validation , grid_search
import numpy as np
import xlsxwriter
from sklearn.svm import SVC
from sklearn import svm
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split

#reading data from excel sheet
dataset = np.array(genfromtxt('/home/raksha/train_vectors.csv', delimiter='\t', dtype='f8')[:] )  
print type(dataset),dataset.shape
target = np.array([x[10] for x in dataset])
train = np.array([x[0:9] for x in dataset])

test = np.array(genfromtxt('/home/raksha/test_vectors.csv', delimiter='\t', dtype='f8')[:])


X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=0.1, random_state=0)
tuned_parameters = [{'C': [100,150,200,250,300,350,400,450,500]}]
rf = svm.LinearSVC()

scores = ['precision', 'recall']

for score in scores:
    
    clf = grid_search.GridSearchCV(rf, tuned_parameters, cv=5,
                       scoring='%s' % score)
    clf.fit(X_train,y_train)
    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()

    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()


prediction = []
prediction.extend(clf.predict(test[:,0:9]))

submission = xlsxwriter.Workbook('/home/raksha/submission.xlsx')
submission_sheet = submission.add_worksheet('test')

size = len(prediction)

for i in range(1,size):
    submission_sheet.write(i,0,test[i,0])
    submission_sheet.write(i,1,prediction[i])

submission.close()



      

