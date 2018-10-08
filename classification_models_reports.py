from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report


def get_models_score(X, y):

    print("Classification started ")
    # LogisticRegression
    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression(random_state=0)
    classifier.fit(X, y)
    y_pred = classifier.predict(X)
    LogisticRegression_score = accuracy_score(y, y_pred)
    print("Logistic score :",LogisticRegression_score)
    print('\n')
    print("Logistic  classification report")
    print('\n')
    get_metrix(y, y_pred)

    print('--------------------------------------------------------------------')
    # knn
    from sklearn.neighbors import KNeighborsClassifier
    classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
    classifier.fit(X, y)
    y_pred = classifier.predict(X)
    knn_score = accuracy_score(y, y_pred)
    print("KNeighbors score :",knn_score)
    print('\n')
    print("KNeighbors  classification report")
    print('\n')
    get_metrix(y, y_pred)
    print('--------------------------------------------------------------------')

    # svc
    from sklearn.svm import SVC
    classifier = SVC(kernel='linear', random_state=0)
    classifier.fit(X, y)
    y_pred = classifier.predict(X)
    svc_score = accuracy_score(y, y_pred)
    print("SVC score :",svc_score)
    print('\n')
    print("SVC  classification report")
    print('\n')
    get_metrix(y, y_pred)
    print('--------------------------------------------------------------------')

    # DecisionTreeClassifier
    from sklearn.tree import DecisionTreeClassifier
    classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
    classifier.fit(X, y)
    y_pred = classifier.predict(X)
    Dtree_score = accuracy_score(y, y_pred)
    print("Dtree score :",Dtree_score)
    print('\n')
    print("Dtree  classification report")
    print('\n')
    get_metrix(y, y_pred)
    print('--------------------------------------------------------------------')

    # RandomForestClassifier
    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
    classifier.fit(X, y)
    y_pred = classifier.predict(X)
    RF_score = accuracy_score(y, y_pred)
    print("RandomForest score :",RF_score)
    print('\n')
    print("RandomForest  classification report")
    print('\n')
    get_metrix(y, y_pred)

    # svm
    from sklearn.svm import SVC
    classifier = SVC(kernel='rbf', random_state=10)
    classifier.fit(X, y)
    y_pred = classifier.predict(X)
    svm_score = accuracy_score(y, y_pred)
    print("SVC score :",svm_score)
    print('\n')
    print("svc rbf kernal  classification report")
    print('\n')
    get_metrix(y, y_pred)



    # naive_bayes
    from sklearn.naive_bayes import GaussianNB
    classifier = GaussianNB()
    classifier.fit(X, y)
    y_pred = classifier.predict(X)
    nb_score = accuracy_score(y, y_pred)
    print("naive_bayes score :",nb_score)
    print('\n')
    print("naive_bayes  classification report")
    print('\n')
    get_metrix(y, y_pred)

    print("Bar plot generating")


    objects = ('Logistic', 'knn', 'svm', 'RF', 'Dtree', 'svc', 'nb')
    y_pos = np.arange(len(objects))
    performance = [LogisticRegression_score, knn_score, svm_score, RF_score, Dtree_score, svc_score, nb_score]

    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('accuracy')
    plt.title('ML alogrithms')

    plt.show()

def get_metrix(y, y_pred):
    target_names = list(set(y))
    target_names = [str(i) for i in target_names]
    print(classification_report(y, y_pred, target_names=target_names))
    cm = confusion_matrix(y, y_pred)
    print("confusion_matrix")
    print('\n')
    print(cm)
    print('\n')
    sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    print('Sensitivity : ', sensitivity)
    print('\n')
    specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    print('Specificity : ', specificity)