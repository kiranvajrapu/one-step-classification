def get_models_score(X,y):
    from sklearn.metrics import accuracy_score
    import matplotlib.pyplot as plt

    print("Classification started ")
    #LogisticRegression
    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression(random_state = 0)
    classifier.fit(X, y)
    y_pred = classifier.predict(X)
    LogisticRegression_score = accuracy_score(y, y_pred)
    #print("LogisticRegression_score")
	#print(LogisticRegression_score)

    
    #knn
    from sklearn.neighbors import KNeighborsClassifier
    classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
    classifier.fit(X, y)
    y_pred = classifier.predict(X)
    knn_score = accuracy_score(y, y_pred)
    #print("KNeighborsClassifier_score",knn_score)

    #svc
    from sklearn.svm import SVC
    classifier = SVC(kernel = 'linear', random_state = 0)
    classifier.fit(X, y)
    y_pred = classifier.predict(X)
    svc_score = accuracy_score(y, y_pred)
	#z3int("SVC_score",svc_score)

    
    #DecisionTreeClassifier
    from sklearn.tree import DecisionTreeClassifier
    classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
    classifier.fit(X, y)
    y_pred = classifier.predict(X)
    Dtree_score = accuracy_score(y, y_pred)
	#print("Dtree_score",Dtree_score)


    #RandomForestClassifier
    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
    classifier.fit(X, y)
    y_pred = classifier.predict(X)
    RF_score = accuracy_score(y, y_pred)
	#print("RandomForestClassifier_score",RF_score)

	
    #svm
    from sklearn.svm import SVC
    classifier = SVC(kernel = 'rbf', random_state = 10)
    classifier.fit(X, y)
    y_pred = classifier.predict(X)
    svm_score = accuracy_score(y, y_pred)
	#print("SVC_score",svm_score)
		
    
    #naive_bayes
    from sklearn.naive_bayes import GaussianNB
    classifier = GaussianNB()
    classifier.fit(X, y)
    y_pred = classifier.predict(X)
    nb_score = accuracy_score(y, y_pred)
	#print("naive_bayes_score",nb_score)

    print("Bar plot generating")
	
		

    import matplotlib.pyplot as plt; 
    plt.rcdefaults()
    import numpy as np
    import matplotlib.pyplot as plt
 
    objects = ('Logistic','knn','svm','RF','Dtree','svc','nb')
    y_pos = np.arange(len(objects))
    performance = [LogisticRegression_score,knn_score,svm_score,RF_score,Dtree_score,svc_score,nb_score]
 
    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('accuracy')
    plt.title('ML alogrithms')
 
    plt.show()

