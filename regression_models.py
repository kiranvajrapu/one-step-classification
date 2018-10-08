def get_models_score_regression(X, y):
    from sklearn import metrics
    from sklearn.metrics import r2_score
    import matplotlib.pyplot as plt

    print("Regression started ")
    # Fitting Simple Linear Regression to the Training set
    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()
    regressor.fit(X, y)
    y_pred = regressor.predict(X)
    LinearRegression_score = r2_score(y, y_pred)
    print("LinearRegression score :", LinearRegression_score)

    # Fitting Decision Tree Regression to the dataset
    from sklearn.tree import DecisionTreeRegressor
    regressor = DecisionTreeRegressor(random_state=0)
    regressor.fit(X, y)
    y_pred = regressor.predict(X)
    Dtree_score = r2_score(y, y_pred)
    print("DecisionTree score", Dtree_score)
    # Fitting Random Forest Regression to the dataset
    from sklearn.ensemble import RandomForestRegressor
    regressor = RandomForestRegressor(n_estimators=10, random_state=0)
    regressor.fit(X, y)
    y_pred = regressor.predict(X)
    RF_score = r2_score(y, y_pred)
    print("RandomForest score", RF_score)

    from sklearn.neighbors import KNeighborsRegressor
    regressor = KNeighborsRegressor(n_neighbors=2)
    regressor.fit(X, y)
    y_pred = regressor.predict(X)
    knn_score = r2_score(y, y_pred)
    print("KNeighbors score", knn_score)

    # Lasso
    regressor = linear_model.LassoCV(cv=5, normalize=True, random_state=10, alphas=[.0005])
    regressor.fit(X, y)
    y_pred = regressor.predict(X)
    Lasso_score = r2_score(y, y_pred)
    print("Lasso score", Lasso_score)

    # Ridge
    regressor = linear_model.Ridge(random_state=10, normalize=True, alpha=.001)
    regressor.fit(X, y)
    y_pred = regressor.predict(X)
    Ridge_score = r2_score(y, y_pred)
    print("Ridge score", Ridge_score)

    # ElasticNet
    from sklearn.linear_model import ElasticNet
    regressor = ElasticNet(random_state=0)
    regressor.fit(X, y)
    y_pred = regressor.predict(X)
    ElasticNet_score = r2_score(y, y_pred)
    print("ElasticNet score", ElasticNet_score)

    print("Bar plot generating")

    import matplotlib.pyplot as plt
    plt.rcdefaults()
    import numpy as np
    import matplotlib.pyplot as plt

    objects = ('Linear', 'RF', 'Dtree', 'Lasso', 'Ridge', 'ElasticNet', 'knn_score')
    y_pos = np.arange(len(objects))
    performance = [LinearRegression_score, RF_score, Dtree_score, Lasso_score, Ridge_score, ElasticNet_score, knn_score]

    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('accuracy')
    plt.title('ML alogrithms')

    plt.show()

