
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


# Load dataset
def read():
    """_summary_
    """
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
    names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    dataset = read_csv(url, names=names)   
    return(dataset)

#build training and data test
def look_splits():
    """_summary_
    """
    # Split-out validation dataset
    array = read().values
    X = array[:,0:4]
    y = array[:,4]
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1, shuffle=True)
    return(X_train, X_validation, Y_train, Y_validation)


#build resgression model
def regression_log():
    """_summary_
    """
    # Spot Check Algorithms
    models = []
    models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))

    # evaluate each model in turn
    names = []
    X_train, X_validation, Y_train, Y_validation = look_splits()
    for name, model in models:
        kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
        cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
        names.append(name)
        print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

#build svm model
def svm():
    """_summary_
    """
    # Spot Check Algorithms
    models = []
    models.append(('SVM', SVC(gamma='auto')))

    # evaluate each model in turn
    names = []
    X_train, X_validation, Y_train, Y_validation = look_splits()
    for name, model in models:
        kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
        cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
        
        names.append(name)
        print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

#build tree model
def tree():
    """_summary_
    """
    # Spot Check Algorithms
    models = []
    models.append(('CART', DecisionTreeClassifier()))

    # evaluate each model in turn
    results = []
    names = []
    X_train, X_validation, Y_train, Y_validation = look_splits()
    for name, model in models:
        kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
        cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
        results.append(cv_results)
        names.append(name)
        print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

#main fonction
def main():
    """_summary_
    """
    
    dataS = read()
    print(dataS)
    regression_log()
    tree()
    svm()
    
if __name__ == '__main__':
    main()