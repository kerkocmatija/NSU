# imports
from sklearn.ensemble import RandomForestRegressor, BaggingClassifier, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR, SVC
from hyperopt import hp, tpe, rand, fmin, Trials, space_eval
from hyperopt import pyll, base
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
#######################################################################################################################

# PRIPRAVA PODATKOV

dataset = pd.read_csv('podatki.csv', header=1, sep=',')

x, y = dataset.iloc[:, :-1], dataset.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=123)


# Konstruiramo gnezdeni prostor "ročno" (saj bomo iskali optimalne anstavitve parametrov)
gnezdeni_prostor = {
    "algo": hp.choice('algo', [
        {
            'ime': 'drevo',
            'max_depth': hp.choice('max_depth', [2, 4, 8, 16, 32]),
            'criterion': hp.choice('criterion', ["gini", "entropy"])
        },
        {
            'ime': 'knn',
            'n_neighbors': hp.choice('n_neighbors', [3, 4, 5, 6, 7, 8]),
            'weights': hp.choice('weights', ['uniform', 'distance'])
        },
        {
            'ime': 'gozd',
            'n_estimators': hp.choice('n_estimators', [1, 2, 3, 4, 5]),
            'max_depth': hp.choice('max_depth', [1, 2, 3, 4]),
            'criterion': hp.choice('criterion', ["gini", "entropy"]),
        },
        {
            'ime': 'svm',
            'C': hp.lognormal('C', 0, 1),
            'kernel': hp.choice('kernel', [
                {
                    'tip': 'linear'  # linearno
                },
                {
                    'tip': 'rbf',  # radialno
                    'gamma': hp.lognormal('gamma', 0, 1)
                },
                {
                    'tip': 'poly',  # polinomsko
                    'degree': hp.choice('degree', [1, 2, 3, 4, 5])
                }
            ]),
        },
    ])
}

# Analogno konstruiramo še prostor za vrečenje dreves
prostor_vrecenje = {
    "algo": hp.choice('algo', [
        {
            'ime': 'vrecenje',
            'n_estimators': hp.choice('n_estimators', [1, 2, 3, 4]),
            'max_samples': hp.choice('max_samples', [1, 2, 3, 4, 5]),
            'max_features': hp.choice('max_features', [1, 2, 3, 4, 5]),
        }
    ]),
}


#######################################################################################################################
# RAČUNANJE NAJBOLJŠIH ALGORITMOV TER NJIHOVIH KONFIGURACIJ
#######################################################################################################################ž
def kriterijska_funkcija(parametri):
    ''' Tukaj v resnici ne delamo nič drugega, kot ročno nastavimo vse parametre, ki smo jih
    zgoraj pripisali posameznemu tipu (drevo, knn, SVM) modela. '''
    a = parametri["algo"]
    ime_algoritma = a["ime"]
    if ime_algoritma == "knn":
        n_neighbors = a["n_neighbors"]
        weights = a["weights"]
        model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)

    elif ime_algoritma == "drevo":
        max_depth = a["max_depth"]
        criterion = a["criterion"]
        model = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion)

    elif ime_algoritma == "gozd":
        n_estimators = parametri["n_estimators"]
        max_depth = parametri["max_depth"]
        criterion = parametri["criterion"]
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, criterion=criterion)

    elif ime_algoritma == "svm":
        # najtezji primer pokrijemo :)
        C = a["C"]
        kernel = a["kernel"]["tip"]
        # gamma in degree moramo definirati v vseh treh primerih: tam, kjer nista vazni, ju damo na 1
        neumna_vrednost = 1
        if kernel == "rbf":
            gamma = a["kernel"]["gamma"]
            degree = neumna_vrednost
        elif kernel == "linear":
            degree = neumna_vrednost
            gamma = neumna_vrednost
        else:
            gamma = neumna_vrednost
            degree = a["kernel"]["degree"]
        model = SVC(kernel=kernel, gamma=gamma, C=C, degree=degree)

    else:
        raise ValueError("Napacne nastavitve!")

    model.fit(x_ucna, y_ucna)
    y_napoved = model.predict(x_testna)
    return 1 - accuracy_score(y_testna, y_napoved)

def kriterijska_funkcija_vrecenje(parametri):
    a = parametri["algo"]
    ime_algoritma = a["ime"]
    if ime_algoritma == "vrecenje":
        n_estimators = a["n_estimators"]
        max_samples = a["max_samples"]
        max_features = a["max_features"]
        model = BaggingClassifier(n_estimators=n_estimators, max_samples=max_samples, max_features=max_features)
    else:
        raise ValueError("Napacne nastavitve!")

    model.fit(x_train, y_train)
    y_napoved = model.predict(model, x_test)
    return 1-model.accuracy_score(y_test, y_napoved)


def poisci_najboljse_parametre(prostor, n_izracunov):
    trials = Trials()
    best = fmin(fn=kriterijska_funkcija,
                space=prostor,
                algo=tpe.suggest,
                max_evals=n_izracunov,
                trials=trials)
    best = space_eval(prostor, best)
    best_value = kriterijska_funkcija(best)
    # vse vrednosti paramtrov in kriterijske funkcije, ki smo jih preizkusili
    xs = [trial["misc"]["vals"] for trial in trials.trials]
    ys = [1-trial["result"]["loss"] for trial in trials.trials]

    print(best, 1-best_value)
    # print(xs)
    # print(ys)
    return best, xs, ys

# poisci_najboljse_parametre(gnezdeni_prostor, 10)

#######################################################################################################################
# VREČENJE IN PRIMERJAVA Z NAŠIMI ALGORITMI
#######################################################################################################################
### TOLE TLE NE DELA OK ####

# algo_best, xs, ys = poisci_najboljse_parametre(gnezndeni_prostor, 100)
# algo_vrecenje, xs, ys = poisci_najboljse_parametre(prostor_vrecenje, 100)

# # definiramo, naučimo ter evalviramo oba modela
# ime_algoritma = algo_best["ime"]
# if ime_algoritma == "knn":
#     n_neighbors = a["n_neighbors"]
#     weights = a["weights"]
#     model_best = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
# elif ime_algoritma == "drevo":
#     max_depth = a["max_depth"]
#     criterion = a["criterion"]
#     model_best = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion)
# elif ime_algoritma == "svm":
#     # najtezji primer pokrijemo :)
#     C = a["C"]
#     kernel = a["kernel"]["tip"]
#     # gamma in degree moramo definirati v vseh treh primerih: tam, kjer nista vazni, ju damo na 1
#     neumna_vrednost = 1
#     if kernel == "rbf":
#         gamma = a["kernel"]["gamma"]
#         degree = neumna_vrednost
#     elif kernel == "linear":
#         degree = neumna_vrednost
#         gamma = neumna_vrednost
#     else:
#         gamma = neumna_vrednost
#         degree = a["kernel"]["degree"]
#     model_best = SVC(kernel=kernel, gamma=gamma, C=C, degree=degree)
# else:
#     raise ValueError("Napacne nastavitve!")
#
# #######################################################################################################################
# # IZRIS GRAFOV
# #######################################################################################################################
# xss = [x["n_estimators"] for x in xs]
#
# plt.scatter(xss, ys)
# plt.xlabel("n_estimators")
# plt.show()


