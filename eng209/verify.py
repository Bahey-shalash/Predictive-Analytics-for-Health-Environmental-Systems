import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def log_msg(msg_id, *args):
    log_msgs = {
        "i001": """Info :   Votre modèle nécessite {0} variables, ce qui dépasse vos {1} caractéristiques {3}.
         Nous supposons que le modèle sélecte automatiquement les caractéristiques parmis {2}.
""",
        "i002": """Info :  Même liste de caractéristiques {1} appliquées sur les {0} modèles.""",
        "i003": """Info :  Même seuil {1} appliquées sur les {0} modèles.""",
        "e003": """Erreur : Le paramètre caractéristiques doit être une liste de {1} booléens, ou de {0} noms de colonnes parmi {2}.
""",
        "e004": """Erreur : Si spécifié, seuil doit être une liste de même taille que le nombre de modèles ({0}).
         Chaque élement de seuil doit être un nombre entre 0 et 1 indiquant le seuil de détection du modèle correspondant.
""",
        "e005": """Erreur : Si spécifié, caracteristiques doit être une liste même taille que le nombre de modèles ({0}).
         Chaque élement de caractéristiques doit être une liste de {1} booléens, ou de noms de colonnes parmi {2}.
""",
        "e007": """Erreur : Votre modèle {0} n'est pas entrainé ou il ne respect pas l'interface scikit-learn.""",
        "e008": """Erreur : InputGenerator doit être une liste de fonctions de même taille que le nombre de modèles ({0}).""",
        "e009": """Erreur : Le nombre de variables n={0} de votre InputGenerator est différent du nombre de variables m={1} attendues par votre modèle .""",
        "e010": """Erreur : InputGenerator retourne un nombre d'échantillons différent du nombre attendu k={0}.""",
    }
    msg = log_msgs.get(msg_id)
    if msg is None:
        print(
            f"""Une erreur c'est produite, veuillez consulter votre professeur. Code: {msg_id}"""
        )
    else:
        print(msg.format(*args))
    return


def get_data(fname):
    df = pd.read_csv(fname)
    return df.iloc[:100, :-1], df.iloc[:100, -1]


def get_support(n, columns, features):
    support = None
    if features is None:
        support = columns
    elif isinstance(features, list):
        if all(isinstance(e, str) for e in features):
            support = [c for c in features if c in columns]
        elif all(isinstance(e, bool) for e in features) and len(columns) == len(
            features
        ):
            support = [c[0] for c in zip(columns, features) if c[1]]
    if (
        len(columns) == n
        and support is not None
        and len(support) < n
        and len(support) > 0
    ):
        log_msg("i001", n, len(support), list(columns), list(support))
        support = columns
    if support is None or len(support) != n:
        log_msg("e003", n, len(columns), list(columns))
        return None
    return support


def verify_input_1(num_models, columns, features=None, threshold=None):
    if features is None:
        features = [None] * num_models
    elif not isinstance(features, list):
        features = []
    elif all(isinstance(e, bool) or isinstance(e, str) for e in features):
        log_msg("i002", num_models, features)
        features = [features] * num_models
    elif any(not isinstance(e, list) or len(e) < 1 for e in features):
        features = []
    if threshold is None:
        threshold = [0.5] * num_models
    elif isinstance(threshold, float):
        log_msg("i003", num_models, threshold)
        threshold = [threshold] * num_models
    elif not isinstance(threshold, list) or any(
        not isinstance(s, float) or s < 0.0 or s > 1.0 for s in threshold
    ):
        threshold = []
    if len(threshold) != num_models:
        log_msg("e004", num_models)
        return (None, None)
    if len(features) != num_models:
        log_msg("e005", num_models, len(columns), list(columns))
        return (None, None)
    return (features, threshold)


def verify_input_2(num_models, inputGenerator):
    if inputGenerator is None:
        return [lambda f: pd.read_csv(f, index_col=0, parse_dates=True)] * num_models
    elif callable(inputGenerator):
        return [inputGenerator] * num_models
    elif (
        isinstance(inputGenerator, list)
        and all(callable(e) for e in inputGenerator)
        and len(models) == len(inputGenerator)
    ):
        return inputGenerator
    else:
        log_msg("e008", num_models)
    return None


def verify_q1(*models, caracteristiques=None, seuil=None):
    print("Vérification ...")
    try:
        xx_test, yy_test = get_data("./projet_1_data.csv")
        (features, threshold) = verify_input_1(
            len(models), xx_test.columns, caracteristiques, seuil
        )
        if features is None:
            return
        for model, f, t in zip(models, features, threshold):
            if any(not hasattr(model, f) for f in ["fit", "predict", "n_features_in_"]):
                log_msg("e007", type(model))
                return
            support = get_support(model.n_features_in_, xx_test.columns, f)
            if support is None:
                return
            XX_test = xx_test[support]
            print(XX_test.columns)
            if model.__dict__.get("feature_names_in_") is None:
                XX_test = XX_test.to_numpy()
                YY_test = yy_test.to_numpy()
            else:
                YY_test = yy_test
            score = model.score(XX_test, YY_test)
            print(f"Vérifiable {score}")
    except Exception as e:
        print(f"Erreur : votre model ne peut pas être validé - {e}")


def verify_q2(*models, inputGenerator=None):
    print("Vérification ...")
    try:
        test_data = "./projet_2_hold_out.csv"
        inputGeneratorList = verify_input_2(len(models), inputGenerator)
        for model, g in zip(models, inputGeneratorList):
            if any(not hasattr(model, f) for f in ["fit", "predict", "n_features_in_"]):
                log_msg("e007", type(model))
                return
            (xx_test, yy_test, ts) = g(test_data)
            if len(xx_test.columns) != model.n_features_in_:
                log_msg("e009", len(xx_test.columns), model.n_features_in_)
                return
            if model.__dict__.get("feature_names_in_") is None:
                XX_test = xx_test.to_numpy()
                YY_test = yy_test.to_numpy()
            else:
                XX_test = xx_test
                YY_test = yy_test
            if YY_test is not None:
                score = model.score(XX_test, YY_test)
            print(f"Vérifiable")
    except Exception as e:
        print(f"Erreur : votre model ne peut pas être validé - {e}")
