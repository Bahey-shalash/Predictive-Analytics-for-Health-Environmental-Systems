#!/usr/bin/env python
# coding: utf-8

# ---
# ### ENG 209 - Rapport Final
# 
# **Groupe** : Veuillez indiquer le nom de votre groupe
# 
# **À rendre avant le 19.12.2024 minuit.**
# 
# Liste des fichiers à inclure dans votre archive .zip pour la soumission :
# 
# - final_projet_1.ipynb
# - final_projet_2.ipynb
# - rapport.ipynb (ce rapport complété et dûment rempli)

# ---
# #### Projet 1
# 
# Veuillez détailler votre méthodologie pour résoudre le problème proposé dans le projet 1. Pour structurer votre argumentation, vous pouvez aborder les éléments suivants :
# 
# - Type de modèles utilisés : Identifiez les modèles testés et justifiez votre choix final, en expliquant pourquoi ce modèle est adapté au problème.
# - Exploration des données : Décrivez vos analyses exploratoires, en mettant en évidence les tendances, anomalies, ou facteurs importants observés.
# - Réduction des caractéristiques : Précisez les méthodes employées pour sélectionner ou réduire les caractéristiques d'entrées, ainsi que celles finalement retenues.
# - Description des modèles : Détaillez vos modèles, y compris les étapes de transformation des données d'entrée et les hyperparamètres principaux.
# - Validation et comparaison des modèles : Décrivez votre méthode d'évaluation, les métriques utilisées, les performances obtenues, et les comparaisons effectuées entre les différents modèles testés.
# - Justification des paramètres significatifs : Si applicable, expliquez clairement vos choix de paramètres ou hyperparamètres et leur impact sur la performance.
# 
# Structurez vos réponses de manière brève avec rigueur et clarté, en justifiant vos choix à chaque étape pour démontrer une démarche méthodique et réfléchie.

# # Rapport du Projet 1 : Contrôle de la propagation d'une maladie contagieuse
# 
# ## Contexte et Objectif
# 
# Le projet consiste à prédire si une personne est infectée (y=1) ou non (y=0) à partir de cinq variables explicatives (x1 à x5). L’objectif est de réduire le nombre de variables collectées tout en maintenant une sensibilité (rappel) suffisante pour que les faux négatifs ne dépassent pas 10% des individus infectés, afin de limiter la propagation de la maladie. Simultanément, nous cherchons à minimiser les faux positifs pour réduire les coûts de contrôle.
# 
# ## Analyse Exploratoire
# 
# - Les données montrent un déséquilibre de la variable cible (environ 20% d’infectés).
# - Les corrélations linéaires directes entre x1, x2, x3, x4, x5 et y sont faibles, mais cela ne signifie pas absence de relations prédictives.
# - Une analyse par pairplots et la matrice de corrélation n’a pas révélé de liens linéaires marqués, justifiant d’utiliser des modèles capables de détecter des relations plus complexes.
# 
# ## Sélection des Caractéristiques
# 
# Une Forêt Aléatoire initiale a permis d’estimer l’importance des variables. Les variables x1 et x2 sont apparues comme les plus prédictives. Après plusieurs tests, réduire le modèle à x1 et x2 n’a pas dégradé les performances de manière significative. Au contraire, les résultats sont restés de bonne qualité, justifiant la simplification en ne conservant que x1 et x2.
# 
# ## Modèles Utilisés
# 
# 1. **Régression Logistique (RL)** :
#    - Initialement entraînée sans pondération, la RL montrait une faible capacité à détecter les individus infectés.
#    - Pour l’améliorer, nous avons utilisé `class_weight='balanced'` et recherché le meilleur paramètre C via `GridSearchCV`.
#    - Nous avons également testé des transformations telles que le `StandardScaler` et, dans certains cas, des `PolynomialFeatures`.
# 
# 2. **Forêt Aléatoire (Random Forest, RF)** :
#    - Modèle d’ensemble performant dès le départ.
#    - Recherche d’hyperparamètres (nombre d’arbres, profondeur) pour optimiser la performance.
#    - L’utilisation de seulement x1 et x2 n’a pas diminué la performance de façon notable.
# 
# ## Validation des Performances et Ajustement du Seuil
# 
# - **Métriques utilisées** : Matrice de confusion, Précision, Rappel, F1-score, AUC (ROC).
# - La Forêt Aléatoire a présenté une AUC ~0.95 sur le jeu de test, détectant beaucoup plus d’infectés que la Régression Logistique initiale.
# - Après amélioration, la RL a gagné en sensibilité, mais reste moins performante que la RF.
# - Pour répondre à la contrainte FNR ≤ 10%, nous avons ajusté le seuil de décision en testant une grille de seuils afin d’atteindre une sensibilité ≥ 90%. La RF, même réduite à x1 et x2, a pu satisfaire cette contrainte.
# - L’ajustement du seuil a augmenté le nombre de faux positifs, mais permet de rater moins de 10% des personnes infectées, objectif essentiel.
# 
# ## Comparaison et Choix Final
# 
# - **Forêt Aléatoire** : Haute capacité discriminante, AUC élevée, bonne sensibilité après ajustement du seuil, et permet de conserver seulement x1 et x2, réduisant la complexité et les coûts de mesure.
# - **Régression Logistique Améliorée** : Performances améliorées grâce au rééquilibrage de classes et à la recherche d’hyperparamètres, mais reste inférieure à la RF.
# 
# Le modèle choisi est donc la Forêt Aléatoire, avec uniquement x1 et x2, et un seuil de détection adapté pour maintenir FNR ≤ 10%. Cette solution est plus efficace pour prévenir la propagation (moins de faux négatifs) tout en réduisant la collecte de données (seulement deux variables).
# 
# ---

# ---
# #### Projet 2
# 
# Veuillez détailler votre méthodologie pour résoudre le problème proposé dans le projet 2. Pour structurer votre argumentation, vous pouvez aborder les éléments suivants :
# 
# - Type de modèle utilisé : Identifiez le modèle testé et justifiez votre choix final, en expliquant pourquoi ce modèle est adapté au problème.
# - Exploration des données : Décrivez vos analyses exploratoires, en mettant en évidence les tendances, anomalies, ou facteurs importants observés.
# - Préparation des caractéristiques : Expliquez les transformations appliquées aux données brutes, notamment l'encodage, le traitement des valeurs manquantes, création de nouvelles caractéristiques à partir des caractéristiques existantes, ou les interactions entre variables.
# - Description du modèle : Détaillez le modèle final, y compris les étapes de transformation des données d'entrée et les hyperparamètres principaux.
# - Validation du modèle : Décrivez votre méthode d'évaluation, les métriques utilisées, les performances obtenues.
# - Justification des paramètres significatifs : Si applicable, expliquez clairement vos choix de paramètres ou hyperparamètres et leur impact sur la performance.
# 
# Structurez vos réponses de manière brève avec rigueur et clarté, en justifiant vos choix à chaque étape pour démontrer une démarche méthodique et réfléchie.

# #### Rapport - Projet 2
# 
# **Type de modèle utilisé :**  
# Le modèle final retenu est un **RandomForestRegressor**. Ce choix est justifié par la capacité des forêts aléatoires à capturer des relations non linéaires et complexes entre les variables explicatives (x1, x2, et les caractéristiques temporelles) et la variable cible (y). De plus, ce modèle ne nécessite pas de fortes hypothèses sur la forme de la relation entre les variables et est généralement robuste sur des données hétérogènes et bruitées.
# 
# **Exploration des données :**  
# L’analyse exploratoire a consisté à visualiser les séries temporelles de y, x1 et x2. Cette étape a permis de constater des variations régulières au cours de la journée, suggérant un caractère cyclique. Aucun élément particulier (comme des anomalies extrêmes) n’a été souligné au-delà de la tendance générale. Les variations de y semblent liées aux changements de x1 et x2 ainsi qu’à l’heure de la journée et au jour de la semaine.
# 
# Après avoir visualisé individuellement y, x1 et x2 en fonction du temps, nous avons constaté des motifs cycliques :
# 	•	Pour y, un schéma récurrent semble se répéter chaque semaine, suggérant une influence hebdomadaire.
# 	•	Pour x1, la forme observée est proche d’une sinusoïde avec une période d’environ 4 semaines, indiquant une tendance plus longue, sur l’échelle du mois.
# 	•	Pour x2, la série semble suivre une sinusoïde avec une période hebdomadaire, cohérente avec une variabilité plus rapide que x1.
# 
# Ces observations confirment l’intérêt d’intégrer des variables temporelles (heure, jour de la semaine) et de modéliser des lags, afin de permettre au modèle de capturer ces effets cycliques.
# 
# **Préparation des caractéristiques :**  
# Afin d’exploiter les motifs temporels, des caractéristiques liées au temps ont été créées à partir du timestamp (heure, jour de la semaine). Pour mieux modéliser la cyclicité journalière, des transformations sinusoïdales (sin_hour, cos_hour) ont été utilisées. Des décalages (lags) de 15 et 30 minutes sur x1 et x2 ont également été introduits pour prendre en compte l’influence retardée des variables explicatives sur y. Les données manquantes résultant des lags ont été supprimées. Ainsi, le jeu de données final intègre des informations temporelles et des interactions retardées entre x1, x2 et y.
# 
# **Description du modèle :**  
# Le modèle final est un RandomForestRegressor avec les hyperparamètres par défaut (n_estimators=100, random_state=42). Les données brutes sont d’abord transformées via la fonction `inputTransform` pour générer les nouvelles caractéristiques (heure, weekday, sin_hour, cos_hour, x1_lag15, x1_lag30, x2_lag15, x2_lag30) puis le modèle est entraîné sur la partie historique des données (avant le 20 février 2024).
# 
# **Validation du modèle :**  
# Les données ont été scindées chronologiquement :  
# - Entraînement : Avant le 20 février 2024  
# - Test : Après le 20 février 2024 (48 points)
# 
# Cette séparation assure que l’évaluation du modèle se fasse sur des données postérieures non utilisées pour l’entraînement. Les métriques utilisées incluent R², MSE, RMSE, MAE, et MAPE, permettant d’évaluer la précision des prédictions de manière exhaustive.
# 
# **Performances obtenues sur le jeu de test :**  
# - R² ≈ 0.9541  
# - MSE ≈ 0.0545  
# - RMSE ≈ 0.2335  
# - MAE ≈ 0.1881  
# - MAPE ≈ 3.14%
# 
# Ces résultats indiquent que le modèle explique environ 95% de la variance des données de test. L’erreur moyenne absolue est faible, tout comme l’erreur relative (environ 3%), suggérant un bon niveau de précision.
# 
# **Justification des paramètres significatifs :**  
# Les hyperparamètres par défaut du RandomForestRegressor n’ont pas été optimisés plus avant, mais ce modèle, par sa nature, offre une bonne robustesse. Le choix des lags (15 et 30 minutes) s’appuie sur la consigne du problème qui indiquait un décalage temporel potentiel jusqu’à 30 minutes. Les transformations sinusoïdales pour l’heure du jour sont justifiées par la nature cyclique des activités anthropiques mentionnées, permettant au modèle de mieux capturer ces motifs récurrents.
# 
# En conclusion, la méthodologie (exploration, feature engineering, séparation temporelle, choix du modèle) a permis d’obtenir un modèle prédictif performant, confirmant la pertinence des choix effectués.  
