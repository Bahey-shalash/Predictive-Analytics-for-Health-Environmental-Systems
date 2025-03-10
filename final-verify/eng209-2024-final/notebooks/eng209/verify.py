import os
import re
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from configparser import ConfigParser
from scipy.stats import norm
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score, roc_auc_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, RocCurveDisplay, PrecisionRecallDisplay
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit 

#warnings.filterwarnings("ignore", category=DeprecationWarning)
#warnings.filterwarnings("ignore", category=UserWarning)

def check_prediction(y_true, y_pred, log=False):
    reshaped = False
    if y_true.shape != y_pred.shape:
        try:
            stack_msg = [ f'y_pred shape={y_pred.shape}, y_true shape={y_true.shape}' ]
            msg = f'>>>>: Erreur -- le modèle aurait été entrainé avec une transformation différentes de celle proposée :<<<<'
            if isinstance(y_pred, np.ndarray):
                y_pred = y_pred.reshape(y_true.shape)
                reshaped = True
        except Exception as e:
            stack_msg += [ f'{e}' ]
        finally:
            stack = "\n".join(stack_msg)
            msg = f'{msg}\n{"-"*len(msg)}\n{stack}\n\nTentative avec y_pred shape={y_pred.shape} and y_true shape={y_true.shape}.\n{"-"*len(msg)}'
            if log:
                print(msg)
    return y_pred, reshaped

def select_rows(X, index_list):
    if isinstance(X, (np.ndarray, pd.core.indexes.datetimes.DatetimeIndex)):
        return X[index_list,]
    elif isinstance(X, pd.DataFrame):
        return X.iloc[index_list, :]
    elif isinstance(X, pd.Series):
        return X.iloc[index_list]
    else:
        raise ValueError("Unsupported type")

def get_features(model):
    if hasattr(model, 'feature_names_in'):
        feature_in = model.feature_names_in
    else:
        feature_in = model.__dict__.get('feature_names_in_')
    if feature_in is None and isinstance(model,Pipeline) and len(model) > 0:
        feature_in = get_features(model[0])
    return feature_in

class SimulatorModel:
    def __init__(self, *, predict_file):
        self.predict_file = predict_file + '.predict.csv'
        df = pd.read_csv(self.predict_file, index_col=0, nrows=1)
        self.n_features_in_ = len(df.columns) - 2
        self.feature_names_in_ = list[df.columns[:-2]]

    def predict(self, Xt):
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message=".*Could not infer format.*")
            df = pd.read_csv(self.predict_file, index_col=0, parse_dates=True)
        return df.loc[Xt.index].iloc[:,-1].to_numpy()

    def get_data(self):
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message=".*Could not infer format.*")
            df = pd.read_csv(self.predict_file, index_col=0, parse_dates=True)
        return df.iloc[:,:-2],df.iloc[:,-2],df.index

    def score(self, Xt, yt):
        return 0

    def fit(self, Xt, yt=None):
        pass

class ClassifierModel(ClassifierMixin, BaseEstimator, SimulatorModel):
    def __init__(self, *, predict_file):
        SimulatorModel.__init__(self, predict_file=predict_file)
        self.proba_file = predict_file + '.proba.csv'
        try:
            self.classes_ = pd.read_csv(self.predict_file, index_col=0).iloc[:,-2].unique()
        except Exception as e:
            print(e)
            raise(e)

    def predict_proba(self, Xt):
        df = pd.read_csv(self.proba_file, header=None, index_col=False)
        return df.loc[Xt.index].to_numpy()

class RegressorModel(RegressorMixin, BaseEstimator, SimulatorModel):
    def __init__(self, *, predict_file):
        SimulatorModel.__init__(self, predict_file=predict_file)

class PerfStore:
    def __init__(self, cfg):
        self.database = cfg.get('DEFAULT', 'store', fallback='store.db')
        try:
            from IPython import get_ipython
            ip = get_ipython()
            path = '.'
            for key in ['PAPERMILL_INPUT_PATH', '__session__']:
                if key in ip.user_ns:
                    path = ip.user_ns[key]
                    break
            dir_path = os.path.dirname(os.path.realpath(path))
            re_match = re.search('groupe?_?([0-9A-Z]+)', dir_path, re.I)
            if re_match != None:
                self.group_id=re_match.group(1)
            else:
                self.group_id='None'
        except Exception as e:
            raise Exception(f'Validation failed - {e}')
        print(f'Groupe: {self.group_id}')
        try:
            with sqlite3.connect(self.database) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS `results` (
                    Groupe STRING,
                        Projet STRING,
                        Metric STRING,
                        Value  DECIMALE(6,4),
                        CONSTRAINT `uq_in` UNIQUE (`Groupe`,`Projet`,`Metric`)
                )
                ''')
        except sqlite3.OperationalError as e:
            raise Exception(f'Cannot initialize perf store - {e}')

    def save(self, projet, perfs):
        try:
            with sqlite3.connect(self.database) as conn:
                cursor = conn.cursor()
                cursor.executemany('''
                INSERT INTO `results` (`Groupe`, `Projet`, `Metric`, `Value`) VALUES (?, ?, ?, ?)
                ON CONFLICT DO UPDATE SET `Value` = EXCLUDED.`Value`
                ''', [ (self.group_id, projet, k, v) for k,v in perfs.items()])
        except sqlite3.OperationalError as e:
            raise Exception(f'Cannot update store for {projet}:{self.group_id} - {perfs} -> {e}')

class Config(ConfigParser):
    def __init__(self):
        super().__init__(default_section='DEFAULT')
        self.read_dict({ 'DEFAULT': { k:v for k,v in  os.environ.items() if k in {'HOME','PWD','USER'} } })
        self.read(os.environ.get('ENG209_INI', './eng209.ini'))

class ClassifierEval:
    def __init__(self, cfg, project_name):
        self.cfg = cfg
        self.project_name = project_name
        self.store = PerfStore(cfg)

    def confusionMatrix(self, model, Xt, yt, threshold=None, n_splits=5, label=''):
        # model name in titles, prepend "model" if label is short
        if label is None or len(label) < 1:
            model_label = ''
        elif len(label) < 2:
            model_label = "model "+label
        else:
            model_label = label

        if threshold <= 0.50001 and threshold >= 0.49999:
            threshold = None
        if threshold  == None:
            _, ax = plt.subplots(figsize=(5, 5))
            yt_pred=model.predict(Xt)
            cd=ConfusionMatrixDisplay.from_predictions(yt, yt_pred, ax=ax, labels=model.classes_)
            cd.ax_.set(title=f"Matrice de confusion {model_label}")
        else:
            _, ax = plt.subplots(1,2,figsize=(11, 5))
            yt_pred=model.predict(Xt)
            cd=ConfusionMatrixDisplay.from_predictions(yt, yt_pred, ax=ax[0], labels=model.classes_)
            cd.ax_.set(title=f"Matrice de confusion {model_label}")
            yt_pred_prob=model.predict_proba(Xt)
            yt_pred_thr = (yt_pred_prob[:, 1] > threshold).astype('float')
            cd=ConfusionMatrixDisplay.from_predictions(yt, yt_pred_thr, ax=ax[1], labels=model.classes_)
            cd.ax_.set(title=f"Matrice de confusion {model_label} seuil={threshold:.2f}")
        #cd.plot()
        cv = StratifiedKFold(n_splits=n_splits)
        fold_fnr=[]
        fold_fpr=[]
        for fold, (train, test) in enumerate(cv.split(Xt, yt)):
            Xt_test = select_rows(Xt, test)
            yt_test = select_rows(yt, test)
            if threshold == None:
                yt_pred_thr_test = model.predict(Xt_test)
            else:
                yt_pred_prob_test=model.predict_proba(Xt_test)
                yt_pred_thr_test = (yt_pred_prob_test[:, 1] > threshold).astype('float')
            fold_cm=confusion_matrix(yt_test,yt_pred_thr_test,labels=model.classes_)
            fold_fnr+=[fold_cm[1][0]/(fold_cm[1][0]+fold_cm[1][1])]
            fold_fpr+=[fold_cm[0][1]/(fold_cm[0][0]+fold_cm[0][1])]
        avg_fnr = np.mean(fold_fnr)
        avg_fpr = np.mean(fold_fpr)
        std_fnr = np.std(fold_fnr)
        std_fpr = np.std(fold_fpr)
        postfix = "."+label if len(label) > 0 else ''
        self.store.save(self.project_name, {
            f'cm_avg_fnr{postfix}': avg_fnr,
            f'cm_std_fnr{postfix}': std_fnr,
            f'cm_avg_fpr{postfix}': avg_fpr,
            f'cm_std_fpr{postfix}': std_fpr
        })
        return cd, (avg_fnr, std_fnr, avg_fpr, std_fpr)

    def rocCurve(self, model, Xt, yt, n_splits=5, label='', figax=None, displayFolds=False, style={}):
        if label is None or len(label) < 1:
            model_label = ''
        elif len(label) < 2:
            model_label = "model "+label
        else:
            model_label = label
        cv = StratifiedKFold(n_splits=n_splits)
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, int(yt.shape[0]/n_splits))
        pos_label=1

        if figax is None:
            fig, ax = plt.subplots(figsize=(5, 5))
        else:
            (fig, ax) = figax
        for fold, (train, test) in enumerate(cv.split(Xt, yt)):
            Xt_test = select_rows(Xt, test)
            yt_test = select_rows(yt, test)
            if displayFolds:
                viz = RocCurveDisplay.from_estimator(
                    model,
                    Xt_test,
                    yt_test,
                    name=f"ROC {model_label} fold {fold}",
                    alpha=0.3,
                    lw=1,
                    ax=ax,
                    pos_label=pos_label,
                    plot_chance_level=False,
                )
                sorted_idx=np.argsort(viz.fpr)
                interp_tpr = np.interp(x=mean_fpr, xp=viz.fpr[sorted_idx], fp=viz.tpr[sorted_idx])
                #interp_tpr[0] = 0.5
                tprs.append(interp_tpr)
                aucs.append(viz.roc_auc)
            else:
                yt_pred = model.predict_proba(Xt_test)[:,pos_label]
                fpr, tpr, _ = roc_curve(yt_test, yt_pred)
                sorted_idx=np.argsort(fpr)
                roc_auc = roc_auc_score(yt_test, yt_pred)
                interp_tpr = np.interp(x=mean_fpr, xp=fpr[sorted_idx], fp=tpr[sorted_idx])
                #interp_tpr[0] = 0.5
                tprs.append(interp_tpr)
                aucs.append(roc_auc)

        mean_tpr = np.mean(tprs, axis=0)
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        ax.plot(
            mean_fpr,
            mean_tpr,
            label=r"ROC moyenne %s (AUC = %0.3f $\pm$ %0.3f, %d-folds)" % (model_label, mean_auc, std_auc, n_splits),
            alpha=0.8,
            lw=3,
            color=style.get('line','b'),
        )
        viz = RocCurveDisplay.from_estimator(
            model,
            Xt,
            yt,
            name=f"ROC",
            alpha=0.3,
            lw=2,
            color=style.get('line','b'),
            ax=ax,
            pos_label=pos_label,
            plot_chance_level=(figax is None),
        )

        #yt_pred_prob=model.predict_proba(Xt)
        #f,t,thr=roc_curve(yt,yt_pred_prob[:,1])
        #ax.plot(f,t,'r-')

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(
            mean_fpr,
            tprs_lower,
            tprs_upper,
            color=style.get('fill','grey'),
            alpha=0.2,
            label=r"$\pm$ 1 std. dev.",
        )

        ax.set(
            xlabel="False Positive Rate",
            ylabel="True Positive Rate",
            title=f"ROC moyenne {model_label} avec k={n_splits} folds",
        )
        #ax.legend(loc="lower right")
        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
        ax.grid(alpha=0.2, visible=True)

        postfix = "."+label if len(label) > 0 else ''
        self.store.save(self.project_name, {
            f'roc_avg_auc{postfix}': mean_auc,
            f'roc_std_auc{postfix}': std_auc,
        })

        return (fig, ax), (viz.roc_auc, mean_auc, std_auc)

    def precisionRecallCurve(self, model, Xt, yt, n_splits=5, label='', figax=None, displayFolds=False, style={} ):
        if label is None or len(label) < 1:
            model_label = ''
        elif len(label) < 2:
            model_label = "model "+label
        else:
            model_label = label
        cv = StratifiedKFold(n_splits=n_splits)
        precs = []
        aps = []
        aucs = [] # double check AP
        mean_recall = np.linspace(0, 1, int(yt.shape[0]/n_splits))
        pos_label=1

        if figax is None:
            fig, ax = plt.subplots(figsize=(5, 5))
        else:
            (fig, ax) = figax
        for fold, (train, test) in enumerate(cv.split(Xt, yt)):
            Xt_test = select_rows(Xt, test)
            yt_test = select_rows(yt, test)
            if displayFolds:
                viz = PrecisionRecallDisplay.from_estimator(
                    model,
                    Xt_test,
                    yt_test,
                    name=f"PR {model_label} fold {fold}",
                    alpha=0.3,
                    lw=1,
                    ax=ax,
                    pos_label=pos_label,
                    plot_chance_level=False,
                )
                aps.append(viz.average_precision)
                sorted_idx=np.argsort(viz.recall)
                interp_prec = np.interp(x=mean_recall, xp=viz.recall[sorted_idx], fp=viz.precision[sorted_idx])
                #interp_prec[0] = 1.0
                #interp_prec[-1] = viz.prevalence_pos_label
                precs.append(interp_prec)
                aucs.append(auc(mean_recall, interp_prec))
            else:
                yt_pred = model.predict_proba(Xt_test)[:,pos_label]
                precision, recall, _ = precision_recall_curve(yt_test, yt_pred)
                average_precision = average_precision_score(yt_test, yt_pred)
                aps.append(average_precision)
                sorted_idx=np.argsort(recall)
                interp_prec = np.interp(x=mean_recall, xp=recall[sorted_idx], fp=precision[sorted_idx])
                #interp_prec[0] = 1.0
                #interp_prec[-1] = np.count_nonzero(yt_test==pos_label)/yt.shape[0]
                precs.append(interp_prec)
                aucs.append(auc(mean_recall, interp_prec))

        mean_prec = np.mean(precs, axis=0)
        #mean_prec[0]=1.0
        #mean_ap = np.avg(aps)
        mean_ap = auc(mean_recall, mean_prec)
        std_ap = np.std(aps)

        ax.plot(
            mean_recall,
            mean_prec,
            label=r"Courbe PR moyenne %s (AP = %0.3f $\pm$ %0.3f, %d-folds)" % (model_label, mean_ap, std_ap, n_splits),
            alpha=0.8,
            lw=3,
            color=style.get('line','b'),
        )
        viz = PrecisionRecallDisplay.from_estimator(
            model,
            Xt,
            yt,
            name=f"Courbe PR 1-fold",
            alpha=0.3,
            lw=2,
            color=style.get('line','b'),
            ax=ax,
            pos_label=pos_label,
            plot_chance_level=(figax is None),
        )

        #yt_pred_prob=model.predict_proba(Xt)
        #p,r,thr=precision_recall_curve(yt,yt_pred_prob[:,1])
        #ax.plot(p,r,'r-')
        
        std_prec = np.std(precs, axis=0)
        precs_upper = np.minimum(mean_prec + std_prec, 1)
        precs_lower = np.maximum(mean_prec - std_prec, 0)
        ax.fill_between(
            mean_recall,
            precs_lower,
            precs_upper,
            color=style.get('fill','grey'),
            alpha=0.2,
            label=r"$\pm$ 1 std. dev.",
        )

        ax.set(
            xlabel="Recall",
            ylabel="Précision",
            title=f"Precision-Recall (PR) moyenne avec k={n_splits} folds",
        )
        #ax.legend(loc="lower right")
        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
        ax.grid(alpha=0.2, visible=True)
        postfix = "."+label if len(label) > 0 else ''
        self.store.save(self.project_name, {
            f'avg_prec{postfix}': mean_prec,
            f'std_prec{postfix}': std_prec,
        })
        return (fig, ax), (viz.average_precision, mean_ap, std_ap)

class RegressionEval:
    def __init__(self, cfg, project_name):
        self.cfg = cfg
        self.project_name = project_name
        self.store = PerfStore(cfg)
        self.reshaped = False

    def statistics(self, y_true, y_pred):
        mse=mean_squared_error(y_true, y_pred)
        mae=mean_absolute_error(y_true, y_pred)
        bias = np.mean(y_pred - y_true)
        mu, std = norm.fit(y_pred - y_true)
        return mse, mae, bias, mu, std

    def sideBySide(self, model, Xt, yt, ts, n_splits=5, label='', figax=None, style={}):
        if label is None or len(label) < 1:
            model_label = ''
        elif len(label) < 2:
            model_label = "model "+label
        else:
            model_label = label
        cv = TimeSeriesSplit(n_splits=n_splits)

        ### TODO: hardcoded smaller interval for better visibility
        ts = ts[(ts>='2024-02-23') & (ts <='2024-02-26')]

        sorted_idx=np.argsort(ts)
        Xt_sorted = select_rows(Xt, sorted_idx)
        yt_sorted = select_rows(yt, sorted_idx)
        ts_sorted = select_rows(ts, sorted_idx)
        y_predict = model.predict(Xt_sorted)
        y_predict, self.reshaped = check_prediction(yt_sorted, y_predict, self.reshaped == False)
        if figax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(ts_sorted, yt_sorted, color=style.get("rline","orange"), alpha=0.8, ls='-', label='réelle')
        else:
            (fig, ax) = figax
        ax.plot(ts_sorted, y_predict, color=style.get("line","lightcoral"), alpha=0.6, ls='-',label=f'prédiction {model_label}')
        ax.legend(loc='lower left', bbox_to_anchor=(0, -0), borderaxespad=0.)
        ax.grid(alpha=0.2, visible=True)
        ax.set(
            xlabel="Time",
            ylabel="y",
            title=f'Comparison valeurs de prédiction et données réélles (échantillon)'
        )
        ax.tick_params(axis='x', labelrotation=45)
        postfix = "."+label if len(label) > 0 else ''
        mse, mae, bias, mu, std = self.statistics(yt_sorted, y_predict)
        self.store.save(self.project_name, {
            f'mse{postfix}': mse,
            f'mae{postfix}': mae,
            f'bias{postfix}': bias
        })
        plt.ion()
        return (fig, ax), (mse, mae, bias, mu, std)

    def residues(self, model, Xt, yt, ts, n_splits=5, label='', figax=None, style={}):
        if label is None or len(label) < 1:
            model_label = ''
        elif len(label) < 2:
            model_label = "model "+label
        else:
            model_label = label
        cv = TimeSeriesSplit(n_splits=n_splits)
        sorted_idx=np.argsort(ts)
        Xt_sorted = select_rows(Xt, sorted_idx)
        yt_sorted = select_rows(yt, sorted_idx)
        ts_sorted = select_rows(ts, sorted_idx)
        y_predict = model.predict(Xt_sorted)
        y_predict, self.reshaped = check_prediction(yt_sorted, y_predict, self.reshaped == False)
        if figax is None:
            fig, ax = plt.subplots(1,2,width_ratios=[3,1], figsize=(12, 6))
        else:
            (fig, ax) = figax
        mse, mae, bias, mu, std = self.statistics(yt_sorted, y_predict)
        ax[0].scatter(ts_sorted, y_predict - yt_sorted,
                      color=style.get("line","forestgreen"),
                      alpha=0.8, label=f'$\hat{{y}}-y$ {model_label} MAE={mae:.3f} MSE={mse:.3f} bias={bias:.3f}')
        ax[0].legend(loc='lower left', bbox_to_anchor=(0, 0), borderaxespad=0.)
        ax[0].grid(alpha=0.2, visible=True)
        ax[0].tick_params(axis='x', labelrotation=45)
        ax[0].grid(alpha=0.2, visible=True)
        ax[0].set(
            xlabel="Time",
            ylabel="$\hat{y}-y$",
            title="Residues $\hat{y}-y$"
        )
        ax[1].hist(y_predict - yt_sorted, density=True, color=style.get("fill","lightgreen"))
        xmin, xmax = ax[1].get_xlim()
        x_vals = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x_vals, mu, std)
        ax[1].plot(x_vals, p, lw=2, color=style.get("line","forestgreen"), label=f'Normal Fit\n$\mu={mu:.2f}$, $\sigma={std:.2f}$')
        ax[1].set(
            xlabel="$\hat{y} - y$",
            ylabel="Densité",
            title=f'Densité de probabilité'
        )
        ax[1].grid(alpha=0.2, visible=True)
        fig.tight_layout()
        postfix = "."+label if len(label) > 0 else ''
        self.store.save(self.project_name, {
            f'mse{postfix}': mse,
            f'mae{postfix}': mae,
            f'bias{postfix}': bias
        })
        return (fig, ax), (mse, mae, bias, mu, std)
#
#def verify(model, Xt, yt, adj_thr):
#    cm=confusionMatrix(model, Xt, yt, adj_thr)
#    rocCurve(model, Xt, yt)
#    precRecallCurve(model, Xt, yt)
#    yt_pred_prob=model.predict_proba(Xt)
#    fpr,tpr,thr=roc_curve(yt,yt_pred_prob[:,1])
#    roc_auc=auc(fpr,tpr)
#    fnr=cm[1][0]/(cm[1][0]+cm[1][1])
#    fpr=cm[0][1]/(cm[0][0]+cm[0][1])
#    print(f'ROC AUC: {roc_auc}')
#    print(f'False negative rates: {fnr:.4f}')
#    print(f'False positive rates: {fpr:.4f}')
#    return roc_auc, fnr, fpr
#
#def verify_q1(model, adj_thr, features=None):
#    font_size=plt.rcParams.get('font.size', 10)
#    plt.rcParams['font.size']=10
#    vf=pd.read_csv('/data/q1/verification_species.csv.zip')
#    if features != None:
#        X=vf[features].values
#    else:
#        X=vf.drop(['invasive'],axis=1).values
#    y=vf['invasive'].values
#    roc_auc,fnr,fpr=verify(model, X, y, adj_thr)
#    savePerformances({ 'q1auc': [roc_auc], 'q1Fnr': [fnr], 'q1Fpr': [fpr] }, { 'q1auc': '.3f', 'q1Fnr': '.4f', 'q1Fpr': '.4f' } )
#    plt.rcParams["font.size"]=font_size
#

def log_msg(msg_id, *args):
    log_msgs = {
       'i001': """Info :   Votre modèle nécessite {0} variables, ce qui dépasse vos {1} caractéristiques {3}.
         Nous supposons que le modèle sélecte automatiquement les caractéristiques parmis {2}.
""",
       'i002': """Info :  Même liste de caractéristiques {1} appliquées sur les {0} modèles.""",
       'i003': """Info :  Même seuil {1} appliquées sur les {0} modèles.""",
       'e003': """Erreur : Le paramètre caractéristiques doit être une liste de {1} booléens, ou de {0} noms de colonnes parmi {2}.
""",
       'e004': """Erreur : Si spécifié, seuil doit être une liste de même taille que le nombre de modèles ({0}).
         Chaque élement de seuil doit être un nombre entre 0 et 1 indiquant le seuil de détection du modèle correspondant.
""",
       'e005': """Erreur : Si spécifié, caracteristiques doit être une liste même taille que le nombre de modèles ({0}).
         Chaque élement de caractéristiques doit être une liste de {1} booléens, ou de noms de colonnes parmi {2}.
""",
       'e007': """Erreur : Votre modèle {0} n'est pas entrainé ou il ne respect pas l'interface scikit-learn.""",
       'e008': """Erreur : InputGenerator doit être une fonction ou une liste de fonctions de même taille que le nombre de modèles ({0}).""",
       'e009': """Erreur : Le nombre de variables n={0} de votre InputGenerator est différent du nombre de variables m={1} attendues par votre modèle .""",
       'e010': """Erreur : InputGenerator retourne un nombre d'échantillons différent du nombre attendu k={0}.""",
       'e011': """Erreur : InputGenerator doit retourner une pandas.DataFrame ou numpy.ndarray pour X, et pandas.Series ou numpy.ndarray de 1 colonne pour y et ts. De plus vérifier que X.shape[0]=y.shape[0] et X.shape[0]=ts.shape[0].""",
       'e012': """Erreur : X, y (si pas None) et ts doivent avoir le même nombre de lignes""",
    }
    msg = log_msgs.get(msg_id)
    if msg is None:
        raise Exception(f"""Une erreur c'est produite, veuillez consulter votre professeur. Code: {msg_id}""")
    else:
        if msg_id[0] == 'e':
            raise Exception(msg.format(*args))
        else:
            print(msg.format(*args))
    return

def get_data(fname):
    df = pd.read_csv(fname)
    return df.iloc[:,:-1],df.iloc[:,-1],None

def get_support(n, columns, features):
    support = None
    if features is None:
        support = columns
    elif isinstance(features, list):
        if all(isinstance(e, str) for e in features):
            support = [ c for c in features if c in columns ]
        elif all(isinstance(e, bool) for e in features) and len(columns) == len(features):
            support = [ c[0] for c in zip(columns, features) if c[1] ]
    if len(columns) == n and support is not None and len(support) < n and len(support) > 0:
        log_msg('i001', n, len(support), list(columns), list(support))
        support = columns
    if support is None or len(support) != n:
        log_msg('e003', n, len(columns), list(columns))
    return support

def verify_input_1(num_models, columns, features=None, threshold=None):
    if features is None:
        features = [None] * num_models
    elif not isinstance(features, list):
        features = []
    elif all(isinstance(e, bool) or isinstance(e, str) for e in features):
        log_msg('i002', num_models, features)
        features = [features] * num_models
    elif any(not isinstance(e, list) or len(e) < 1 for e in features):
        features = []
    if threshold is None:
        threshold = [0.5] * num_models
    elif isinstance(threshold, float):
        log_msg('i003', num_models, threshold)
        threshold = [threshold] * num_models
    elif not isinstance(threshold, list) or any(not isinstance(s, float) or s < 0.0 or s > 1.0 for s in threshold):
        threshold = []
    if len(threshold) != num_models:
        log_msg('e004', num_models)
    if len(features) != num_models:
        log_msg('e005', num_models, len(columns), list(columns))
    return (features, threshold)

def verify_input_2(num_models, inputGenerator):
    if inputGenerator is None:
        log_msg('e008', num_models)
    elif callable(inputGenerator):
        return [inputGenerator] * num_models
    elif isinstance(inputGenerator,list) and all(callable(e) for e in inputGenerator) and num_models == len(inputGenerator):
        return inputGenerator
    log_msg('e008', num_models)

def verify_q1(*models, caracteristiques=None, seuil=None):
    try:
        stack_msg=[]
        font_size=plt.rcParams.get('font.size', 10)
        plt.rcParams['font.size']=10
        cfg=Config()
        validator = ClassifierEval(cfg, 1)
        xx_test, yy_test, _ = get_data(cfg.get('DEFAULT', option='hold_out_data_1', fallback='./projet_1_data.csv'))
        (features, threshold) = verify_input_1(len(models), xx_test.columns, caracteristiques, seuil)
        try:
            import cloudpickle
            file = cfg.get('DEFAULT', option='baseline_1__model')
            with read(file, 'rb') as fd:
                ref_model = cloudpickle.load(fd)
            ref_threshold = cfg.getfloat('DEFAULT', option='baseline_1__threshold', fallback=0.5)
            models+=ref_model,
            features+=list(xx_test.columns),
            threshold+=ref_threshold,
        except:
            pass
        try:
            file = cfg.get('DEFAULT', option='baseline_1__save')
            ref_threshold = cfg.getfloat('DEFAULT', option='baseline_1__threshold', fallback=0.5)
            models+=ClassifierModel(predict_file=file),
            features+=list(xx_test.columns),
            threshold+=ref_threshold,
        except:
            pass
        figax_roc = None
        figax_pr = None
        fill = [ 'lightgreen', 'lightblue', 'mistyrose' ]
        line = [ 'forestgreen', 'royalblue', 'lightcoral' ]
        n = len(fill)
        i = 0
        for (model,f,t) in zip(models, features, threshold):
            stack_msg=[]
            if any(not hasattr(model, f) for f in ['fit','predict','n_features_in_']):
                log_msg('e007', type(model))
            support = get_support(model.n_features_in_, xx_test.columns, f)
            if hasattr(model, 'get_data'):
                label = 'baseline'
                XX_test, YY_test, _ = model.get_data()
            else:
                label = str(i+1)
                XX_test = xx_test[support]
                YY_test = yy_test
            stack_msg+=[f'Caractéristiques X {list(XX_test.columns)}']
            feature_in = get_features(model)

            if feature_in is None:
                XX_test = XX_test.to_numpy()
                YY_test = YY_test.to_numpy()
            else:
                stack_msg+=[f'Caractéristiques du model {feature_in}']

            try:
                score = model.score(XX_test, YY_test)
            except ValueError as e:
                msg = f'>>>>: Erreur -- un réarrangement des caractéristiques X est peut-être nécessaire :<<<<'
                stack = "\n".join(stack_msg)
                msg = f'{msg}\n{"-"*len(msg)}\n{e}\n{stack}\n\nTentative avec caractéristiques X réarrangées.\n{"-"*len(msg)}'
                print(msg)
                XX_test = XX_test[feature_in]
                score = model.score(XX_test, YY_test)

            if cfg.getboolean('DEFAULT', option='roc', fallback=False):
                figax_roc, perfs = validator.rocCurve(model, Xt=XX_test, yt=YY_test,
                            figax=figax_roc,
                            label=label,
                            style={ 'fill': fill[i%n], 'line': line[i%n] },
                        )
                print(f'Model {label}:  ROC AUC {perfs[1]:0.4f} std.dev={perfs[2]:0.4f}')
            if cfg.getboolean('DEFAULT', option='precision_recall', fallback=False):
                figax_pr, perfs = validator.precisionRecallCurve(model, Xt=XX_test, yt=YY_test,
                            figax=figax_pr,
                            label=label,
                            style={ 'fill': fill[i%n], 'line': line[i%n] },
                        )
                print(f'Model {label}:  Average Précision {perfs[1]:0.4f} std.dev={perfs[2]:0.4f}')
            if cfg.getboolean('DEFAULT', option='confusion_matrix', fallback=False):
                _, perfs = validator.confusionMatrix(model, Xt=XX_test, yt=YY_test, threshold=t, label=label)
                print(f'Model {label}:  FNR {perfs[0]:0.4f} std.dev={perfs[1]:0.4f}, FPR {perfs[2]:0.4f} std.dev={perfs[3]:0.4f}')

            i=i+1

        plt.show()
    except Exception as e:
        stack = "\n".join(stack_msg)
        msg = f'>>>>: Erreur -- les performances du model ne peuvent pas être correctement mesurées :<<<<'
        msg = f'{msg}\n{"-"*len(msg)}\n{e}\n{stack}\n{"-"*len(msg)}'
        if cfg.get('DEFAULT', option='on_error', fallback='raise') == 'raise':
           raise Exception(msg)
        else:
            print(msg)
    finally:
        plt.rcParams["font.size"]=font_size

def verify_q2(*models, inputGenerator=None):
    try:
        stack_msg=[]
        font_size=plt.rcParams.get('font.size', 10)
        plt.rcParams['font.size']=10
        cfg = Config()
        validator = RegressionEval(cfg, 2)
        test_data = cfg.get('DEFAULT', option='hold_out_data_2', fallback='./projet_2_data.csv')
        inputGeneratorList = verify_input_2(len(models), inputGenerator)

        try:
            import cloudpickle
            file = cfg.get('DEFAULT', option='baseline_2__model')
            with read(file, 'rb') as fd:
                ref_model = cloudpickle.load(fd)
            file = cfg.get_list('DEFAULT', option='baseline_2__generator')
            with read(file, 'rb') as fd:
                ref_generator = cloudpickle.load(fd)
            models+=ref_model,
            inputGeneratorList+=ref_generator,
        except:
            pass
        try:
            file = cfg.get('DEFAULT', option='baseline_2__save')
            model=RegressorModel(predict_file=file)
            models += model,
            inputGeneratorList+=None,
        except Exception as e:
            print(e)
            pass

        figax_res = None
        figax_s2s = None
        fill = [ 'lightgreen', 'lightblue', 'mistyrose' ]
        line = [ 'forestgreen', 'royalblue', 'lightcoral' ]
        n = len(fill)
        i = 0
        for (model,g) in zip(models, inputGeneratorList):
            stack_msg=[]
            if any(not hasattr(model, f) for f in ['fit','predict','n_features_in_']):
                log_msg('e007', type(model))
            if hasattr(model, 'get_data'):
                label = 'baseline'
                xx_test, yy_test, ts = model.get_data()
            else:
                label = str(i+1)
                (xx_test, yy_test, ts) = g(test_data)
            if isinstance(yy_test, list):
                yy_test=np.array(yy_test)
            if isinstance(ts, list):
                ts=np.array(ts)
            if (not isinstance(xx_test, pd.DataFrame)
                and not isinstance(xx_test, np.ndarray) or len(xx_test.shape) != 2):
                log_msg('e011', type(xx_test))
            if (not isinstance(yy_test, pd.Series)
                and not isinstance(yy_test, np.ndarray)
                or len(yy_test.shape) != 1):
                log_msg('e011')
            if xx_test.shape[0] != ts.shape[0] or xx_test.shape[0] != yy_test.shape[0]:
                log.msg('e012')
            if not isinstance(ts, (pd.Series, np.ndarray, pd.core.indexes.datetimes.DatetimeIndex)) or len(ts.shape) != 1:
                log_msg('e011', type(ts))
            if xx_test.shape[1] != model.n_features_in_:
                log_msg('e009', xx_test.shape[1], model.n_features_in_)

            if isinstance(xx_test, pd.DataFrame):
                stack_msg+=[f'Caractéristiques X {list(xx_test.columns)}']

            feature_in = get_features(model)
            if feature_in is None:
                if isinstance(xx_test, pd.DataFrame):
                    XX_test = xx_test.to_numpy()
                if isinstance(yy_test, pd.Series):
                    YY_test = yy_test.to_numpy()
            else:
                stack_msg+=[f'Caractéristiques du model {feature_in}']
                XX_test = xx_test
                YY_test = yy_test
            try:
                score = model.score(XX_test, YY_test)
            except ValueError as e:
                msg = f'>>>>: Erreur -- un réarrangement des caractéristiques X est peut-être nécessaire :<<<<'
                stack = "\n".join(stack_msg)
                msg = f'{msg}\n{"-"*len(msg)}\n{e}\n{stack}\n\nTentative avec caractéristiques X réarrangées.\n{"-"*len(msg)}'
                print(msg)
                XX_test = XX_test[feature_in]
                score = model.score(XX_test, YY_test)

            if cfg.getboolean('DEFAULT', option='residues', fallback=False):
                figax_res, perfs = validator.residues(model, Xt=XX_test, yt=YY_test, ts=ts,
                            figax=figax_res,
                            label=label,
                            style={ 'fill': fill[i%n], 'line': line[i%n] },
                        )
                print(f'Model {label}:  MSE {perfs[0]:0.3f}, MAE {perfs[1]:0.3f}, Error Bias {perfs[2]:0.3f}')
            if cfg.getboolean('DEFAULT', option='side_by_side', fallback=False):
                figax_s2s, perfs = validator.sideBySide(model, Xt=XX_test, yt=YY_test, ts=ts,
                            figax=figax_s2s,
                            label=label,
                            style={ 'fill': fill[i%n], 'line': line[i%n] },
                        )
            i=i+1
        plt.show()

    except Exception as e:
        stack = "\n".join(stack_msg)
        msg = f'>>>>: Erreur -- les performances du model ne peuvent pas être correctement mesurées :<<<<'
        msg = f'{msg}\n{"-"*len(msg)}\n{e}\n{stack}\n{"-"*len(msg)}'
        if cfg.get('DEFAULT', option='on_error', fallback='raise') == 'raise':
           raise Exception(msg)
        else:
            print(msg)
    finally:
        plt.rcParams["font.size"]=font_size
