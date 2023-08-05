# Ingesta
import numpy as np
import pandas as pd
import scipy.stats as stats

# Preprocesamiento
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler

# Visualización
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msngo

# Modelación
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import VotingRegressor

# Métricas de evaluación
from sklearn.metrics import mean_absolute_error, r2_score

# Otros
import pickle
import warnings

# Observacion de nulos
from funciones import print_column_info # Imprime metodo .info
from funciones import estudio_nulos
from funciones import porcentaje_nulos_en_df
from funciones import nulos_en_variable # 
from funciones import imputar_nulos

# Premodelacion
from funciones import probar_modelos # modela, predice e imprime resultados MAE y R2

def preproceso_nulos(nombre_archivo:str):
    """Preproceso nulos
    [Resumen]: Ingresa nombre del archivo sin el .csv para aplicar pre proceso definido
    """
    # importación de base de datos
    data = pd.read_csv(f'{nombre_archivo}.csv', delimiter=';')

    # tranformación de tipo de dato de 'str' a 'float'
    data['rating'] = data['rating'].str.replace(',', '.').astype(float)

    # recodificación de 'Slice of Life'
    data['genre'] = data['genre'].str.replace('Sclice of Life', 'Slice of Life')

    # Recodificación de 'Unknown' y transformación del tipo de dato
    data['episodes'] = np.where(data['episodes'] == 'Unknown', np.nan, data['episodes'])
    data['episodes'] = data['episodes'].astype(float)

    data_tmp, imputer_fit = imputar_nulos(data.drop('rating', axis=1))
    data_imputado = pd.concat([data_tmp, data['rating']], axis=1)
    del data_tmp


    # Guardar matrices
    # matriz que contiene datos nulos en el V.O.
    data_test_nulos = data_imputado[data_imputado['rating'].isna()]
    data_test_nulos.to_csv('anime_test_nulos.csv', index=False)

    #eliminar nulos del data train
    data_clean = data_imputado.dropna()
    data_clean.to_csv('anime_clean.csv', index=False)

#Estandarizacion

def standard_outlier_dummies_gen(data, dict_pipe, dummies=False):
    """
    [resumen]: Se ingresa un dataframe y un escalador y retorna el dataframe escalado.
                A columnas categoricas ponerles None
    [data]: Dataframe
    [escalador]: Escalador de la librería sklearn.preprocessing.
    [dict_pipe]: Diccionario key=columna : Value=proceso
    [return]: Dataframe escalado de acuerdo a argumento ingresado.
    """

    tmp_data = pd.DataFrame()
    # obtener columnas desde el dict y crear un df

    for key, val in dict_pipe.items():
        escalador = val[0]
        if data[key].dtypes == 'O':
            tmp_data[key] = pd.DataFrame(data[key])

        elif escalador == None:
            tmp_data[key] = pd.DataFrame(data[key])

        elif escalador == np.log:
            tmp_data[key] = pd.DataFrame(data[key].apply(lambda x: np.log(x +0.001)))
        
        elif escalador == np.sqrt:
            tmp_data[key] = pd.DataFrame(data[key].apply(lambda x: np.sqrt(x + 0.001)))

        elif str(escalador) == str(StandardScaler()):
            tmp_data[key] = pd.DataFrame(escalador.fit_transform(pd.DataFrame(data[key])))

        elif str(escalador) == str(RobustScaler()):
            tmp_data[key] = pd.DataFrame(escalador.fit_transform(pd.DataFrame(data[key])))

#drop outliers
    drop_outliers = []
    for key, val in dict_pipe.items():
        if val[1]:
            ic_min_tmp = tmp_data[key].mean() - 3*tmp_data[key].std()
            ic_max_tmp = tmp_data[key].mean() + 3*tmp_data[key].std()
            tmp_drop = tmp_data[(tmp_data[key] < ic_min_tmp) | (tmp_data[key] > ic_max_tmp)] #.sort_values(ascending=False)
            drop_outliers = list(set(drop_outliers + list(tmp_drop.index.values)))  
    print(f'{np.round((len(drop_outliers)/tmp_data.shape[0])*100,4)} % Outliers eliminados')
    tmp_data = tmp_data.drop(index=drop_outliers).reset_index(drop=True)

# OHE y concatenación 
    if dummies == True:   
        tmp_dummies = pd.get_dummies(tmp_data)
        for col in tmp_data.columns:
            if tmp_data[col].dtypes != 'O':
                tmp_dummies[col] = tmp_data[col]
        tmp_data = tmp_dummies.copy()
        
    return tmp_data


