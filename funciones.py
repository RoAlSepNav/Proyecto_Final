# Ingesta
import numpy as np
import pandas as pd
import scipy.stats as stats

# Preprocesamiento
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Visualización
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("seaborn-v0_8-whitegrid")
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
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error

# Otros
import pickle
import warnings

#############################################################################################################################################################

def estudio_nulos(data, simbolo):
    ''' 
    [resumen]: Imprime cantidad de datos perdidos en cada columna y devuelve listas,
                todo en base al símbolo proporcionado
    [data]: DataFrame
    [simbolo]: Carácter, string, valor a buscar como nulo o perdido
    [return]: lista_sin_vacios, lista_con_vacios
    '''
    lista_sin_vacios = []
    lista_con_vacios = []
    for i in data.columns:
        try:
            print(f'{i}: {data[i].value_counts()[simbolo]}')
            lista_con_vacios.append(i)
        except:
            lista_sin_vacios.append(i)
    return lista_sin_vacios, lista_con_vacios

#############################################################################################################################################################

# conteo de np.nan
def porcentaje_nulos_en_df(data, lista_nulos):
    ''' 
    Ingresa una data y una lista de columnas
    para imprimir los % de nan por columnas, y el total por líneas       
    '''
    print('Porcentaje de nulos:')
    print('-'*28)
    for col in lista_nulos:
        nulos = data[col].isnull().value_counts('%')
        try:
            nulos = nulos[True]*100
            print(f'{col}: {np.round(nulos, 4)} %')
        except:
            print(f'{col}: No contiene nulos')

    tmp_drop = data.dropna().copy()
    porcion_nulos = (1-(tmp_drop.shape[0]/data.shape[0]))*100
    print('-'*28, f'\nPorcentaje de filas con nulos en el dataset: {np.round(porcion_nulos, 4)} %')

#############################################################################################################################################################

# graficar distribución de nulos
def graficar_nulos(df, variable):
    ''' 
    Ingresa un dataframe y y la variable a graficar     
    '''
    null_counts_genre = df[variable].isna().value_counts()
    plt.figure(figsize=(3, 4))
    plt.bar(['No nulos', 'Nulos'], null_counts_genre, color=['goldenrod', 'red'])
    plt.title(f"Distribución de nulos en {variable}")
    plt.ylabel('Cantidad')

    for i, count in enumerate(null_counts_genre):
        plt.text(i, count, str(count), ha='center', va='bottom')

#############################################################################################################################################################

# ajuste de modelos, predicciones y métricas
def probar_modelos(modelo, X, y, test_size=.3, random_state=10, imprime=True, gen_modelacion=False):
    '''
    [resumen]: Modela, predice e imprime resultados MAE y R2
    [modelo]: Modelo de sklearn
    [X]: Matriz de atributos
    [y]: Vector objetivo
    [test_size]: Tamaño de muestra de pruebas
    [random_state]: Semilla para replicar resultados
    [imprime]: Booleano, si True imprime resultados, False en caso contrario
    [return]: mae, r2
    '''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    modelo_fit = modelo.fit(X_train, y_train)
    y_hat_train = modelo_fit.predict(X_train)
    y_hat = modelo_fit.predict(X_test)

    modelacion = {
        'x_train': X_train,
        'x_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'modelo_fit': modelo_fit,
        'y_hat_train': y_hat_train,
        'y_hat': y_hat,
    }

    mape_train = mean_absolute_percentage_error(y_true=y_train, y_pred=y_hat_train)
    mae_train = mean_absolute_error(y_true=y_train, y_pred=y_hat_train)
    r2_train = r2_score(y_true=y_train, y_pred=y_hat_train)

    mape = mean_absolute_percentage_error(y_true=y_test, y_pred=y_hat)
    mae = mean_absolute_error(y_true=y_test, y_pred=y_hat)
    r2 = r2_score(y_true=y_test, y_pred=y_hat)

    if imprime == True:
        print(f'{modelo}','\n',
            f'Métricas en train:\n',
            f'MAPE train: {mape_train}','\n',
            f'MAE train: {mae_train}','\n',
            f'R2 train: {r2_train}', '\n', '-'*50,'\n'
        
            f'Métricas en test:\n',  
            f'MAPE test: {mape}','\n',
            f'MAE test: {mae}','\n',
            f'R2 test: {r2}', '\n', '-'*50)
        
            

    if gen_modelacion == True:
        return modelacion
    return mape, mae, r2

#############################################################################################################################################################

def imputar_nulos(data, estrategia='most_frequent'):
    """
    [Resumen]: Recibe un dataframe con np.nan e imputa por la estrategia ingresada
    [data]: dataframe con nulos
    [estrategia]: estrategia de imputacion
    [Returns]:
    [df_imp]: Data con np.nan imputados
    [imputer]: objeto entrenado para imputar np.nan
    """
    #obtener dtypes
    tmp_dict = {}
    for col in data.columns:
        tmp_dict[col] = str(data[col].dtype)
    #imputador
    columnas_nulas = data.columns[data.isnull().any()].tolist()
    imputer = SimpleImputer(strategy=estrategia)
    data_imp = data.copy()
    imputer.fit(data[columnas_nulas])
    data_imp[columnas_nulas] = imputer.transform(data[columnas_nulas])
    #cambiar dtypes
    for col in data.columns:
        data_imp[col] = data_imp[col].astype(tmp_dict[col])
    return data_imp, imputer

#############################################################################################################################################################

def graficar_cont(data, lista_variables, figura=(10,30), sep_plot=0.4, cols=1): 
    '''
    Grafica variables continuas mostrando su media y mediana,
    con la curva de tendencia normal.
    '''
    filas = int(np.ceil(len(lista_variables) / cols))
    fig = plt.figure(figsize=figura)  # tamaño de la figura
    fig.subplots_adjust(hspace=sep_plot)  # ajuste de las subplots

    for n, col in enumerate(lista_variables):
        plt.subplot(filas, cols, n+1)
        sns.histplot(data[col], kde=True)
        plt.axvline(data[col].mean(), color='darkorange', linestyle='--', label='Media')
        plt.axvline(data[col].median(), color='green', linestyle='--', label='Mediana')
        plt.axvline(data[col].max(), color='red', linestyle='--', label='Máximo')
        plt.axvline(data[col].min(), color='magenta', linestyle='--', label='Mínimo')
        plt.legend(loc='upper right')
        plt.title(f"Variable {col}")
        plt.ylabel('Observaciones')
        plt.legend()

#############################################################################################################################################################

def print_column_info(data):
    ''' 
    Imprime metodo .info de pandas a partir de un dataframe por columna
    '''
    for column in data.columns:
        non_null_count = data[column].count()
        data_type = data[column].dtype
        print("Column:", column)
        print("Non-null count:", non_null_count)
        print("Data type:", data_type)
        print()

#############################################################################################################################################################

def nulos_en_variable(df:pd.DataFrame, variable:str):
    ''' 
    Ingresa un dataframe y la variable. Imprime la cantidad y porcentaje de nulos en dicha variable    
    '''
    print('Cantidad:\n', df[variable].isna().value_counts(),'\n','-'*30)
    print('Porcentaje:\n', df[variable].isna().value_counts('%').round(4)*100)

#############################################################################################################################################################

def graficar_nulos_total(df: pd.DataFrame):
    '''
    Ingresa un dataframe y grafica los valores nulos de las variables con nulos
    '''
    null_vars = df.columns[df.isnull().any()]  # Obtener las variables con valores nulos

    num_rows = (len(null_vars) + 1) // 2  # Cantidad de filas para los subplots
    fig, axes = plt.subplots(num_rows, 2, figsize=(10, 5 * num_rows))  # Subplots en formato de grilla

    # Graficar valores nulos para las variables con nulos
    for i, var in enumerate(null_vars):
        null_counts = df[var].isna().value_counts()
        ax = axes[i // 2, i % 2] if num_rows > 1 else axes[i % 2]  # Manejo de subplots en la grilla
        ax.bar(['No nulos', 'Nulos'], null_counts, color=['goldenrod', 'red'])
        ax.set_title(f"Distribución de nulos en {var}")
        ax.set_ylabel('Cantidad')

        for j, count in enumerate(null_counts):
            ax.text(j, count, str(count), ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

#############################################################################################################################################################

def dataframe_behaviour(data):
    '''
    [resumen]: Dado un dataframe, grafica el comportamiendo de las variables 'episodes' y 'members'. 
                En la parte superior de manera general, y en la inferior según el tipo de anime.
    [data]: DataFrame
    [return]: None
    '''
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(14, 7))
    sns.boxplot(data=data, x='episodes', ax=axs[0, 0])
    axs[0, 0].set_title('Comportamiento general de episodes')

    sns.boxplot(data=data, x='members', ax=axs[0, 1])
    axs[0, 1].set_title('Comportamiento general de members')

    sns.boxplot(data=data, y='episodes', x='type', ax=axs[1, 0])
    axs[1, 0].set_title('Comportamiento de episodes ordenados por type')

    sns.boxplot(data=data, y='members', x='type', ax=axs[1, 1])
    axs[1, 1].set_title('Comportamiento de members ordenados por type')

    sns.boxplot(data=data, x='rating', ax=axs[0, 2])
    axs[0, 2].set_title('Comportamiento general de rating')

    sns.boxplot(data=data, y='rating', x='type', ax=axs[1, 2])
    axs[1, 2].set_title('Comportamiento de episodes ordenados por type')

    fig.patch.set_alpha(0)
    for ax in axs.flatten():
        ax.set_facecolor('none')

    plt.tight_layout()
    plt.show()

#############################################################################################################################################################

def dataframe_behaviour_univariate(data):
    '''
    [resumen]: Dado un dataframe, grafica el comportamiendo univariado de las variables 'episodes', 'members' y 'rating'. 
    [data]: DataFrame
    [return]: None
    '''
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(14, 4))
    sns.boxplot(data=data, x='episodes', ax=axs[0])
    axs[0].set_title('Comportamiento general de episodes')

    sns.boxplot(data=data, x='members', ax=axs[1])
    axs[1].set_title('Comportamiento general de members')

    sns.boxplot(data=data, x='rating', ax=axs[2])
    axs[2].set_title('Comportamiento general de rating')

    fig.patch.set_alpha(0)
    for ax in axs.flatten():
        ax.set_facecolor('none')

    plt.tight_layout()
    plt.show()

#############################################################################################################################################################

def dataframe_behaviour_multivariate(data):
    '''
    [resumen]: Dado un dataframe, grafica el comportamiendo multivariado de las variables 'episodes',
                'members' y 'rating', ordenados por tipo de anime.
    [data]: DataFrame
    [return]: None
    '''
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 4))
    sns.boxplot(data=data, y='episodes', x='type', ax=axs[0])
    axs[0].set_title('Comportamiento de episodes ordenados por type')

    sns.boxplot(data=data, y='members', x='type', ax=axs[1])
    axs[1].set_title('Comportamiento de members ordenados por type')

    sns.boxplot(data=data, y='rating', x='type', ax=axs[2])
    axs[2].set_title('Comportamiento de episodes ordenados por type')

    fig.patch.set_alpha(0)
    for ax in axs.flatten():
        ax.set_facecolor('none')

    plt.tight_layout()
    plt.show()

#############################################################################################################################################################

def graficar_dispersion(data):
    '''
    [resumen]: Dado un dataframe, grafica la dispersión para las variables Members, Episodes vs Rating.
                En la parte superior Members vs Rating, y en la inferior Episodes vs Rating.
    [data]: DataFrame
    [return]: None
    '''
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5), sharey=True)
    sns.regplot(data=data, x='members', y='rating',
                scatter_kws={'alpha': 0.5, 'color': 'lightblue'},
                line_kws={'alpha': 0.7, 'color': 'green'}, ax=ax[0])
    ax[0].set_xlabel('members', fontsize=14)
    ax[0].set_ylabel('rating', fontsize=14)
    ax[0].set_title(f'Dispersión Members vs Rating', fontsize=16)

    sns.regplot(data=data, x='episodes', y='rating',
                scatter_kws={'alpha': 0.5, 'color': 'lightblue'},
                line_kws={'alpha': 0.7, 'color': 'green'}, ax=ax[1])
    ax[1].set_xlabel('episodes', fontsize=14)
    ax[1].set_ylabel('rating', fontsize=14)
    ax[1].set_title('Dispersión Episodes vs Rating', fontsize=16)

    fig.patch.set_alpha(0)
    ax[0].set_facecolor('none')
    ax[1].set_facecolor('none')

    plt.tight_layout()
    plt.show()

#############################################################################################################################################################

def dataframe_sin_outliers(data):
    '''
    [resumen]: Dado un dataframe, retorna el dataframe con los outliers eliminados para episodes y members.
    [data]: Dataframe
    [return]: Dataframe sin outliers para las variables episodes y members.
    '''
    episodes_ic_min_tmp = data['episodes'].mean() - 3*data['episodes'].std()
    episodes_ic_max_tmp = data['episodes'].mean() + 3*data['episodes'].std()

    members_ic_min_tmp = data['members'].mean() - 3*data['members'].std()
    members_ic_max_tmp = data['members'].mean() + 3*data['members'].std()

    data_ic_episodes = data[(data['episodes'] < episodes_ic_min_tmp) | (data['episodes'] > episodes_ic_max_tmp)].sort_values(by='rating', ascending=False)
    data_ic_members = data[(data['members'] < members_ic_min_tmp) | (data['members'] > members_ic_max_tmp)].sort_values(by='rating', ascending=False)

    index_drop = list(set(list(data_ic_episodes.index.values) + list(data_ic_members.index.values )))
    data_sin_outliers = data.drop(index=index_drop).reset_index(drop=True)

    return data_sin_outliers

#############################################################################################################################################################

def dataframe_scaler(data, escalador, var_a_escalar, var_sin_escalar):
    """
    [resumen]: Se ingresa un dataframe y un escalador y retorna el dataframe escalado.
    [data]: Dataframe
    [escalador]: Escalador de la librería sklearn.preprocessing.
    [return]: Dataframe escalado de acuerdo a argumento ingresado.
    """
    if escalador == np.log:
        df_estandarizado = data.copy()
        for i in var_a_escalar:
            df_estandarizado[i] = df_estandarizado[i].apply(lambda x: np.log(x +0.001))
        return df_estandarizado
    
    elif escalador == np.sqrt:
        df_estandarizado = data.copy()
        for i in var_a_escalar:
            df_estandarizado[i] = df_estandarizado[i].apply(lambda x: np.sqrt(x + 0.001))
        return df_estandarizado

    else:
        data_numerico = data[var_a_escalar].reset_index(drop=True)
        scaler = escalador.fit(data_numerico)
        data_scaled_tmp = pd.DataFrame(scaler.transform(data_numerico), columns=['episodes', 'members'])
        data_scaled = pd.concat([data[var_sin_escalar], data_scaled_tmp], axis=1)
        # Obtener la lista de columnas excepto 'rating'
        cols = [col for col in data_scaled.columns if col != 'rating']
        cols.append('rating')  # Agrega 'rating' al final de la lista de columnas
        # Reorganiza el DataFrame utilizando la nueva lista de columnas
        df_estandarizado = data_scaled.reindex(columns=cols)

        return df_estandarizado, scaler

#############################################################################################################################################################

def mostrar_estandarizacion(data):
    '''
    [resumen]: Dado un dataframe estandarizado, muestra 5 observaciones, forma, columnas y conteo de nulos.
    [data]: Dataframe
    [return]: None
    '''
    display(data.head(3))
    display(data.shape)

#############################################################################################################################################################

def comparacion_correlaciones(data_1, data_2, nombre_df):
    '''
    [resumen]: Dado un dataframe_1 y un dataframe_2, 
                muestra en paralelo el heatmap de correlación de cada uno.
    [data_1]: Dataframe 1
    [data_2]: Dataframe 2
    [nombre_df]: Nombre a colocar en el título general.
    [return]: None
    '''
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.heatmap(data_1.corr(), annot=True, cmap='Blues', ax=axes[0])
    axes[0].set_title('Correlación con outliers', fontweight='bold')

    sns.heatmap(data_2.corr(), annot=True, cmap='Blues', ax=axes[1])
    axes[1].set_title('Correlación sin outliers', fontweight='bold')

    fig.suptitle(f'Comparación de correlaciones\n {nombre_df}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

#############################################################################################################################################################

def graficar_metricas(dict_mae, dict_r2):
    '''
    [resumen]: Se ingresan dos diccionarios y muestra un gráfico de barras con la comparativa entre MAE y R2 para cada modelo.
    [dict_mae]: Diccionario que contiene los valores de MAE para cada modelo
    [dict_r2]: Diccionario que contiene los valores de R2 para cada modelo
    [return]: None
    '''

    labels = list(dict_mae.keys())
    mae_values = list(dict_mae.values())
    r2_values = list(dict_r2.values())

    # Crear un dataframe con los datos
    df_metrics_tmp = pd.DataFrame({'Modelos': labels, 'MAE': mae_values, 'R2': r2_values})
    df_metrics_tmp = df_metrics_tmp.melt('Modelos', var_name='Métricas', value_name='Valor')
    plt.figure(figsize=(30, 5))
    ax = sns.barplot(x='Modelos', y='Valor', hue='Métricas', data=df_metrics_tmp, width=0.6)  # Ajustar el ancho de las barras aquí
    plt.title('Comparación de MAE y R2', fontsize=15, fontweight='bold')
    plt.legend(loc='best')

    # Agregar los valores sobre cada barra
    for val in ax.patches:
        ax.annotate(format(val.get_height(), '.2f'), (val.get_x() + val.get_width() / 2., val.get_height()),
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points')

    ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=11)
    ax.set_xlabel('Modelos', fontsize=12)
    ax.set_ylabel('Valor', fontsize=13)
    plt.show()

#############################################################################################################################################################

def convertir_a_string(lista):
    return ', '.join(map(str, lista))

#############################################################################################################################################################

def graficar_importancia(modelo, lista_col, nro_a_imprimir=10): 
    """
    [resumen]: Se ingresa un modelo y una lista de columnas y muestra un gráfico con las variables más importantes
    [modelo]: Modelo de sklearn
    [lista_col]: Lista de columnas de un dataframe
    [nro_a_imprimir]: Cantidad de variables a mostrar en el gráfico
    [Return]: Índices y nombres de las variables
    """
    importancia = modelo.feature_importances_
    indices = np.argsort(importancia)[::-1]
    indices_10 = indices[0:nro_a_imprimir]
    names = [lista_col[i] for i in indices_10]
    plt.figure(figsize=(7, 6))
    plt.title("Feature importance")
    plt.barh(range(len(names)), importancia[indices_10])
    plt.yticks(range(len(names)), names, rotation=0)
    plt.subplots_adjust(left=0.2)  
    return indices, names

#############################################################################################################################################################

def escalador_simple(X_a_escalar):
    """
    [resumen]: Se ingresa una matriz de datos a escalar
    [X_a_escalar]: Matriz de datos
    [return]: Matriz escalada
    """
    escalador = StandardScaler()
    scaled_data = escalador.fit_transform(X_a_escalar)
    X_escalado = pd.DataFrame(scaled_data, columns=X_a_escalar.columns)
    return X_escalado

#############################################################################################################################################################

def creacion_rangos(modelacion, funcion=None):
    """
    [resumen]:Obtiene los rangos para las predicciones según el modelo ingresado
    [modelacion]: nombre de los resultados obtenidos en la modelación
    [funcion]: Funcion inversa a la cual se utilizo en el preproceso
    [return]: Dataframe con los rangos para las valoraciones
    """
    #revertir preproceso en V.O.
    tmp_dict = modelacion.copy()
    if funcion is not None:
        tmp_dict['y_test'] = funcion(tmp_dict['y_test'])-0.001
        tmp_dict['y_hat'] = funcion(tmp_dict['y_hat'])-0.001

    #Crear Rangos
    result = pd.cut(tmp_dict['y_test'], 
                    bins=3, 
                    labels=["Valoración baja", "Valoración media", "Valoración alta"],
                    include_lowest=True,
                    retbins=True)

    rangos_yhat = pd.DataFrame({'y_hat': tmp_dict['y_hat'], 'y_test': tmp_dict['y_test'], 'rangos':result[0]})
    rangos_yhat['error_abs'] = np.abs(rangos_yhat['y_hat'] - rangos_yhat['y_test'])

    return rangos_yhat

#############################################################################################################################################################

def graficar_error(rangos_y_hat_modelo):
    """
    [resumen]: muestra distintas visualizaciones de gráficos para el rango de las valoraciones
    [rangos_y_hat_modelo]: dataframe que contiene los rangos de las valoraciones
    [Return]: None
    """
    tmp_rangos = rangos_y_hat_modelo.copy().round(4)

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # Displot
    sns.histplot(data=tmp_rangos, x='rangos', y='error_abs', ax=axes[0, 0])
    axes[0, 0].set_title('Concentracion de Errores', fontsize=20)
    axes[0, 0].set_xlabel('Tipo de valoración', fontsize=16)
    axes[0, 0].set_ylabel('Error absoluto', fontsize=14)
    axes[0, 0].tick_params(axis='y', labelsize=12)
    axes[0, 0].tick_params(axis='x', labelsize=12)

    # Barplot
    sns.barplot(data=tmp_rangos, x='rangos', y='error_abs', ax=axes[0, 1])
    axes[0, 1].set_title('Media del Error', fontsize=20)
    axes[0, 1].set_xlabel('Tipo de valoración', fontsize=16)
    axes[0, 1].set_ylabel('Error absoluto', fontsize=14) 
    axes[0, 1].tick_params(axis='y', labelsize=12)
    axes[0, 1].tick_params(axis='x', labelsize=12)

    # Heatmap
    heatmap_data = tmp_rangos.pivot_table(index='y_test', columns='rangos', values='error_abs', aggfunc='mean')
    sns.heatmap(data=heatmap_data, ax=axes[1, 0], cmap='viridis')
    axes[1, 0].set_title('Heatmap del Error', fontsize=20)
    axes[1, 0].set_xlabel('Tipo de valoración', fontsize=16)
    axes[1, 0].set_ylabel('Datos validación', fontsize=14)
    axes[1, 0].tick_params(axis='y', labelsize=12)
    axes[1, 0].tick_params(axis='x', labelsize=12)

    # Scatterplot
    sns.scatterplot(data=tmp_rangos, x='y_test', y='error_abs', hue='rangos', ax=axes[1, 1])
    axes[1, 1].set_title('Dispersión del error', fontsize=20)
    axes[1, 1].set_xlabel('Datos validación', fontsize=16)
    axes[1, 1].set_ylabel('Error absoluto', fontsize=14)
    axes[1, 1].tick_params(axis='y', labelsize=12)
    axes[1, 1].tick_params(axis='x', labelsize=12)

    fig.patch.set_facecolor('none')  # Fondo transparente

    plt.tight_layout(pad=3.0)
    plt.show()

################################################################################################

def resultados_modelacion_sqrt(dict_modelos_finales, graficar=True):
    """
    [resumen]: entrega las métricas y el MAE 'destransformado'
    [dict_modelos_finales]: Diccionario que contenga los modelos finales almacenados
    [return]: Dataframe con los rangos para las valoraciones
    """
    #revertir preproceso en V.O.
    tmp_mape, tmp_mae, tmp_r2 = {}, {}, {}
    for modelo in dict_modelos_finales: 
        tmp_mape[modelo] = mean_absolute_percentage_error(dict_modelos_finales[modelo]['y_test'], dict_modelos_finales[modelo]['y_hat'])

        tmp_mae[modelo] = mean_absolute_error(np.square(dict_modelos_finales[modelo]['y_test'])-0.001,
                                            np.square(dict_modelos_finales[modelo]['y_hat'])-0.001)

        tmp_r2[modelo] = r2_score(dict_modelos_finales[modelo]['y_test'], dict_modelos_finales[modelo]['y_hat'])
    
    metricas = pd.DataFrame({'MAPE': tmp_mape, 'MAE':tmp_mae, 'R2':tmp_r2}).sort_values(by='MAPE', ascending=True)

    if graficar:
        fig, axes = plt.subplots(1, 1, figsize=(12, 7))
        metricas.plot(kind='line', ax=axes)
        axes.tick_params(axis='x', rotation=0, labelsize=16)
        axes.tick_params(axis='y', labelsize=14)
        axes.set_title('Métricas modelos', fontsize=20, pad=20)
        axes.set_xlabel('Modelos', fontsize=16) 
        axes.legend(fontsize=14)

        fig.patch.set_alpha(0)
        axes.set_facecolor('none')

        plt.tight_layout()
        plt.show()

    return metricas

###########################################################################################

def metricas_transformacion_inversa(dict_modelos_finales):
    """
    [resumen]: entrega las métricas y el MAE 'destransformado'
    [dict_modelos_finales]: Diccionario que contenga los modelos finales almacenados
    [return]: Dataframe con los rangos para las valoraciones
    """
    #revertir preproceso en V.O.
    tmp_mape_train, tmp_mae_train, tmp_r2_train = {}, {}, {}
    tmp_mape_test, tmp_mae_test, tmp_r2_test = {}, {}, {}

    for modelo in dict_modelos_finales: 
        tmp_mape_train[modelo] = mean_absolute_percentage_error(dict_modelos_finales[modelo]['y_train'], dict_modelos_finales[modelo]['y_hat_train'])
        tmp_mape_test[modelo] = mean_absolute_percentage_error(dict_modelos_finales[modelo]['y_test'], dict_modelos_finales[modelo]['y_hat'])

        tmp_mae_train[modelo] = mean_absolute_error(np.square(dict_modelos_finales[modelo]['y_train'])-0.001, np.square(dict_modelos_finales[modelo]['y_hat_train'])-0.001)
        tmp_mae_test[modelo] = mean_absolute_error(np.square(dict_modelos_finales[modelo]['y_test'])-0.001, np.square(dict_modelos_finales[modelo]['y_hat'])-0.001)

        tmp_r2_train[modelo] = r2_score(dict_modelos_finales[modelo]['y_train'], dict_modelos_finales[modelo]['y_hat_train'])
        tmp_r2_test[modelo] = r2_score(dict_modelos_finales[modelo]['y_test'], dict_modelos_finales[modelo]['y_hat'])
    
    metricas = pd.DataFrame({'MAPE train': tmp_mape_train, 'MAE train':tmp_mae_train, 'R2 train':tmp_r2_train, 'MAPE test': tmp_mape_test, 'MAE test':tmp_mae_test, 'R2 test':tmp_r2_test}).sort_values(by='MAPE test', ascending=True)

    return metricas

###########################################################################################

def df1_vs_df2(df1, df2):
    ''' compara columnas de dos df
    devuelve lista con fiderencias df1 > df2
    '''
    lista = df1.columns.to_list()
    for i in df2.columns.to_list():
        if i in lista:
            lista.remove(i)
    return lista