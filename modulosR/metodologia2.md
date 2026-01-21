# FRAMEWORK COMPLETO DE PREPROCESAMIENTO DE DATOS

## Guía Metodológica para Análisis de Machine Learning (ML)

---

## 📋 FASE 0: COMPRENSIÓN DEL PROBLEMA

| Etapa                          | Función/Método Común (Python)                              | Librería Típica            |
| :----------------------------- | :--------------------------------------------------------- | :------------------------- |
| **Análisis Inicial**           | Definición del objetivo, métricas.                         | -                          |
| **Exploración Preliminar**     |                                                            |                            |
| Carga y visualización inicial  | `pd.read_csv()`, `df.head()`, `df.info()`, `df.describe()` | **Pandas**                 |
| Dimensionalidad                | `df.shape`, `df.memory_usage()`                            | **Pandas**                 |
| Tipos de datos                 | `df.dtypes`, `df.astype()`                                 | **Pandas**                 |
| Distribución variable objetivo | `df['target'].value_counts()`, `df['target'].plot.hist()`  | **Pandas**, **Matplotlib** |

---

## 📊 FASE 1: ANÁLISIS EXPLORATORIO DE DATOS (EDA)

| Etapa                                   | Función/Método Común (Python)                                                      | Librería Típica             |
| :-------------------------------------- | :--------------------------------------------------------------------------------- | :-------------------------- |
| **1.1 Calidad de Datos**                |                                                                                    |                             |
| Valores duplicados                      | `df.duplicated()`, `df.drop_duplicates()`                                          | **Pandas**                  |
| Valores faltantes                       | `df.isnull().sum()`, `msno.matrix()`, `msno.heatmap()`                             | **Pandas**, **Missingno**   |
| Valores atípicos                        | Fórmula IQR/Z-score (NumPy), `IsolationForest`                                     | **NumPy**, **Scikit-learn** |
| **1.2 Análisis Univariado**             |                                                                                    |                             |
| Variables numéricas                     | `df['col'].hist()`, `df['col'].plot.box()`, `df['col'].skew()`, `df['col'].kurt()` | **Pandas**, **Seaborn**     |
| Variables categóricas                   | `df['col'].value_counts()`, `df['col'].nunique()`                                  | **Pandas**                  |
| **1.3 Análisis Bivariado/Multivariado** |                                                                                    |                             |
| Correlaciones                           | `df.corr(method='pearson')`, `sns.heatmap()`                                       | **Pandas**, **Seaborn**     |
| Relación con target                     | `sns.scatterplot()`, `sns.boxplot()`, `pd.crosstab()`, `chi2_contingency`          | **Seaborn**, **SciPy**      |
| Multicolinealidad                       | Función para calcular **VIF** (usando `OLS`)                                       | **Statsmodels**             |

---

## 🧹 FASE 2: LIMPIEZA DE DATOS

| Etapa                                  | Función/Método Común (Python)                               | Librería Típica    |
| :------------------------------------- | :---------------------------------------------------------- | :----------------- |
| **2.1 Limpieza Básica**                |                                                             |                    |
| Eliminación de espacios                | `df['col'].str.strip()`, `df['col'].str.replace()`          | **Pandas**         |
| Normalización de texto                 | `df['col'].str.lower()`, `re.sub()` (Regex)                 | **Pandas**, **re** |
| Parseo/Conversión de tipos             | `pd.to_datetime()`, `pd.to_numeric()`, `df['col'].astype()` | **Pandas**         |
| **2.2 Tratamiento de Duplicados**      | `df.drop_duplicates()`                                      | **Pandas**         |
| **2.3 Tratamiento de Inconsistencias** | `df.loc[condicion] = nuevo_valor`, `df['col'].replace()`    | **Pandas**         |

---

## 🔧 FASE 3: MANEJO DE VALORES FALTANTES

| Estrategia                | Función/Método Común (Python)                                                                                            | Librería Típica                    |
| :------------------------ | :----------------------------------------------------------------------------------------------------------------------- | :--------------------------------- |
| **Imputación Numérica**   | `df.fillna(df['col'].mean())`, `SimpleImputer` (media/mediana), `Interpolate()`, `KNNImputer`, `IterativeImputer` (MICE) | **Pandas**, **Scikit-learn**       |
| **Imputación Categórica** | `df.fillna(df['col'].mode()[0])`, `df.fillna('Missing')`                                                                 | **Pandas**, `SimpleImputer` (moda) |
| **Indicadores**           | `df['missing_indicator'] = df['col'].isnull().astype(int)`                                                               | **Pandas**                         |

---

## 🎨 FASE 4: INGENIERÍA DE CARACTERÍSTICAS (FEATURE ENGINEERING)

| Estrategia                      | Función/Método Común (Python)                                                  | Librería Típica             |
| :------------------------------ | :----------------------------------------------------------------------------- | :-------------------------- |
| **Creación desde Existentes**   | Operaciones aritméticas, `df.groupby().agg()`, `pd.cut()`, `pd.qcut()`         | **Pandas**, **NumPy**       |
| **Desde Temporales**            | `df['date_col'].dt.year`, `df['date_col'].dt.dayofweek`, `np.sin()`/`np.cos()` | **Pandas**, **NumPy**       |
| **Transformaciones Normalidad** | `np.log()`, `np.sqrt()`, `PowerTransformer` (Box-Cox/Yeo-Johnson)              | **NumPy**, **Scikit-learn** |
| **Reducir Outliers**            | `df['col'].clip(lower=p5, upper=p95)` (Winsorización/Clipping)                 | **Pandas**                  |
| **Simplificación Categorías**   | `df['col'].map()` (para agrupar raras)                                         | **Pandas**                  |

---

## 🏷️ FASE 5: ENCODING DE VARIABLES CATEGÓRICAS

| Cardinalidad | Función/Método Común (Python)                                                                                           | Librería Típica                   |
| :----------- | :---------------------------------------------------------------------------------------------------------------------- | :-------------------------------- |
| **Baja**     | `pd.get_dummies(drop_first=True)`, `OneHotEncoder`                                                                      | **Pandas**, **Scikit-learn**      |
| **Media**    | `LabelEncoder`, `OrdinalEncoder`                                                                                        | **Scikit-learn**                  |
| **Alta**     | `df['col'].map(df['col'].value_counts(normalize=True))` (Frequency), `TargetEncoder`, `BinaryEncoder`, `HashingEncoder` | **Pandas**, **Category Encoders** |

---

## ⚖️ FASE 6: TRATAMIENTO DE OUTLIERS

| Detección/Tratamiento | Función/Método Común (Python)                                                   | Librería Típica              |
| :-------------------- | :------------------------------------------------------------------------------ | :--------------------------- |
| **Detección**         | `df.quantile()` (IQR), `IsolationForest().fit_predict()`                        | **Pandas**, **Scikit-learn** |
| **Tratamiento**       | `df['col'].clip()`, Eliminación por máscara booleana (`df = df[~outlier_mask]`) | **Pandas**                   |

---

## 🔄 FASE 7: MANEJO DE DESBALANCEO (Clasificación)

| Técnica           | Función/Método Común (Python)                       | Librería Típica      |
| :---------------- | :-------------------------------------------------- | :------------------- |
| **Undersampling** | `RandomUnderSampler`, `TomekLinks`, `NearMiss`      | **Imbalanced-learn** |
| **Oversampling**  | `RandomOverSampler`, `SMOTE`, `ADASYN`              | **Imbalanced-learn** |
| **Híbridos**      | `SMOTETomek`                                        | **Imbalanced-learn** |
| **Alternativa**   | `class_weight='balanced'` (en modelos Scikit-learn) | **Scikit-learn**     |

---

## 📏 FASE 8: ESCALADO Y NORMALIZACIÓN

| Método                               | Función/Método Común (Python)             | Librería Típica  |
| :----------------------------------- | :---------------------------------------- | :--------------- |
| **StandardScaler (Estandarización)** | `StandardScaler().fit_transform(X_train)` | **Scikit-learn** |
| **MinMaxScaler (Normalización)**     | `MinMaxScaler().fit_transform()`          | **Scikit-learn** |
| **RobustScaler**                     | `RobustScaler().fit_transform()`          | **Scikit-learn** |
| **Normalizer (Por muestra)**         | `Normalizer().transform()`                | **Scikit-learn** |

---

## 🔍 FASE 9: SELECCIÓN DE CARACTERÍSTICAS

| Método                        | Función/Método Común (Python)                       | Librería Típica  |
| :---------------------------- | :-------------------------------------------------- | :--------------- |
| **Filter Methods**            | `SelectKBest(score_func=chi2)`, `VarianceThreshold` | **Scikit-learn** |
| **Wrapper Methods**           | `RFE` (Recursive Feature Elimination)               | **Scikit-learn** |
| **Embedded Methods**          | `model.feature_importances_`, `Lasso()`, `Ridge()`  | **Scikit-learn** |
| **Reducción Dimensionalidad** | `PCA().fit_transform()`, `LDA`                      | **Scikit-learn** |

---

## ✂️ FASE 10: DIVISIÓN DE DATOS

| Estrategia             | Función/Método Común (Python)                        | Librería Típica  |
| :--------------------- | :--------------------------------------------------- | :--------------- |
| **Train/Test Split**   | `train_test_split(..., stratify=y, random_state=42)` | **Scikit-learn** |
| **Validación Cruzada** | `KFold`, `StratifiedKFold`, `TimeSeriesSplit`        | **Scikit-learn** |

---

## 🎯 FASE 11: VALIDACIÓN DEL PREPROCESAMIENTO

| Etapa                 | Función/Método Común (Python)                                               | Librería Típica  |
| :-------------------- | :-------------------------------------------------------------------------- | :--------------- |
| **Checks de Calidad** | `X_test.isnull().sum()`, `X_train.shape`, `X_test.shape`                    | **Pandas**       |
| **Modelo Baseline**   | `RandomForestClassifier().fit()`, `LogisticRegression().fit()`              | **Scikit-learn** |
| **Métricas**          | `accuracy_score()`, `f1_score()`, `roc_auc_score()`, `mean_squared_error()` | **Scikit-learn** |

---

## 📦 FASE 12: PIPELINE Y PRODUCCIÓN

| Etapa                    | Función/Método Común (Python)                                         | Librería Típica        |
| :----------------------- | :-------------------------------------------------------------------- | :--------------------- |
| **Creación de Pipeline** | `Pipeline([('step', Transformer or Estimator)])`, `ColumnTransformer` | **Scikit-learn**       |
| **Serialización**        | `joblib.dump(pipeline, 'pipeline.pkl')`, `joblib.load()`              | **Joblib**, **Pickle** |
