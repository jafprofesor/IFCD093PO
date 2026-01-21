📊 Ejercicios Prácticos: Árboles de Decisión y SVM
Aplicación en Datasets Reales - California Housing, Diabetes y Titanic

## Introducción General

Este notebook está diseñado para que aprendas de manera práctica cómo aplicar **Árboles de Decisión** y **Máquinas de Soporte Vectorial (SVM)** en problemas reales de Machine Learning. Usaremos tres datasets famosos: California Housing (para regresión), Diabetes (para clasificación) y Titanic (para clasificación con feature engineering).

**¿Por qué estos modelos?**

- **Árboles de Decisión**: Son fáciles de interpretar, no requieren escalado de datos y manejan tanto variables numéricas como categóricas. Son ideales para principiantes porque puedes visualizar las decisiones como un árbol.
- **SVM**: Son potentes para datos complejos y robustos a outliers, pero requieren escalado y son más difíciles de interpretar. Son útiles cuando la precisión es clave.

**¿Por qué estos datasets?**

- **California Housing**: Problema de regresión para predecir precios de casas. Te ayudará a entender cómo los modelos predicen valores continuos.
- **Diabetes**: Problema de clasificación para predecir si la diabetes está avanzada. Es un dataset numérico limpio, perfecto para comparar modelos.
- **Titanic**: Problema de clasificación para predecir supervivencia. Incluye feature engineering, lo que te enseña a preparar datos reales con valores faltantes y variables categóricas.

**Estructura del Notebook:**

1. Configuración inicial: Importar librerías necesarias.
2. Ejercicio 1: California Housing (Regresión).
3. Ejercicio 2: Diabetes (Clasificación).
4. Ejercicio 3: Titanic (Clasificación con preprocesamiento avanzado).
5. Comparaciones y conclusiones.

**Consejos para Aprender:**

- Lee las explicaciones antes de ejecutar el código.
- Experimenta cambiando parámetros y observa cómo cambian los resultados.
- Pregúntate: ¿Por qué se hace esto? ¿Qué pasa si no lo hago?

---

## 1. Configuración Inicial

Antes de empezar, necesitamos importar todas las librerías que usaremos. Esto es como preparar tus herramientas antes de construir algo.

**Explicación de las librerías:**

- `numpy` y `pandas`: Para manejar datos numéricos y tablas (DataFrames).
- `matplotlib` y `seaborn`: Para crear gráficos y visualizaciones.
- `sklearn`: La librería principal para Machine Learning. Incluye modelos como DecisionTree y SVM, herramientas para dividir datos (train_test_split), optimizar parámetros (GridSearchCV) y calcular métricas.
- `warnings`: Para ignorar advertencias que no son críticas, manteniendo el output limpio.

**¿Por qué usar estas librerías?** Scikit-learn (sklearn) es estándar en ML porque es fácil de usar y tiene implementaciones eficientes de algoritmos. Seaborn hace gráficos bonitos y fáciles de leer.

```python
# Configuración inicial

# Importamos librerías esenciales para datos y visualización
import numpy as np  # Para operaciones numéricas rápidas
import pandas as pd  # Para manejar datos en forma de tablas (DataFrames)
import matplotlib.pyplot as plt  # Para crear gráficos básicos
import seaborn as sns  # Para gráficos más avanzados y atractivos

# Importamos herramientas de scikit-learn para Machine Learning
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score  # Para dividir datos y optimizar modelos
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree  # Modelos de Árboles de Decisión
from sklearn.svm import SVC, SVR  # Modelos de SVM para clasificación (SVC) y regresión (SVR)
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             mean_squared_error, r2_score, precision_score, recall_score, f1_score)  # Métricas para evaluar modelos
from sklearn.preprocessing import StandardScaler, LabelEncoder  # Para escalar datos y codificar categorías
from sklearn.datasets import fetch_california_housing, load_diabetes  # Datasets de ejemplo incluidos en sklearn
from sklearn.ensemble import RandomForestClassifier  # Otro modelo de ensemble (no usado aquí, pero disponible)

# Configuramos el entorno para evitar advertencias innecesarias
import warnings
warnings.filterwarnings('ignore')  # Ignoramos warnings para mantener el output limpio (en producción, revisa estos warnings)

# Configuramos el estilo de los gráficos para que sean más atractivos
plt.style.use('seaborn-v0_8')  # Estilo de seaborn para gráficos
sns.set_palette("husl")  # Paleta de colores variada y legible

# Mensaje de confirmación para saber que todo se cargó bien
print("✅ Librerías cargadas correctamente")
```

**Explicación del código:**

- Cada `import` trae funciones específicas. Por ejemplo, `DecisionTreeRegressor` es para predecir números (regresión), mientras que `DecisionTreeClassifier` es para categorías (clasificación).
- `warnings.filterwarnings('ignore')`: En un entorno de aprendizaje, ignoramos warnings para no distraernos, pero en el mundo real, siempre revísalos porque pueden indicar problemas.
- `plt.style.use` y `sns.set_palette`: Hacen que los gráficos se vean profesionales. ¿Por qué? Porque las visualizaciones ayudan a entender los datos y resultados.

**¿Qué sigue?** Ahora vamos al primer ejercicio: California Housing, donde aplicaremos estos modelos para predecir precios de casas.

---

## 2. Ejercicio 1: California Housing - Predicción de Precios

Este ejercicio se centra en un problema de **regresión**: predecir el precio de las casas en California basado en características como ingresos medios, edad de las casas, etc. Usaremos el dataset incluido en scikit-learn.

**¿Por qué este dataset?** Es un problema real de regresión, ideal para aprender cómo los modelos predicen valores continuos. Te ayudará a entender métricas como RMSE (error promedio) y R² (qué tan bien el modelo explica los datos).

**Pasos generales:**

1. Cargar y explorar los datos.
2. Visualizar relaciones.
3. Preprocesar (dividir y escalar).
4. Entrenar modelos (Árbol y SVM).
5. Evaluar y comparar.

### 2.1 Carga y Exploración de Datos

Primero, cargamos el dataset y lo exploramos para entender su estructura. Esto es crucial porque nos da insights sobre qué variables son importantes.

**Explicación:**

- `fetch_california_housing()`: Carga el dataset directamente de sklearn. Incluye datos de casas en California con 8 características y el precio como objetivo.
- Convertimos el precio a dólares multiplicando por 100,000 (el dataset original está en cientos de miles).
- Usamos `print` para mostrar información: forma (filas y columnas), nombres de columnas, rango de precios, primeras filas y estadísticas descriptivas (media, mediana, etc.).
- **¿Por qué explorar?** Para detectar outliers, distribuciones sesgadas o correlaciones. Por ejemplo, si una variable tiene muchos valores faltantes, tendríamos que manejarlo.

```python
# Cargar dataset California Housing

print("🏠 EJERCICIO 1: CALIFORNIA HOUSING - PREDICCIÓN DE PRECIOS")

# Cargar el dataset desde sklearn (incluye datos reales de casas en California)
california = fetch_california_housing()

# Convertir a DataFrame de pandas para manipulación fácil
df_california = pd.DataFrame(california.data, columns=california.feature_names)

# Añadir la columna objetivo (precio) y convertir a dólares para mayor claridad
df_california['Price'] = california.target * 100000  # El target original está en cientos de miles, lo convertimos a dólares

# Mostrar información básica del dataset
print("📊 INFORMACIÓN DEL DATASET:")
print(f"Forma del dataset: {df_california.shape}")  # Número de filas y columnas
print(f"Características: {list(df_california.columns)}")  # Nombres de las columnas
print(f"Rango de precios: ${df_california['Price'].min():,.0f} - ${df_california['Price'].max():,.0f}")  # Precio mínimo y máximo

# Mostrar las primeras 5 filas para ver ejemplos de datos
print("\n🔍 PRIMERAS 5 FILAS:")
print(df_california.head())

# Análisis estadístico: media, mediana, std, etc., para cada columna
print("\n📈 ESTADÍSTICAS DESCRIPTIVAS:")
print(df_california.describe())
```

**Explicación del código:**

- `fetch_california_housing()`: Proporciona datos limpios y listos para usar. ¿Por qué usarlo? Es estándar y no requiere descarga externa.
- `df_california.describe()`: Muestra estadísticas como media y desviación estándar. **¿Por qué?** Para ver si hay valores extremos (outliers) que podrían afectar el modelo.
- **Consejo:** Observa el rango de precios. Si es muy amplio, el modelo podría tener dificultades; eso es normal en regresión.

**¿Qué esperamos ver?** Características como 'MedInc' (ingreso medio) probablemente correlacionen con el precio. Las estadísticas nos dirán si necesitamos normalizar o manejar outliers.

### 2.2 Análisis Exploratorio

Ahora, visualizamos las relaciones entre las características y el precio para entender patrones. Esto nos ayuda a ver si hay correlaciones lineales o no lineales, y a identificar outliers.

**Explicación:**

- Usamos scatter plots para ver cómo cada característica se relaciona con el precio. `alpha=0.3` hace los puntos semi-transparentes para ver densidades.
- La matriz de correlación muestra coeficientes entre -1 y 1. Valores cercanos a 1 indican correlación positiva fuerte (e.g., más ingresos = precios más altos).
- **¿Por qué visualizar?** Los gráficos revelan insights que las estadísticas no muestran, como relaciones no lineales o clusters de datos.
- **¿Por qué grid y tight_layout?** Para que los gráficos se vean ordenados y no se superpongan.

```python
# Visualización de relaciones

# Crear una figura con subplots (2 filas, 3 columnas) para 6 gráficos
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Lista de características a analizar
caracteristicas = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup']

# Bucle para crear scatter plots para cada característica vs precio
for i, feature in enumerate(caracteristicas):
    ax = axes[i//3, i%3]  # Seleccionar el subplot correcto
    ax.scatter(df_california[feature], df_california['Price'], alpha=0.3)  # Scatter plot con transparencia
    ax.set_xlabel(feature)  # Etiqueta del eje X
    ax.set_ylabel('Precio ($)')  # Etiqueta del eje Y
    ax.set_title(f'{feature} vs Precio')  # Título del gráfico
    ax.grid(True, alpha=0.3)  # Añadir grid para mejor lectura

# Ajustar el layout para que no se superpongan los gráficos
plt.tight_layout()
plt.show()

# Matriz de correlación para ver relaciones entre todas las variables
plt.figure(figsize=(10, 8))
correlation_matrix = df_california.corr()  # Calcular matriz de correlación
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')  # Heatmap con anotaciones
plt.title('Matriz de Correlación - California Housing')
plt.tight_layout()
plt.show()

# Imprimir observaciones iniciales basadas en los gráficos
print("🔍 OBSERVACIONES INICIALES:")
print("- MedInc (Ingreso medio) tiene alta correlación con el precio")
print("- Algunas características tienen correlación baja")
print("- Hay outliers en algunas variables como AveRooms")
```

**Explicación del código:**

- `plt.subplots(2, 3)`: Crea una grid de 6 gráficos. **¿Por qué?** Para comparar todas las características de manera eficiente.
- `ax.scatter`: Dibuja puntos para cada par (característica, precio). `alpha=0.3` evita que se vean densos.
- `sns.heatmap`: Visualiza la matriz de correlación. Colores rojos indican correlación positiva, azules negativa. **¿Por qué?** Para identificar rápidamente qué variables influyen en el precio.
- **Consejo:** Busca correlaciones >0.5 o <-0.5. Si una variable no correlaciona, podría no ser útil para el modelo.

**¿Qué esperamos ver?** 'MedInc' debería tener correlación alta con 'Price'. Outliers en 'AveRooms' podrían indicar datos erróneos o extremos (e.g., casas con muchas habitaciones).

### 2.3 Preprocesamiento

Antes de entrenar los modelos, necesitamos preparar los datos: separarlos en características (X) y objetivo (y), dividir en entrenamiento y prueba, y escalar para SVM.

**Explicación:**

- **Separar X e y:** X son las características (inputs), y es el objetivo (precio). Usamos `drop` para eliminar 'Price' de X.
- **Dividir datos:** Usamos `train_test_split` para crear conjuntos de entrenamiento (80%) y prueba (20%). `random_state=42` asegura reproducibilidad. **¿Por qué dividir?** Para evaluar el modelo en datos no vistos y evitar overfitting.
- **Escalar:** SVM es sensible a escalas, así que usamos `StandardScaler` para normalizar (media=0, std=1). Árboles no lo necesitan, pero lo preparamos para ambos. **¿Por qué escalar para SVM?** Variables con escalas grandes dominan el modelo; escalar las hace comparables.
- `test_size=0.2`: 20% para prueba es estándar. `stratify` no se usa aquí porque es regresión.

```python
# Preparar datos para modelos

# Separar características (X) y objetivo (y)
X_house = df_california.drop('Price', axis=1)  # X: todas las columnas excepto 'Price'
y_house = df_california['Price']  # y: la columna 'Price' (objetivo)

# Dividir en entrenamiento y prueba (80% train, 20% test)
X_house_train, X_house_test, y_house_train, y_house_test = train_test_split(
    X_house, y_house, test_size=0.2, random_state=42  # random_state para reproducibilidad
)

# Escalar características (importante para SVM, no para Árboles)
scaler_house = StandardScaler()  # Inicializar el escalador
X_house_train_scaled = scaler_house.fit_transform(X_house_train)  # Ajustar y transformar train
X_house_test_scaled = scaler_house.transform(X_house_test)  # Solo transformar test (para evitar data leakage)

# Confirmar que los datos están listos
print("✅ DATOS PREPARADOS:")
print(f"Entrenamiento: {X_house_train.shape[0]} muestras")  # Número de filas en train
print(f"Prueba: {X_house_test.shape[0]} muestras")  # Número de filas en test
print(f"Características: {X_house_train.shape[1]}")  # Número de columnas en X
```

**Explicación del código:**

- `drop('Price', axis=1)`: Elimina la columna 'Price' de X. **¿Por qué?** El modelo no debe ver el objetivo durante el entrenamiento.
- `train_test_split`: Divide aleatoriamente pero reproduciblemente. **¿Por qué random_state?** Para que los resultados sean consistentes al reejecutar.
- `StandardScaler`: Transforma datos a media 0 y std 1. `fit_transform` en train aprende la media/std y transforma; `transform` en test usa los mismos valores. **¿Por qué no fit en test?** Para simular datos reales no vistos.
- **Consejo:** Siempre escala después de dividir para evitar "data leakage" (el modelo ve info de test).

**¿Qué sigue?** Ahora entrenamos los modelos: primero el Árbol de Decisión.

### 2.4 Modelo: Árbol de Decisión para Regresión

Entrenamos un Árbol de Decisión para predecir precios. Usamos GridSearchCV para encontrar los mejores parámetros y evaluamos con métricas de regresión.

**Explicación:**

- **DecisionTreeRegressor:** Modelo para regresión. Divide los datos en nodos basados en características para predecir valores continuos.
- **GridSearchCV:** Prueba combinaciones de parámetros para encontrar el mejor modelo. **¿Por qué?** Para optimizar y evitar overfitting. Usamos 'neg_mean_squared_error' porque GridSearch maximiza, pero MSE es minimizado.
- **Parámetros probados:** 'max_depth' limita la profundidad del árbol (evita overfitting); 'min_samples_split' y 'min_samples_leaf' controlan splits (evitan árboles demasiado complejos).
- **Métricas:** RMSE (error promedio en dólares), R² (proporción de varianza explicada, 1 es perfecto).
- **Importancia de características:** Muestra qué variables son más influyentes. **¿Por qué?** Para entender el modelo y posiblemente simplificarlo.

```python
# Árbol de Decisión para regresión

print("🌳 ENTRENANDO ÁRBOL DE DECISIÓN (REGRESIÓN)")

# Definir parámetros a probar para optimización
param_grid_tree = {
    'max_depth': [3, 5, 7, 10, 15],  # Profundidad máxima del árbol (más profundo = más complejo)
    'min_samples_split': [2, 5, 10],  # Mínimo muestras para dividir un nodo
    'min_samples_leaf': [1, 2, 4]  # Mínimo muestras en una hoja
}

# Inicializar el modelo base
tree_reg = DecisionTreeRegressor(random_state=42)  # random_state para reproducibilidad

# Usar GridSearchCV para encontrar los mejores parámetros
grid_tree = GridSearchCV(tree_reg, param_grid_tree, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_tree.fit(X_house_train, y_house_train)  # Entrenar con datos de entrenamiento

# Obtener el mejor modelo encontrado
best_tree = grid_tree.best_estimator_
y_pred_tree = best_tree.predict(X_house_test)  # Predecir en datos de prueba

# Calcular métricas de evaluación
mse_tree = mean_squared_error(y_house_test, y_pred_tree)  # Error cuadrático medio
rmse_tree = np.sqrt(mse_tree)  # Raíz del MSE (en dólares)
r2_tree = r2_score(y_house_test, y_pred_tree)  # Coeficiente de determinación

# Imprimir resultados
print(f"✅ MEJORES PARÁMETROS: {grid_tree.best_params_}")
print(f"📊 RESULTADOS ÁRBOL:")
print(f" RMSE: ${rmse_tree:,.0f}")  # Error promedio en dólares
print(f" R²: {r2_tree:.3f}")  # Proporción de varianza explicada

# Visualizar importancia de características
importancia = pd.DataFrame({
    'caracteristica': X_house.columns,  # Nombres de las características
    'importancia': best_tree.feature_importances_  # Importancia calculada por el modelo
}).sort_values('importancia', ascending=False)  # Ordenar de mayor a menor

plt.figure(figsize=(10, 6))
plt.barh(importancia['caracteristica'], importancia['importancia'])  # Gráfico de barras horizontal
plt.xlabel('Importancia')
plt.title('Importancia de Características - Árbol de Decisión')
plt.gca().invert_yaxis()  # Invertir eje Y para que la más importante esté arriba
plt.grid(True, alpha=0.3, axis='x')  # Grid para mejor lectura
plt.show()
```

**Explicación del código:**

- `GridSearchCV:** Prueba todas las combinaciones de parámetros (e.g., 5x3x3=45 modelos). cv=5 usa validación cruzada. **¿Por qué cv=5?\*\* Para una evaluación robusta sin usar datos de prueba.
- `best_tree.predict:** Usa el modelo optimizado para predecir. **¿Por qué no usar train para predecir?\*\* Para evaluar generalización.
- `mean_squared_error` y `r2_score:\*\* Métricas estándar para regresión. RMSE es interpretable (en unidades del objetivo); R² indica qué tan bien el modelo se ajusta (0=mal, 1=perfecto).
- `feature*importances*:** Árboles calculan importancia basada en cómo reducen el error. **¿Por qué visualizar?\*\* Para ver si el modelo usa las variables esperadas (e.g., 'MedInc' debería ser alta).
- **Consejo:** Si R² es bajo, el modelo no explica bien los datos; prueba más parámetros o feature engineering.

**¿Qué esperamos ver?** Un RMSE bajo (e.g., <50,000) y R² alto (e.g., >0.5). La importancia debería resaltar 'MedInc'.

### 2.5 Modelo: SVM para Regresión

Ahora entrenamos un SVM para regresión. SVM usa un kernel para mapear datos a un espacio de mayor dimensión donde son linealmente separables.

**Explicación:**

- **SVR (Support Vector Regressor):** Versión de SVM para regresión. Encuentra un hiperplano que maximice el margen, pero permite errores dentro de un umbral (epsilon).
- **GridSearchCV:** Optimiza parámetros como C (penalización por errores), gamma (influencia de un punto) y kernel (forma de la función).
- **Parámetros probados:** C alto penaliza errores más; gamma alto hace el modelo más complejo; kernel 'rbf' es no lineal, 'linear' es lineal.
- **Métricas:** Mismas que el Árbol (RMSE, R²). SVM puede ser más preciso pero más lento.
- **Escalado:** Usamos datos escalados porque SVM es sensible a escalas.

```python
# SVM para regresión

print("🎯 ENTRENANDO SVM (REGRESIÓN)")

# Definir parámetros a probar para optimización
param_grid_svm = {
    'C': [0.1, 1, 10, 100],  # Penalización por errores (más alto = menos tolerancia)
    'gamma': ['scale', 'auto', 0.1, 0.01],  # Influencia de un punto (más alto = más complejo)
    'kernel': ['rbf', 'linear']  # Tipo de kernel: 'rbf' para no lineal, 'linear' para lineal
}

# Inicializar el modelo SVR
svm_reg = SVR()

# Usar GridSearchCV para encontrar los mejores parámetros
grid_svm = GridSearchCV(svm_reg, param_grid_svm, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
grid_svm.fit(X_house_train_scaled, y_house_train)  # Entrenar con datos escalados

# Obtener el mejor modelo
best_svm = grid_svm.best_estimator_
y_pred_svm = best_svm.predict(X_house_test_scaled)  # Predecir en datos de prueba escalados

# Calcular métricas
mse_svm = mean_squared_error(y_house_test, y_pred_svm)  # Error cuadrático medio
rmse_svm = np.sqrt(mse_svm)  # Raíz del MSE
r2_svm = r2_score(y_house_test, y_pred_svm)  # R²

# Imprimir resultados
print(f"✅ MEJORES PARÁMETROS: {grid_svm.best_params_}")
print(f"📊 RESULTADOS SVM:")
print(f" RMSE: ${rmse_svm:,.0f}")
print(f" R²: {r2_svm:.3f}")
```

**Explicación del código:**

- `param_grid_svm:\*\* Prueba combinaciones (e.g., 4x4x2=32 modelos). cv=3 es más rápido que cv=5 para SVM, ya que son lentos.
- `SVR():** Usa kernel 'rbf' por defecto. **¿Por qué kernel?\*\* Para capturar relaciones no lineales en los datos.
- `fit con datos escalados:** SVM requiere escalado para funcionar bien. **¿Por qué?\*\* Variables con escalas diferentes afectan el cálculo del margen.
- `predict en test escalado:\*\* Asegura consistencia.
- **Consejo:** Si el modelo es lento, reduce el grid o usa cv=3. Compara RMSE y R² con el Árbol.

**¿Qué esperamos ver?** SVM podría tener RMSE similar o mejor que el Árbol, pero toma más tiempo. R² alto indica buen ajuste.

### 2.6 Comparación y Análisis

Finalmente, comparamos los resultados de ambos modelos para ver cuál es mejor en este dataset.

**Explicación:**

- **DataFrame de resultados:** Creamos una tabla con métricas para comparar fácilmente.
- **Gráficos de barras:** Visualizamos R² (cuanto más alto mejor) y RMSE (cuanto más bajo mejor). Colores diferentes para distinguir modelos.
- **Textos en barras:** Añadimos valores exactos para precisión.
- **Conclusiones:** Basadas en métricas, determinamos el mejor modelo. **¿Por qué comparar?** Para elegir el modelo más adecuado para el problema.

```python
# Comparación de modelos

print("📊 COMPARACIÓN FINAL - CALIFORNIA HOUSING")

# Crear DataFrame con resultados para comparación
resultados_california = pd.DataFrame({
    'Modelo': ['Árbol de Decisión', 'SVM'],
    'RMSE': [rmse_tree, rmse_svm],  # Error promedio en dólares
    'R²': [r2_tree, r2_svm]  # Proporción de varianza explicada
})

print(resultados_california)  # Mostrar tabla de resultados

# Visualización comparativa
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Comparación R² (más alto es mejor)
axes[0].bar(resultados_california['Modelo'], resultados_california['R²'], color=['skyblue', 'lightcoral'])
axes[0].set_ylabel('R² Score')
axes[0].set_title('Comparación: R² Score')
axes[0].grid(True, alpha=0.3)

# Comparación RMSE (más bajo es mejor)
axes[1].bar(resultados_california['Modelo'], resultados_california['RMSE'], color=['skyblue', 'lightcoral'])
axes[1].set_ylabel('RMSE ($)')
axes[1].set_title('Comparación: Error (RMSE)')
axes[1].grid(True, alpha=0.3)

# Añadir valores exactos en las barras
for i, (modelo, r2, rmse) in enumerate(zip(resultados_california['Modelo'],
                                            resultados_california['R²'],
                                            resultados_california['RMSE'])):
    axes[0].text(i, r2 + 0.01, f'{r2:.3f}', ha='center', va='bottom')  # Texto para R²
    axes[1].text(i, rmse + 1000, f'${rmse:,.0f}', ha='center', va='bottom')  # Texto para RMSE

plt.tight_layout()
plt.show()

# Conclusiones basadas en métricas
print("💡 CONCLUSIONES CALIFORNIA HOUSING:")
print(f"• Mejor modelo: {'Árbol de Decisión' if r2_tree > r2_svm else 'SVM'}")
print(f"• El error promedio es de aproximadamente ${min(rmse_tree, rmse_svm):,.0f}")
print(f"• El modelo explica el {max(r2_tree, r2_svm)*100:.1f}% de la variación en los precios")
```

**Explicación del código:**

- `pd.DataFrame:** Crea una tabla para comparar métricas. **¿Por qué?\*\* Facilita la lectura de resultados.
- `plt.subplots(1, 2):** Dos gráficos lado a lado. **¿Por qué?\*\* Para comparar R² y RMSE visualmente.
- `axes[0].bar:\*\* Gráfico de barras para R². Color 'skyblue' para Árbol, 'lightcoral' para SVM.
- `axes[1].bar:** Gráfico para RMSE. **¿Por qué invertir colores?\*\* Para consistencia.
- `text:** Añade valores en las barras. **¿Por qué?\*\* Para ver números exactos sin leer el DataFrame.
- **Conclusiones:** Usa condicionales para determinar el mejor basado en R² (más alto gana). **¿Por qué R²?** Indica qué tan bien el modelo explica los datos.

**¿Qué esperamos ver?** El Árbol podría ser mejor en interpretabilidad, SVM en precisión. El error promedio nos dice cuánto se equivoca el modelo en dólares.

---

## 3. Ejercicio 2: Diabetes - Predicción de Enfermedad

Este ejercicio es un problema de **clasificación**: predecir si la diabetes está avanzada basado en características médicas. Convertimos el dataset de regresión a binario para clasificación.

**¿Por qué este dataset?** Es un dataset numérico limpio de sklearn, ideal para comparar modelos en clasificación. Te enseña a convertir problemas de regresión a clasificación.

**Pasos generales:**

1. Cargar y explorar los datos.
2. Visualizar distribuciones por clase.
3. Preprocesar (dividir y escalar).
4. Entrenar modelos (Árbol y SVM).
5. Evaluar y comparar.

### 3.1 Carga y Exploración de Datos

Cargamos el dataset Diabetes y lo convertimos a un problema binario para clasificación.

**Explicación:**

- `load_diabetes()`: Carga datos de progresión de diabetes. Originalmente para regresión, lo convertimos a binario usando la mediana como umbral.
- **Convertir a binario:** Valores > mediana = 1 (avanzada), <= mediana = 0 (no avanzada). **¿Por qué?** Para hacer un problema de clasificación.
- Mostramos forma, características, distribución de clases, primeras filas y estadísticas.
- **¿Por qué explorar?** Para ver balance de clases y distribuciones.

```python
# Cargar dataset Diabetes

print("\n" + "="*60)
print("🩺 EJERCICIO 2: DIABETES - PREDICCIÓN DE ENFERMEDAD")

# Cargar el dataset desde sklearn (datos de progresión de diabetes)
diabetes = load_diabetes()
df_diabetes = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
df_diabetes['target'] = diabetes.target  # Añadir la columna objetivo (progresión de diabetes)

# Para clasificación, convertimos en problema binario
# Valores altos de target indican mayor progresión de diabetes
umbral = df_diabetes['target'].median()  # Usar mediana como umbral
df_diabetes['diabetes_avanzada'] = (df_diabetes['target'] > umbral).astype(int)  # 1 si avanzada, 0 si no

# Mostrar información del dataset
print("📊 INFORMACIÓN DEL DATASET:")
print(f"Forma del dataset: {df_diabetes.shape}")  # Filas y columnas
print(f"Características: {list(diabetes.feature_names)}")  # Nombres de variables
print(f"Distribución de clases:")
print(df_diabetes['diabetes_avanzada'].value_counts())  # Conteo de 0 y 1

# Mostrar las primeras 5 filas
print("\n🔍 PRIMERAS 5 FILAS:")
print(df_diabetes.head())
```

**Explicación del código:**

- `load_diabetes():` Proporciona datos médicos estandarizados. **¿Por qué usarlo?** Es limpio y no requiere preprocesamiento inicial.
- `umbral = df_diabetes['target'].median():` Mediana asegura balance aproximado de clases. **¿Por qué mediana?** Para evitar sesgo si hay outliers.
- `astype(int):` Convierte booleanos a enteros (0/1).
- **Consejo:** Observa la distribución de clases. Si está desbalanceada, considera técnicas como oversampling.

**¿Qué esperamos ver?** Clases balanceadas (alrededor de 50% cada una). Características como 'bmi' (IMC) podrían correlacionar con diabetes avanzada.

### 3.2 Análisis Exploratorio

Visualizamos las distribuciones de las características por clase para ver diferencias entre diabetes avanzada y no avanzada.

**Explicación:**

- **Boxplots:** Muestran la distribución de cada característica por clase. La caja representa el 50% central, la línea la mediana, y los bigotes los extremos.
- **Matriz de correlación:** Similar a antes, pero ahora incluye la nueva columna 'diabetes_avanzada'.
- **¿Por qué boxplots?** Para ver si hay diferencias en las distribuciones (e.g., BMI más alto en diabetes avanzada).
- **¿Por qué grid?** Para organizar múltiples gráficos.

```python
# Análisis exploratorio diabetes

# Crear figura con subplots para boxplots
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Lista de características a analizar
caracteristicas_diabetes = ['age', 'bmi', 'bp', 's1', 's2', 's3']

# Bucle para crear boxplots por clase
for i, feature in enumerate(caracteristicas_diabetes):
    ax = axes[i//3, i%3]  # Seleccionar subplot
    # Datos para cada clase
    data_to_plot = [df_diabetes[df_diabetes['diabetes_avanzada'] == 0][feature],
                   df_diabetes[df_diabetes['diabetes_avanzada'] == 1][feature]]
    ax.boxplot(data_to_plot, labels=['No Avanzada', 'Avanzada'])  # Boxplot con etiquetas
    ax.set_ylabel(feature)  # Etiqueta del eje Y
    ax.set_title(f'Distribución de {feature} por Clase')  # Título
    ax.grid(True, alpha=0.3)  # Grid para mejor lectura

plt.tight_layout()
plt.show()

# Matriz de correlación para ver relaciones
plt.figure(figsize=(10, 8))
corr_diabetes = df_diabetes.corr()  # Calcular correlación
sns.heatmap(corr_diabetes, annot=True, cmap='coolwarm', center=0, fmt='.2f')  # Heatmap
plt.title('Matriz de Correlación - Diabetes')
plt.tight_layout()
plt.show()
```

**Explicación del código:**

- `plt.subplots(2, 3):` Crea 6 gráficos. **¿Por qué?** Para comparar todas las características.
- `ax.boxplot:** Dibuja cajas para cada clase. **¿Por qué?\*\* Para ver si hay separación entre clases (e.g., BMI más alto en avanzada).
- `sns.heatmap:** Visualiza correlaciones. **¿Por qué?\*\* Para ver qué variables correlacionan con 'diabetes_avanzada'.
- **Consejo:** Busca diferencias en medianas o outliers entre clases. Si no hay separación, el modelo podría tener dificultades.

**¿Qué esperamos ver?** 'bmi' y 'bp' podrían tener distribuciones diferentes por clase. Correlaciones con 'diabetes_avanzada' indican variables útiles.

### 3.3 Preprocesamiento

Preparamos los datos para clasificación: separamos X e y, dividimos en train/test con estratificación, y escalamos para SVM.

**Explicación:**

- **Separar X e y:** X son las características (sin 'target' ni 'diabetes_avanzada'), y es la clase binaria.
- **Dividir datos:** 70% train, 30% test. `stratify=y_diabetes` asegura balance de clases en ambos sets. **¿Por qué estratificar?** Para que train y test tengan la misma proporción de clases.
- **Escalar:** SVM necesita escalado; Árboles no, pero lo preparamos.
- **¿Por qué test_size=0.3?** Más datos para test en clasificación para evaluar mejor.

```python
# Preparar datos para clasificación

# Separar características (X) y objetivo (y)
X_diabetes = df_diabetes.drop(['target', 'diabetes_avanzada'], axis=1)  # X: características originales
y_diabetes = df_diabetes['diabetes_avanzada']  # y: clase binaria

# Dividir en entrenamiento y prueba (70% train, 30% test) con estratificación
X_diab_train, X_diab_test, y_diab_train, y_diab_test = train_test_split(
    X_diabetes, y_diabetes, test_size=0.3, random_state=42, stratify=y_diabetes  # stratify para balance de clases
)

# Escalar características para SVM
scaler_diab = StandardScaler()  # Inicializar escalador
X_diab_train_scaled = scaler_diab.fit_transform(X_diab_train)  # Ajustar y transformar train
X_diab_test_scaled = scaler_diab.transform(X_diab_test)  # Solo transformar test

# Confirmar preparación
print("✅ DATOS DIABETES PREPARADOS:")
print(f"Entrenamiento: {X_diab_train.shape[0]} muestras")  # Número de filas en train
print(f"Prueba: {X_diab_test.shape[0]} muestras")  # Número de filas en test
print(f"Proporción clases entrenamiento: {y_diab_train.value_counts(normalize=True).to_dict()}")  # Balance de clases
```

**Explicación del código:**

- `drop(['target', 'diabetes_avanzada'], axis=1):` Elimina columnas no necesarias. **¿Por qué?** 'target' es el original, 'diabetes_avanzada' es y.
- `stratify=y_diabetes:` Mantiene proporción de clases. **¿Por qué?** Evita que un set tenga solo una clase.
- `StandardScaler:` Normaliza datos. **¿Por qué?** SVM funciona mejor con escalas similares.
- **Consejo:** Verifica el balance de clases. Si desbalanceado, considera técnicas como SMOTE.

**¿Qué sigue?** Entrenamos los modelos de clasificación.

### 3.4 Modelo: Árbol de Decisión para Clasificación

Entrenamos un Árbol de Decisión para clasificar diabetes avanzada. Usamos GridSearchCV para optimizar parámetros y evaluamos con métricas de clasificación.

**Explicación:**

- **DecisionTreeClassifier:** Modelo para clasificación. Usa criterios como 'gini' o 'entropy' para dividir nodos.
- **GridSearchCV:** Prueba parámetros para maximizar exactitud. **¿Por qué?** Para evitar overfitting y mejorar generalización.
- **Parámetros probados:** 'max_depth' limita complejidad; 'min_samples_split' y 'criterion' controlan splits y medida de impureza.
- **Métricas:** Exactitud (porcentaje correcto), Precisión (de positivos predichos, cuántos son reales), Recall (de positivos reales, cuántos predice), F1-Score (balance de precisión y recall).
- **Matriz de confusión:** Muestra verdaderos positivos, falsos positivos, etc. **¿Por qué?** Para ver errores específicos.

```python
# Árbol de Decisión para clasificación

print("🌳 ENTRENANDO ÁRBOL DE DECISIÓN (CLASIFICACIÓN)")

# Definir parámetros a probar
param_grid_tree_clf = {
    'max_depth': [3, 5, 7, 10],  # Profundidad máxima
    'min_samples_split': [2, 5, 10],  # Mínimo muestras para split
    'criterion': ['gini', 'entropy']  # Criterio de división: gini (default) o entropy
}

# Inicializar modelo
tree_clf = DecisionTreeClassifier(random_state=42)  # random_state para reproducibilidad

# Optimizar con GridSearchCV
grid_tree_clf = GridSearchCV(tree_clf, param_grid_tree_clf, cv=5, scoring='accuracy', n_jobs=-1)
grid_tree_clf.fit(X_diab_train, y_diab_train)  # Entrenar

# Mejor modelo y predicciones
best_tree_clf = grid_tree_clf.best_estimator_
y_pred_tree_clf = best_tree_clf.predict(X_diab_test)  # Predecir en test

# Calcular métricas
accuracy_tree = accuracy_score(y_diab_test, y_pred_tree_clf)  # Exactitud
precision_tree = precision_score(y_diab_test, y_pred_tree_clf)  # Precisión
recall_tree = recall_score(y_diab_test, y_pred_tree_clf)  # Recall
f1_tree = f1_score(y_diab_test, y_pred_tree_clf)  # F1-Score

# Imprimir resultados
print(f"✅ MEJORES PARÁMETROS: {grid_tree_clf.best_params_}")
print(f"📊 RESULTADOS ÁRBOL:")
print(f" Exactitud: {accuracy_tree:.3f}")
print(f" Precisión: {precision_tree:.3f}")
print(f" Recall: {recall_tree:.3f}")
print(f" F1-Score: {f1_tree:.3f}")

# Matriz de confusión
cm_tree = confusion_matrix(y_diab_test, y_pred_tree_clf)  # Calcular matriz
plt.figure(figsize=(8, 6))
sns.heatmap(cm_tree, annot=True, fmt='d', cmap='Blues',  # Heatmap con números
            xticklabels=['No Avanzada', 'Avanzada'],
            yticklabels=['No Avanzada', 'Avanzada'])
plt.title('Matriz de Confusión - Árbol de Decisión (Diabetes)')
plt.ylabel('Real')  # Etiqueta Y: valores reales
plt.xlabel('Predicción')  # Etiqueta X: valores predichos
plt.show()
```

**Explicación del código:**

- `param_grid_tree_clf:\*\* Prueba combinaciones (e.g., 4x3x2=24 modelos). cv=5 para validación robusta.
- `DecisionTreeClassifier:** Usa 'gini' por defecto. **¿Por qué criterion?\*\* 'gini' es más rápido; 'entropy' es más informativo.
- `grid_tree_clf.fit:\*\* Entrena con datos no escalados (Árboles no lo necesitan).
- `predict:\*\* Clasifica en 0 o 1.
- `accuracy_score, etc.:** Métricas para clasificación. **¿Por qué múltiples?\*\* Exactitud es general; precisión/recall para clases desbalanceadas.
- `confusion_matrix:** Filas = real, columnas = predicho. **¿Por qué visualizar?\*\* Para ver falsos positivos/negativos.
- **Consejo:** Si recall es bajo, el modelo pierde muchos positivos reales; ajusta parámetros.

**¿Qué esperamos ver?** Exactitud >0.7, matriz con más aciertos en diagonal. 'bmi' podría ser importante.

### 3.5 Modelo: SVM para Clasificación

Entrenamos un SVM para clasificar diabetes avanzada. SVM es efectivo para datos numéricos limpios como este.

**Explicación:**

- **SVC (Support Vector Classifier):** Versión de SVM para clasificación. Encuentra el hiperplano que maximice el margen entre clases.
- **GridSearchCV:** Optimiza parámetros para maximizar exactitud. **¿Por qué?** Para encontrar el mejor modelo.
- **Parámetros probados:** C (penalización por errores), gamma (influencia), kernel (forma de la función).
- **Métricas:** Mismas que el Árbol. SVM podría ser más preciso en datos limpios.
- **Matriz de confusión:** Similar al Árbol, pero con colores diferentes para distinguir.

```python
# SVM para clasificación

print("🎯 ENTRENANDO SVM (CLASIFICACIÓN)")

# Definir parámetros a probar
param_grid_svm_clf = {
    'C': [0.1, 1, 10, 100],  # Penalización por errores (más alto = menos tolerancia)
    'gamma': ['scale', 'auto', 0.1, 0.01],  # Influencia de un punto
    'kernel': ['rbf', 'linear']  # Kernel: 'rbf' para no lineal, 'linear' para lineal
}

# Inicializar modelo
svm_clf = SVC(random_state=42)  # random_state para reproducibilidad

# Optimizar con GridSearchCV
grid_svm_clf = GridSearchCV(svm_clf, param_grid_svm_clf, cv=5, scoring='accuracy', n_jobs=-1)
grid_svm_clf.fit(X_diab_train_scaled, y_diab_train)  # Entrenar con datos escalados

# Mejor modelo y predicciones
best_svm_clf = grid_svm_clf.best_estimator_
y_pred_svm_clf = best_svm_clf.predict(X_diab_test_scaled)  # Predecir en test escalado

# Calcular métricas
accuracy_svm = accuracy_score(y_diab_test, y_pred_svm_clf)  # Exactitud
precision_svm = precision_score(y_diab_test, y_pred_svm_clf)  # Precisión
recall_svm = recall_score(y_diab_test, y_pred_svm_clf)  # Recall
f1_svm = f1_score(y_diab_test, y_pred_svm_clf)  # F1-Score

# Imprimir resultados
print(f"✅ MEJORES PARÁMETROS: {grid_svm_clf.best_params_}")
print(f"📊 RESULTADOS SVM:")
print(f" Exactitud: {accuracy_svm:.3f}")
print(f" Precisión: {precision_svm:.3f}")
print(f" Recall: {recall_svm:.3f}")
print(f" F1-Score: {f1_svm:.3f}")

# Matriz de confusión
cm_svm = confusion_matrix(y_diab_test, y_pred_svm_clf)  # Calcular matriz
plt.figure(figsize=(8, 6))
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Reds',  # Heatmap con colores rojos
            xticklabels=['No Avanzada', 'Avanzada'],
            yticklabels=['No Avanzada', 'Avanzada'])
plt.title('Matriz de Confusión - SVM (Diabetes)')
plt.ylabel('Real')  # Etiqueta Y: valores reales
plt.xlabel('Predicción')  # Etiqueta X: valores predichos
plt.show()
```

**Explicación del código:**

- `param_grid_svm_clf:\*\* Prueba combinaciones (e.g., 4x4x2=32 modelos). cv=5 para validación.
- `SVC:** Usa kernel 'rbf' por defecto. **¿Por qué kernel?\*\* Para capturar relaciones no lineales.
- `fit con datos escalados:** SVM requiere escalado. **¿Por qué?\*\* Para calcular márgenes correctamente.
- `predict:\*\* Clasifica en 0 o 1.
- `accuracy_score, etc.:\*\* Mismas métricas que el Árbol.
- `confusion_matrix:\*\* Similar al Árbol, pero cmap='Reds' para diferenciar.
- **Consejo:** Compara con el Árbol. SVM podría tener mejor exactitud en datos numéricos.

**¿Qué esperamos ver?** Exactitud similar o mejor que el Árbol, matriz con aciertos en diagonal.

### 3.6 Comparación y Análisis

Comparamos los modelos para Diabetes usando múltiples métricas de clasificación.

**Explicación:**

- **DataFrame de resultados:** Tabla con métricas para ambos modelos.
- **Gráficos de barras:** 2x2 grid para comparar Exactitud, Precisión, Recall y F1-Score. **¿Por qué múltiples métricas?** Para evaluar balance entre precisión y recall.
- **Textos en barras:** Valores exactos para claridad.
- **Conclusiones:** Basadas en exactitud, determinamos el mejor. **¿Por qué exactitud?** Es la métrica principal para clasificación balanceada.

```python
# Comparación de modelos diabetes

print("📊 COMPARACIÓN FINAL - DIABETES")

# Crear DataFrame con métricas
resultados_diabetes = pd.DataFrame({
    'Modelo': ['Árbol de Decisión', 'SVM'],
    'Exactitud': [accuracy_tree, accuracy_svm],  # Porcentaje de predicciones correctas
    'Precisión': [precision_tree, precision_svm],  # De positivos predichos, cuántos son reales
    'Recall': [recall_tree, recall_svm],  # De positivos reales, cuántos predice
    'F1-Score': [f1_tree, f1_svm]  # Balance de precisión y recall
})

print(resultados_diabetes)  # Mostrar tabla

# Visualización comparativa
fig, axes = plt.subplots(2, 2, figsize=(15, 10))  # 2x2 grid de gráficos
metricas = ['Exactitud', 'Precisión', 'Recall', 'F1-Score']  # Métricas a comparar

for i, metrica in enumerate(metricas):
    ax = axes[i//2, i%2]  # Seleccionar subplot
    valores = resultados_diabetes[metrica]  # Valores para la métrica
    bars = ax.bar(resultados_diabetes['Modelo'], valores,  # Gráfico de barras
                  color=['skyblue', 'lightcoral'])
    ax.set_ylabel(metrica)  # Etiqueta Y
    ax.set_title(f'Comparación: {metrica}')  # Título
    ax.set_ylim(0, 1)  # Límite Y de 0 a 1
    ax.grid(True, alpha=0.3)  # Grid

    # Añadir valores en las barras
    for bar, valor in zip(bars, valores):
        ax.text(bar.get_x() + bar.get_width()/2, valor + 0.01,
                f'{valor:.3f}', ha='center', va='bottom')  # Texto con valor

plt.tight_layout()
plt.show()

# Conclusiones
print("💡 CONCLUSIONES DIABETES:")
mejor_modelo_diab = 'Árbol de Decisión' if accuracy_tree > accuracy_svm else 'SVM'
print(f"• Mejor modelo: {mejor_modelo_diab}")
print(f"• Exactitud del mejor modelo: {max(accuracy_tree, accuracy_svm):.3f}")
print(f"• El modelo puede ayudar a identificar pacientes con diabetes avanzada")
```

**Explicación del código:**

- `pd.DataFrame:** Crea tabla con métricas. **¿Por qué?\*\* Para comparar fácilmente.
- `plt.subplots(2, 2):** 4 gráficos en grid. **¿Por qué?\*\* Para ver todas las métricas.
- `ax.bar:\*\* Gráfico para cada métrica. Colores para diferenciar modelos.
- `text:** Añade valores. **¿Por qué?\*\* Para precisión sin leer tabla.
- **Conclusiones:** Usa exactitud para decidir. **¿Por qué?** Es la más intuitiva para clasificación balanceada.
- **Consejo:** Si recall es importante (e.g., detectar diabetes), priorízalo.

**¿Qué esperamos ver?** SVM podría tener mejor exactitud en datos numéricos. El modelo ayuda a identificar diabetes avanzada para intervenciones tempranas.

---

## 4. Ejercicio 3: Titanic - Predicción de Supervivencia

Este ejercicio es un problema de **clasificación** con feature engineering: predecir supervivencia en el Titanic. Incluye manejo de valores faltantes y creación de nuevas características.

**¿Por qué este dataset?** Es un dataset clásico con datos mixtos (numéricos y categóricos), valores faltantes y necesidad de feature engineering. Te enseña a preparar datos reales.

**Pasos generales:**

1. Cargar y explorar los datos.
2. Visualizar relaciones y crear insights.
3. Preprocesar (manejar faltantes, feature engineering, codificar).
4. Entrenar modelos (Árbol y SVM).
5. Evaluar y comparar.

### 4.1 Carga y Exploración de Datos

Cargamos el dataset Titanic y exploramos su estructura, incluyendo valores faltantes.

**Explicación:**

- `sns.load_dataset('titanic'):` Carga desde seaborn. Si falla, usa URL alternativa.
- Mostramos forma, columnas, distribución de supervivencia, valores faltantes y primeras filas.
- **¿Por qué explorar?** Para ver desbalance de clases, variables categóricas y datos faltantes (e.g., 'age' y 'embarked').

```python
# Cargar dataset Titanic

print("\n" + "="*60)
print("🚢 EJERCICIO 3: TITANIC - PREDICCIÓN DE SUPERVIVENCIA")

# Cargar datos desde seaborn (o URL si falla)
try:
    df_titanic = sns.load_dataset('titanic')  # Intentar cargar desde seaborn
except:  # Si no funciona, cargar desde URL
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df_titanic = pd.read_csv(url)

# Mostrar información del dataset
print("📊 INFORMACIÓN DEL DATASET:")
print(f"Forma del dataset: {df_titanic.shape}")  # Filas y columnas
print(f"Columnas: {list(df_titanic.columns)}")  # Nombres de columnas
print(f"\nDistribución de supervivencia:")
print(df_titanic['survived'].value_counts())  # Conteo de 0 (no sobrevivió) y 1 (sobrevivió)
print(f"\nValores faltantes:")
print(df_titanic.isnull().sum())  # Número de valores faltantes por columna

# Mostrar las primeras 5 filas
print("\n🔍 PRIMERAS 5 FILAS:")
print(df_titanic.head())
```

**Explicación del código:**

- `try-except:` Para manejar si seaborn no está disponible. **¿Por qué?** Para robustez.
- `df_titanic.shape:` Muestra (filas, columnas). **¿Por qué?** Para ver tamaño del dataset.
- `value_counts():` Conteo de clases. **¿Por qué?** Para ver desbalance (e.g., más no sobrevivieron).
- `isnull().sum():` Valores faltantes. **¿Por qué?** Para planificar preprocesamiento.
- **Consejo:** Observa desbalance (e.g., 60% no sobrevivieron). Usa `stratify` en split.

**¿Qué esperamos ver?** Clases desbalanceadas, valores faltantes en 'age', 'embarked', 'deck'. Variables como 'sex' y 'pclass' influyen en supervivencia.

### 4.2 Análisis Exploratorio y Feature Engineering

Visualizamos relaciones clave y creamos insights iniciales para el Titanic.

**Explicación:**

- **Gráficos de barras:** Para variables categóricas como 'pclass', 'sex', 'embarked', 'sibsp' vs supervivencia.
- **Boxplots:** Para 'age' y 'fare' vs supervivencia, para ver distribuciones.
- **Insights:** Basados en gráficos, notamos patrones como mujeres y clases altas con mayor supervivencia.
- **¿Por qué estos gráficos?** Para entender factores de supervivencia y planificar feature engineering.

```python
# Análisis exploratorio Titanic

# Crear figura con subplots para visualizaciones
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 1. Supervivencia por clase (pclass)
survived_class = df_titanic.groupby(['pclass', 'survived']).size().unstack()  # Agrupar y contar
survived_class.plot(kind='bar', ax=axes[0, 0])  # Gráfico de barras
axes[0, 0].set_title('Supervivencia por Clase')
axes[0, 0].set_xlabel('Clase')
axes[0, 0].set_ylabel('Count')

# 2. Supervivencia por sexo
survived_sex = df_titanic.groupby(['sex', 'survived']).size().unstack()
survived_sex.plot(kind='bar', ax=axes[0, 1])
axes[0, 1].set_title('Supervivencia por Sexo')
axes[0, 1].set_xlabel('Sexo')

# 3. Edad vs Supervivencia
df_titanic.boxplot(column='age', by='survived', ax=axes[0, 2])  # Boxplot por supervivencia
axes[0, 2].set_title('Edad vs Supervivencia')

# 4. Tarifa vs Supervivencia
df_titanic.boxplot(column='fare', by='survived', ax=axes[1, 0])
axes[1, 0].set_title('Tarifa vs Supervivencia')

# 5. Supervivencia por puerto de embarque
if 'embarked' in df_titanic.columns:
    survived_embarked = df_titanic.groupby(['embarked', 'survived']).size().unstack()
    survived_embarked.plot(kind='bar', ax=axes[1, 1])
    axes[1, 1].set_title('Supervivencia por Puerto de Embarque')

# 6. Supervivencia por número de familiares (sibsp)
if 'sibsp' in df_titanic.columns:
    survived_sibsp = df_titanic.groupby(['sibsp', 'survived']).size().unstack()
    survived_sibsp.plot(kind='bar', ax=axes[1, 2])
    axes[1, 2].set_title('Supervivencia por Hermanos/Esposos')

plt.tight_layout()
plt.show()

# Imprimir insights iniciales
print("🔍 INSIGHTS INICIALES TITANIC:")
print("• Las mujeres tuvieron mayor tasa de supervivencia")
print("• La clase social influyó significativamente")
print("• Los niños tuvieron prioridad en los botes salvavidas")
```

**Explicación del código:**

- `plt.subplots(2, 3):` Crea 6 gráficos. **¿Por qué?** Para analizar múltiples variables.
- `groupby(['pclass', 'survived']).size().unstack():` Cuenta por clase y supervivencia. **¿Por qué?** Para ver tasas de supervivencia por categoría.
- `boxplot(column='age', by='survived'):` Distribuciones de edad por supervivencia. **¿Por qué?** Para ver si edad afecta supervivencia.
- **Consejo:** Observa patrones como mujeres y clases altas con mayor supervivencia. Usa esto para feature engineering.

**¿Qué esperamos ver?** Mujeres y clases altas con mayor supervivencia. Edad y tarifa podrían diferir por supervivencia.

### 4.3 Preprocesamiento Avanzado

Preparamos los datos del Titanic con feature engineering: manejamos faltantes, creamos nuevas características y codificamos categóricas.

**Explicación:**

- **Manejar faltantes:** Rellenamos 'age' con mediana, 'embarked' con moda, eliminamos 'deck' (muchos faltantes).
- **Feature engineering:** Creamos 'family_size' (tamaño familia), 'is_alone' (si viaja solo), 'title' (título de nombre).
- **Codificar categóricas:** Usamos LabelEncoder para 'sex', 'embarked', 'title'.
- **Seleccionar features:** Elegimos variables relevantes.
- **Dividir y escalar:** Similar a antes, con estratificación por desbalance.
- **¿Por qué feature engineering?** Para mejorar el modelo con variables derivadas.

```python
# Feature Engineering para Titanic

print("🔧 PREPROCESAMIENTO AVANZADO - TITANIC")

# Crear copia para no modificar original
df_titanic_clean = df_titanic.copy()

# 1. Manejar valores faltantes
df_titanic_clean['age'].fillna(df_titanic_clean['age'].median(), inplace=True)  # Rellenar edad con mediana
df_titanic_clean['embarked'].fillna(df_titanic_clean['embarked'].mode()[0], inplace=True)  # Rellenar embarked con moda
df_titanic_clean.drop(columns=['deck'], inplace=True, errors='ignore')  # Eliminar 'deck' (muchos faltantes)

# 2. Crear nuevas características
df_titanic_clean['family_size'] = df_titanic_clean['sibsp'] + df_titanic_clean['parch'] + 1  # Tamaño familia
df_titanic_clean['is_alone'] = (df_titanic_clean['family_size'] == 1).astype(int)  # 1 si solo, 0 si no
df_titanic_clean['title'] = df_titanic_clean['name'].str.extract(' ([A-Za-z]+)\.', expand=False)  # Extraer título

# Simplificar títulos
title_mapping = {'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
                 'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
                 'Mlle': 'Miss', 'Countess': 'Rare', 'Ms': 'Miss', 'Lady': 'Rare',
                 'Jonkheer': 'Rare', 'Don': 'Rare', 'Dona': 'Rare', 'Mme': 'Mrs',
                 'Capt': 'Rare', 'Sir': 'Rare'}
df_titanic_clean['title'] = df_titanic_clean['title'].map(title_mapping)  # Mapear títulos

# 3. Codificar variables categóricas
label_encoders = {}
categorical_cols = ['sex', 'embarked', 'title']

for col in categorical_cols:
    if col in df_titanic_clean.columns:
        le = LabelEncoder()  # Inicializar encoder
        df_titanic_clean[col] = le.fit_transform(df_titanic_clean[col].astype(str))  # Codificar
        label_encoders[col] = le  # Guardar encoder

# 4. Seleccionar características finales
features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked',
            'family_size', 'is_alone', 'title']
X_titanic = df_titanic_clean[features]  # X: features seleccionadas
y_titanic = df_titanic_clean['survived']  # y: supervivencia

# 5. Dividir datos
X_titanic_train, X_titanic_test, y_titanic_train, y_titanic_test = train_test_split(
    X_titanic, y_titanic, test_size=0.3, random_state=42, stratify=y_titanic  # stratify por desbalance
)

# 6. Escalar para SVM
scaler_titanic = StandardScaler()  # Inicializar escalador
X_titanic_train_scaled = scaler_titanic.fit_transform(X_titanic_train)  # Ajustar y transformar train
X_titanic_test_scaled = scaler_titanic.transform(X_titanic_test)  # Solo transformar test

# Confirmar preparación
print("✅ DATOS TITANIC PREPARADOS:")
print(f"Características usadas: {features}")
print(f"Entrenamiento: {X_titanic_train.shape[0]} muestras")
print(f"Prueba: {X_titanic_test.shape[0]} muestras")
print(f"Proporción supervivencia: {y_titanic_train.value_counts(normalize=True).to_dict()}")
```

**Explicación del código:**

- `fillna con median/mode:` Para 'age' (numérica) y 'embarked' (categórica). **¿Por qué?** Mediana para numérica, moda para categórica.
- `drop('deck'):` Elimina columna con muchos faltantes. **¿Por qué?** Evita ruido.
- `family_size = sibsp + parch + 1:` Incluye al pasajero. **¿Por qué?** Para capturar efecto de familia.
- `is_alone:` Binaria para si viaja solo. **¿Por qué?** Podría afectar supervivencia.
- `title:` Extrae de 'name'. **¿Por qué?** Títulos como 'Mr' o 'Mrs' indican género/edad.
- `LabelEncoder:` Convierte categóricas a números. **¿Por qué?** Modelos necesitan números.
- `features:` Lista de variables finales. **¿Por qué?** Para enfocarse en relevantes.
- **Consejo:** Feature engineering mejora el modelo. Experimenta con más variables.

**¿Qué sigue?** Entrenamos los modelos con estos datos preparados.

### 4.4 Modelo: Árbol de Decisión para Titanic

Entrenamos un Árbol de Decisión para predecir supervivencia en el Titanic, con parámetros fijos para simplicidad.

**Explicación:**

- **DecisionTreeClassifier:** Modelo para clasificación con datos mixtos.
- **Parámetros fijos:** max_depth=5, etc., para evitar overfitting en datos pequeños.
- **Métricas:** Mismas que antes.
- **Visualizar árbol:** Muestra los primeros 3 niveles para interpretabilidad.
- **Importancia de características:** Muestra qué variables son más importantes.

```python
# Árbol de Decisión para Titanic

print("🌳 ENTRENANDO ÁRBOL DE DECISIÓN (TITANIC)")

# Inicializar modelo con parámetros fijos
tree_titanic = DecisionTreeClassifier(
    max_depth=5,  # Profundidad máxima para evitar overfitting
    min_samples_split=10,  # Mínimo muestras para split
    min_samples_leaf=5,  # Mínimo muestras en hoja
    random_state=42  # Para reproducibilidad
)
tree_titanic.fit(X_titanic_train, y_titanic_train)  # Entrenar

# Predicciones
y_pred_tree_titanic = tree_titanic.predict(X_titanic_test)  # Predecir en test

# Calcular métricas
accuracy_tree_titanic = accuracy_score(y_titanic_test, y_pred_tree_titanic)  # Exactitud
precision_tree_titanic = precision_score(y_titanic_test, y_pred_tree_titanic)  # Precisión
recall_tree_titanic = recall_score(y_titanic_test, y_pred_tree_titanic)  # Recall
f1_tree_titanic = f1_score(y_titanic_test, y_pred_tree_titanic)  # F1-Score

# Imprimir resultados
print(f"📊 RESULTADOS ÁRBOL - TITANIC:")
print(f" Exactitud: {accuracy_tree_titanic:.3f}")
print(f" Precisión: {precision_tree_titanic:.3f}")
print(f" Recall: {recall_tree_titanic:.3f}")
print(f" F1-Score: {f1_tree_titanic:.3f}")

# Visualizar árbol (versión simplificada)
plt.figure(figsize=(20, 10))
plot_tree(tree_titanic,  # El modelo entrenado
          feature_names=features,  # Nombres de características
          class_names=['No Sobrevivió', 'Sobrevivió'],  # Nombres de clases
          filled=True,  # Colorear nodos por clase
          rounded=True,  # Bordes redondeados
          fontsize=10,  # Tamaño de fuente
          max_depth=3)  # Mostrar solo primeros 3 niveles para claridad
plt.title('Árbol de Decisión - Titanic (Primeros 3 niveles)', fontsize=16)
plt.show()

# Importancia de características
importancia_titanic = pd.DataFrame({
    'caracteristica': features,  # Nombres de features
    'importancia': tree_titanic.feature_importances_  # Importancia calculada
}).sort_values('importancia', ascending=False)  # Ordenar de mayor a menor

plt.figure(figsize=(10, 6))
plt.barh(importancia_titanic['caracteristica'], importancia_titanic['importancia'])  # Gráfico horizontal
plt.xlabel('Importancia')  # Etiqueta X
plt.title('Importancia de Características - Titanic')
plt.gca().invert_yaxis()  # Invertir Y para que la más importante esté arriba
plt.grid(True, alpha=0.3, axis='x')  # Grid
plt.show()
```

**Explicación del código:**

- `DecisionTreeClassifier con parámetros fijos:** No usamos GridSearch para simplicidad. **¿Por qué?\*\* Para enfocarnos en feature engineering.
- `fit y predict:\*\* Entrena y predice.
- `métricas:\*\* Mismas que antes.
- `plot_tree:** Visualiza el árbol. **¿Por qué max_depth=3?\*\* Para no sobrecargar el gráfico.
- `feature*importances*:** Importancia basada en reducción de error. **¿Por qué visualizar?\*\* Para ver qué features son clave (e.g., 'sex' y 'pclass').
- **Consejo:** Interpreta el árbol: cada nodo es una decisión basada en una feature.

**¿Qué esperamos ver?** Exactitud >0.8, importancia alta en 'sex' y 'pclass'. El árbol muestra reglas como "Si es mujer, sobrevivió".

### 4.5 Modelo: SVM para Titanic

Entrenamos un SVM para predecir supervivencia en el Titanic, con parámetros fijos.

**Explicación:**

- **SVC:** Modelo para clasificación con datos mixtos y escalados.
- **Parámetros fijos:** C=1.0, kernel='rbf', gamma='scale' para simplicidad.
- **Métricas:** Mismas que el Árbol.
- **Matrices de confusión:** Comparativas con el Árbol, colores diferentes para distinguir.

```python
# SVM para Titanic

print("🎯 ENTRENANDO SVM (TITANIC)")

# Inicializar modelo con parámetros fijos
svm_titanic = SVC(
    C=1.0,  # Penalización por errores
    kernel='rbf',  # Kernel no lineal
    gamma='scale',  # Influencia de puntos
    random_state=42  # Para reproducibilidad
)
svm_titanic.fit(X_titanic_train_scaled, y_titanic_train)  # Entrenar con datos escalados

# Predicciones
y_pred_svm_titanic = svm_titanic.predict(X_titanic_test_scaled)  # Predecir en test escalado

# Calcular métricas
accuracy_svm_titanic = accuracy_score(y_titanic_test, y_pred_svm_titanic)  # Exactitud
precision_svm_titanic = precision_score(y_titanic_test, y_pred_svm_titanic)  # Precisión
recall_svm_titanic = recall_score(y_titanic_test, y_pred_svm_titanic)  # Recall
f1_svm_titanic = f1_score(y_titanic_test, y_pred_svm_titanic)  # F1-Score

# Imprimir resultados
print(f"📊 RESULTADOS SVM - TITANIC:")
print(f" Exactitud: {accuracy_svm_titanic:.3f}")
print(f" Precisión: {precision_svm_titanic:.3f}")
print(f" Recall: {recall_svm_titanic:.3f}")
print(f" F1-Score: {f1_svm_titanic:.3f}")

# Matrices de confusión comparativas
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Matriz para Árbol
cm_tree_titanic = confusion_matrix(y_titanic_test, y_pred_tree_titanic)  # Calcular para Árbol
sns.heatmap(cm_tree_titanic, annot=True, fmt='d', cmap='Blues', ax=axes[0],  # Colores azules
            xticklabels=['No Sobrevivió', 'Sobrevivió'],
            yticklabels=['No Sobrevivió', 'Sobrevivió'])
axes[0].set_title('Matriz Confusión - Árbol de Decisión')
axes[0].set_ylabel('Real')
axes[0].set_xlabel('Predicción')

# Matriz para SVM
cm_svm_titanic = confusion_matrix(y_titanic_test, y_pred_svm_titanic)  # Calcular para SVM
sns.heatmap(cm_svm_titanic, annot=True, fmt='d', cmap='Reds', ax=axes[1],  # Colores rojos
            xticklabels=['No Sobrevivió', 'Sobrevivió'],
            yticklabels=['No Sobrevivió', 'Sobrevivió'])
axes[1].set_title('Matriz Confusión - SVM')
axes[1].set_ylabel('Real')
axes[1].set_xlabel('Predicción')

plt.tight_layout()
plt.show()
```

**Explicación del código:**

- `SVC con parámetros fijos:** No usamos GridSearch para simplicidad. **¿Por qué?\*\* Para comparar con el Árbol rápidamente.
- `fit con datos escalados:** SVM requiere escalado. **¿Por qué?\*\* Para márgenes correctos.
- `predict:\*\* Clasifica en 0 o 1.
- `métricas:\*\* Mismas que el Árbol.
- `confusion_matrix:** Dos matrices lado a lado. **¿Por qué cmap diferentes?\*\* Para distinguir modelos.
- **Consejo:** Compara matrices. SVM podría tener menos falsos positivos.

**¿Qué esperamos ver?** Exactitud similar al Árbol, matrices con aciertos en diagonal.

### 4.6 Comparación Final y Análisis

Comparamos los modelos para Titanic y resumimos todos los ejercicios.

**Explicación:**

- **DataFrame de resultados:** Tabla con métricas para Titanic.
- **Resumen final:** DataFrame con métricas principales de todos los datasets.
- **Gráficos de barras:** Uno por dataset para comparar modelos.
- **Conclusiones:** Basadas en resultados, destacamos ventajas y recomendaciones.

```python
# Comparación final todos los modelos

print("📊 COMPARACIÓN FINAL - TITANIC")

# Crear DataFrame con métricas para Titanic
resultados_titanic = pd.DataFrame({
    'Modelo': ['Árbol de Decisión', 'SVM'],
    'Exactitud': [accuracy_tree_titanic, accuracy_svm_titanic],  # Exactitud
    'Precisión': [precision_tree_titanic, precision_svm_titanic],  # Precisión
    'Recall': [recall_tree_titanic, recall_svm_titanic],  # Recall
    'F1-Score': [f1_tree_titanic, f1_svm_titanic]  # F1-Score
})

print(resultados_titanic)  # Mostrar tabla

# Resumen de todos los ejercicios
print("\n" + "="*80)
print("🎯 RESUMEN FINAL DE TODOS LOS EJERCICIOS")
print("="*80)

# Crear DataFrame con métricas principales de cada modelo y dataset
resumen_final = pd.DataFrame({
    'Dataset': ['California Housing', 'California Housing', 'Diabetes', 'Diabetes', 'Titanic', 'Titanic'],
    'Modelo': ['Árbol', 'SVM', 'Árbol', 'SVM', 'Árbol', 'SVM'],
    'Métrica Principal': [r2_tree, r2_svm, accuracy_tree, accuracy_svm, accuracy_tree_titanic, accuracy_svm_titanic],  # R² para regresión, Exactitud para clasificación
    'Valor': [r2_tree, r2_svm, accuracy_tree, accuracy_svm, accuracy_tree_titanic, accuracy_svm_titanic]  # Valores numéricos
})

print(resumen_final)  # Mostrar resumen

# Visualización resumen final
fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # 1 fila, 3 columnas

# California Housing (R²)
axes[0].bar(['Árbol', 'SVM'], [r2_tree, r2_svm], color=['skyblue', 'lightcoral'])
axes[0].set_title('California Housing\n(R² Score)')
axes[0].set_ylabel('R²')
axes[0].grid(True, alpha=0.3)
for i, v in enumerate([r2_tree, r2_svm]):
    axes[0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')  # Texto en barras

# Diabetes (Exactitud)
axes[1].bar(['Árbol', 'SVM'], [accuracy_tree, accuracy_svm], color=['skyblue', 'lightcoral'])
axes[1].set_title('Diabetes\n(Exactitud)')
axes[1].set_ylabel('Exactitud')
axes[1].grid(True, alpha=0.3)
for i, v in enumerate([accuracy_tree, accuracy_svm]):
    axes[1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')

# Titanic (Exactitud)
axes[2].bar(['Árbol', 'SVM'], [accuracy_tree_titanic, accuracy_svm_titanic], color=['skyblue', 'lightcoral'])
axes[2].set_title('Titanic\n(Exactitud)')
axes[2].set_ylabel('Exactitud')
axes[2].grid(True, alpha=0.3)
for i, v in enumerate([accuracy_tree_titanic, accuracy_svm_titanic]):
    axes[2].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# Conclusiones generales
print("""
💡 CONCLUSIONES GENERALES:

🌳 ÁRBOLES DE DECISIÓN:
• Ventajas: Interpretables, no necesitan escalado, manejan características mixtas
• Desventajas: Propensos a overfitting, inestables
• Mejor en: Titanic (con feature engineering)

🎯 SVM:
• Ventajas: Buen rendimiento con datos complejos, robustos a outliers
• Desventajas: Necesitan escalado, lentos con muchos datos, difíciles de interpretar
• Mejor en: Diabetes (datos numéricos limpios)

📊 POR DATASET:
• California Housing: Ambos modelos similares, problema de regresión
• Diabetes: SVM ligeramente mejor en clasificación
• Titanic: Árbol mejor gracias a la interpretabilidad y feature engineering

🚀 RECOMENDACIONES:
• Usar árboles cuando necesites explicar las decisiones
• Usar SVM cuando la precisión sea prioridad y tengas datos escalados
• Siempre hacer feature engineering y preprocesamiento adecuado
""")
```

**Explicación del código:**

- `resultados_titanic:** DataFrame para Titanic. **¿Por qué?\*\* Para comparar métricas.
- `resumen_final:** DataFrame con todos los datasets. **¿Por qué?\*\* Para ver overview.
- `plt.subplots(1, 3):** Tres gráficos para cada dataset. **¿Por qué?\*\* Para comparar visualmente.
- `axes[0].bar, etc.:** Gráfico por dataset. **¿Por qué colores?\*\* Para consistencia.
- `text:** Añade valores. **¿Por qué?\*\* Para precisión.
- **Conclusiones:** Texto con ventajas/desventajas. **¿Por qué?** Para guiar al usuario.

**¿Qué esperamos ver?** Árbol mejor en Titanic por interpretabilidad, SVM en Diabetes por precisión. Recomendaciones basadas en resultados.
🧪 EJERCICIOS ADICIONALES PARA PRACTICAR
python
print("\n" + "="*80)
print("🧪 EJERCICIOS ADICIONALES PARA PRACTICAR")
print("="*80)

ejercicios_adicionales = """
🎯 EJERCICIO 4: OPTIMIZACIÓN AVANZADA

1. Mejora el modelo de Titanic:

   - Prueba diferentes parámetros para el árbol
   - Experimenta con otros kernels en SVM
   - Añade más feature engineering (edad en categorías, etc.)

2. Ensemble Methods:

   - Prueba Random Forest en lugar de un solo árbol
   - Compara con los resultados obtenidos

3. Análisis de Errores:
   - ¿Qué pasajeros clasifica mal el modelo?
   - ¿Hay patrones en los errores?

🔍 EJERCICIO 5: INTERPRETACIÓN DE MODELOS

1. Para el árbol de Titanic:

   - Explica 3 reglas de decisión importantes
   - ¿Qué características son más importantes?

2. Para SVM:
   - ¿Por qué crees que funcionó mejor en Diabetes?
   - ¿Cómo afecta el parámetro C a los resultados?

📈 EJERCICIO 6: VALIDACIÓN CRUZADA

1. Implementa k-fold cross validation en todos los modelos
2. Compara los resultados con la división simple train/test
3. Analiza la varianza de los resultados

🚀 EJERCICIO 7: APLICACIÓN A NUEVOS DATOS

1. Encuentra otro dataset de clasificación (ej: Iris, Wine)
2. Aplica el mismo proceso completo
3. Compara tus resultados con los obtenidos aquí
   """

print(ejercicios_adicionales)

print("""
🎉 ¡FELICITACIONES POR COMPLETAR LOS EJERCICIOS!

Has aplicado exitosamente:
• Árboles de Decisión para regresión y clasificación
• SVM para regresión y clasificación  
• Preprocesamiento y feature engineering
• Evaluación y comparación de modelos
• Tres datasets del mundo real

¡Sigue practicando y explorando! 🚀
""")
