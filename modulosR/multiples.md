🔢 Modelos con Múltiples Parámetros de Entrada
Guía Completa para Principiantes
python

# Configuración inicial

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

1. 🆕 ¿QUÉ CAMBIA CON MÚLTIPLES PARÁMETROS?
   markdown

# 🤔 De 1 parámetro a MÚLTIPLES parámetros

## ESCENARIO SIMPLE (1 parámetro):

**Predecir precio de casa basado solo en:**

- Tamaño (m²)

## ESCENARIO REAL (múltiples parámetros):

**Predecir precio de casa basado en:**

- Tamaño (m²)
- Número de habitaciones
- Años de antigüedad
- Barrio
- ¿Tiene garage?
- ¿Tiene piscina?
  python

# Crear dataset realista con múltiples parámetros

np.random.seed(42)
n_muestras = 500

datos = pd.DataFrame({
'tamaño': np.random.normal(120, 40, n_muestras),
'habitaciones': np.random.randint(1, 6, n_muestras),
'antiguedad': np.random.randint(0, 50, n_muestras),
'barrio_cod': np.random.randint(1, 4, n_muestras), # 3 barrios diferentes
'tiene_garage': np.random.choice([0, 1], n_muestras, p=[0.3, 0.7]),
'tiene_piscina': np.random.choice([0, 1], n_muestras, p=[0.7, 0.3])
})

# Crear precio basado en una fórmula realista

precio_base = 50000
precio = (datos['tamaño'] _ 1000 +
datos['habitaciones'] _ 20000 +
-datos['antiguedad'] _ 1000 +
datos['barrio_cod'] _ 30000 +
datos['tiene_garage'] _ 15000 +
datos['tiene_piscina'] _ 25000 +
np.random.normal(0, 20000, n_muestras))

datos['precio'] = precio

print("🏠 DATASET CON MÚLTIPLES PARÁMETROS:")
print(f"Forma de los datos: {datos.shape}")
print(f"Columnas: {list(datos.columns)}")
print("\nPrimeras 5 filas:")
print(datos.head())
📝 ¿QUÉ ESTAMOS HACIENDO?
Creamos un dataset más realista con 6 parámetros que afectan el precio de una casa.

🎯 ¿POR QUÉ MÚLTIPLES PARÁMETROS?
Más realista: En el mundo real, las decisiones dependen de muchos factores

Más preciso: Más información generalmente lleva a mejores predicciones

Más complejo: Necesitamos técnicas especiales para manejar múltiples variables

2. 🔍 ANÁLISIS EXPLORATORIO CON MÚLTIPLES VARIABLES
   python

# Análisis exploratorio avanzado

print("📊 ANÁLISIS EXPLORATORIO CON MÚLTIPLES VARIABLES")

# 1. Estadísticas básicas

print("\n1. 📈 ESTADÍSTICAS BÁSICAS:")
print(datos.describe())

# 2. Matriz de correlación (¡MUY IMPORTANTE!)

print("\n2. 🔗 MATRIZ DE CORRELACIÓN:")
correlation_matrix = datos.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
square=True, linewidths=0.5)
plt.title('Matriz de Correlación entre Variables')
plt.tight_layout()
plt.show()

# 3. Visualizaciones múltiples

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Tamaño vs Precio

axes[0, 0].scatter(datos['tamaño'], datos['precio'], alpha=0.6)
axes[0, 0].set_xlabel('Tamaño (m²)')
axes[0, 0].set_ylabel('Precio')
axes[0, 0].set_title('Tamaño vs Precio')
axes[0, 0].grid(True, alpha=0.3)

# Habitaciones vs Precio

habitaciones_agrupadas = datos.groupby('habitaciones')['precio'].mean()
axes[0, 1].bar(habitaciones_agrupadas.index, habitaciones_agrupadas.values)
axes[0, 1].set_xlabel('Número de Habitaciones')
axes[0, 1].set_ylabel('Precio Promedio')
axes[0, 1].set_title('Habitaciones vs Precio')
axes[0, 1].grid(True, alpha=0.3)

# Antigüedad vs Precio

axes[0, 2].scatter(datos['antiguedad'], datos['precio'], alpha=0.6)
axes[0, 2].set_xlabel('Antigüedad (años)')
axes[0, 2].set_ylabel('Precio')
axes[0, 2].set_title('Antigüedad vs Precio')
axes[0, 2].grid(True, alpha=0.3)

# Barrio vs Precio

barrio_agrupado = datos.groupby('barrio_cod')['precio'].mean()
axes[1, 0].bar(barrio_agrupado.index, barrio_agrupado.values)
axes[1, 0].set_xlabel('Barrio')
axes[1, 0].set_ylabel('Precio Promedio')
axes[1, 0].set_title('Barrio vs Precio')
axes[1, 0].grid(True, alpha=0.3)

# Garage vs Precio

garage_agrupado = datos.groupby('tiene_garage')['precio'].mean()
axes[1, 1].bar(['Sin Garage', 'Con Garage'], garage_agrupado.values)
axes[1, 1].set_ylabel('Precio Promedio')
axes[1, 1].set_title('Garage vs Precio')
axes[1, 1].grid(True, alpha=0.3)

# Piscina vs Precio

piscina_agrupado = datos.groupby('tiene_piscina')['precio'].mean()
axes[1, 2].bar(['Sin Piscina', 'Con Piscina'], piscina_agrupado.values)
axes[1, 2].set_ylabel('Precio Promedio')
axes[1, 2].set_title('Piscina vs Precio')
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
📝 ¿QUÉ ESTAMOS HACIENDO?
Analizamos cómo se relaciona CADA variable con el precio individualmente.

🎯 ¿POR QUÉ ES IMPORTANTE?
Detectar relaciones: Ver qué variables tienen más impacto

Identificar problemas: Variables que no se relacionan con el target

Multicolinealidad: Variables que están correlacionadas entre sí (problema)

🔍 PARA QUÉ SIRVE
Entender cuáles variables son importantes antes de construir el modelo.

3. 🏗️ PREPARACIÓN DE DATOS PARA MÚLTIPLES VARIABLES
   python

# PASO CRUCIAL: Preparar datos para múltiples variables

print("🎯 PREPARANDO DATOS PARA MÚLTIPLES VARIABLES")

# 1. Separar características (X) y target (y)

X = datos.drop('precio', axis=1) # Todas las columnas excepto precio
y = datos['precio'] # Solo la columna precio

print(f"Características (X): {X.shape}")
print(f"Target (y): {y.shape}")

# 2. Dividir en entrenamiento y prueba

X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42
)

print(f"\n📊 DIVISIÓN:")
print(f"Entrenamiento: {X_train.shape[0]} muestras")
print(f"Prueba: {X_test.shape[0]} muestras")

# 3. Estandarizar/normalizar variables (OPCIONAL pero recomendado)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n🔧 PREPROCESAMIENTO:")
print("Variables estandarizadas (media=0, desviación=1)")
print(f"Ejemplo de datos estandarizados: {X_train_scaled[0]}")
📝 ¿QUÉ ESTAMOS HACIENDO?
Preparamos los datos específicamente para trabajar con múltiples variables.

🎯 ¿POR QUÉ ES DIFERENTE?
X ahora es una tabla: No solo una columna

Estandarización: Importante cuando variables tienen diferentes escalas

Manejo de variables categóricas: Barrios necesitan tratamiento especial

🔍 PARA QUÉ SIRVE
Evitar que variables con valores grandes (como tamaño) dominen sobre variables con valores pequeños (como número de habitaciones).

4. 🧠 ENTRENAR MODELO CON MÚLTIPLES VARIABLES
   python

# Entrenar modelo con múltiples variables

modelo_multi = LinearRegression()
modelo_multi.fit(X_train_scaled, y_train)

print("🎯 MODELO CON MÚLTIPLES VARIABLES ENTRENADO")

# Mostrar los coeficientes (importancia de cada variable)

coeficientes = pd.DataFrame({
'Variable': X.columns,
'Coeficiente': modelo*multi.coef*,
'Impacto*Absoluto': np.abs(modelo_multi.coef*)
}).sort_values('Impacto_Absoluto', ascending=False)

print("\n📊 IMPORTANCIA DE CADA VARIABLE:")
print(coeficientes)

# Interpretación sencilla

print("\n💡 INTERPRETACIÓN SENCILLA:")
print("POSITIVO: Cuando la variable AUMENTA, el precio AUMENTA")
print("NEGATIVO: Cuando la variable AUMENTA, el precio DISMINUYE")
print("VALOR ABSOLUTO: Qué tan fuerte es el efecto")

# Visualizar importancia

plt.figure(figsize=(10, 6))
bars = plt.barh(coeficientes['Variable'], coeficientes['Impacto_Absoluto'])
plt.xlabel('Impacto en el Precio (Valor Absoluto)')
plt.title('Importancia de Cada Variable en la Predicción del Precio')
plt.grid(True, alpha=0.3, axis='x')

# Añadir valores en las barras

for bar in bars:
width = bar.get_width()
plt.text(width, bar.get_y() + bar.get_height()/2,
f'{width:.0f}', ha='left', va='center')

plt.tight_layout()
plt.show()
📝 ¿QUÉ ESTAMOS HACIENDO?
Entrenamos un modelo que considera TODAS las variables simultáneamente.

🎯 ¿CÓMO FUNCIONA?
El modelo aprende una fórmula como:

text
Precio = (coef1 × tamaño) + (coef2 × habitaciones) + ... + intercepto
🔍 PARA QUÉ SIRVE
Cada coeficiente nos dice cuánto afecta cada variable al precio, manteniendo las otras constantes.

5. 📈 EVALUAR MODELO CON MÚLTIPLES VARIABLES
   python

# Evaluar el modelo múltiple

y_pred_multi = modelo_multi.predict(X_test_scaled)

# Métricas

mse_multi = mean_squared_error(y_test, y_pred_multi)
rmse_multi = np.sqrt(mse_multi)
r2_multi = r2_score(y_test, y_pred_multi)

print("📊 EVALUACIÓN DEL MODELO MÚLTIPLE:")
print(f"Error Cuadrático Medio (MSE): ${mse_multi:,.0f}")
print(f"Raíz del Error Cuadrático (RMSE): ${rmse_multi:,.0f}")
print(f"Coeficiente R²: {r2_multi:.3f}")

# Comparar con modelo simple (solo tamaño)

modelo_simple = LinearRegression()
modelo_simple.fit(X_train[['tamaño']], y_train)
y_pred_simple = modelo_simple.predict(X_test[['tamaño']])
r2_simple = r2_score(y_test, y_pred_simple)

print(f"\n🔍 COMPARACIÓN:")
print(f"R² con solo TAMAÑO: {r2_simple:.3f}")
print(f"R² con TODAS las variables: {r2_multi:.3f}")
print(f"Mejora: {r2_multi - r2_simple:.3f}")

# Visualizar comparación

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Predicciones vs Reales

axes[0].scatter(y_test, y_pred_multi, alpha=0.6)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0].set_xlabel('Precio Real')
axes[0].set_ylabel('Precio Predicho')
axes[0].set_title('Predicciones vs Reales\n(Modelo Múltiple)')
axes[0].grid(True, alpha=0.3)

# Errores

errores = y_test - y_pred_multi
axes[1].hist(errores, bins=30, edgecolor='black', alpha=0.7)
axes[1].axvline(x=0, color='red', linestyle='--')
axes[1].set_xlabel('Error de Predicción')
axes[1].set_ylabel('Frecuencia')
axes[1].set_title('Distribución de Errores')
axes[1].grid(True, alpha=0.3)

# Comparación modelos

modelos = ['Solo Tamaño', 'Todas Variables']
r2_scores = [r2_simple, r2_multi]
bars = axes[2].bar(modelos, r2_scores, color=['lightblue', 'lightgreen'])
axes[2].set_ylabel('R² Score')
axes[2].set_title('Comparación: Modelo Simple vs Múltiple')
axes[2].grid(True, alpha=0.3)

# Añadir valores en las barras

for bar, score in zip(bars, r2_scores):
height = bar.get_height()
axes[2].text(bar.get_x() + bar.get_width()/2, height + 0.01,
f'{score:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()
📝 ¿QUÉ ESTAMOS HACIENDO?
Evaluamos si agregar más variables realmente mejora el modelo.

🎯 ¿CÓMO SABEMOS SI ES MEJOR?
R² más alto: Explica más variación en los datos

Errores más pequeños: Predicciones más precisas

Errores aleatorios: Sin patrones sistemáticos

🔍 PARA QUÉ SIRVE
Confirmar que las variables adicionales aportan valor real al modelo.

6. 🚨 PELIGROS CON MÚLTIPLES VARIABLES
   python

# PELIGRO 1: Overfitting (modelo demasiado complejo)

print("🚨 PELIGROS CON MÚLTIPLES VARIABLES")

# Crear variables irrelevantes (ruido)

X_train_con_ruido = X_train.copy()
X_test_con_ruido = X_test.copy()

for i in range(5): # Añadir 5 variables de ruido
X*train_con_ruido[f'ruido*{i}'] = np.random.normal(0, 1, len(X*train))
X_test_con_ruido[f'ruido*{i}'] = np.random.normal(0, 1, len(X_test))

# Entrenar modelo con variables irrelevantes

modelo_con_ruido = LinearRegression()
modelo_con_ruido.fit(X_train_con_ruido, y_train)

# Evaluar

train_score_ruido = modelo_con_ruido.score(X_train_con_ruido, y_train)
test_score_ruido = modelo_con_ruido.score(X_test_con_ruido, y_test)

print("\n🔍 PELIGRO 1: OVERFITTING")
print(f"R² Entrenamiento (con ruido): {train_score_ruido:.3f}")
print(f"R² Prueba (con ruido): {test_score_ruido:.3f}")
print(f"R² Prueba (sin ruido): {r2_multi:.3f}")

if test_score_ruido < r2_multi:
print("✅ CONCLUSIÓN: Variables irrelevantes empeoran el modelo")

# PELIGRO 2: Multicolinealidad

print("\n🔍 PELIGRO 2: MULTICOLINEALIDAD")
print("Ocurre cuando variables están muy correlacionadas entre sí")

# Crear ejemplo de multicolinealidad

datos_multicol = datos.copy()
datos_multicol['tamaño_habitaciones'] = datos_multicol['tamaño'] \* datos_multicol['habitaciones'] # Variable redundante

corr_alta = datos_multicol[['tamaño', 'tamaño_habitaciones']].corr().iloc[0, 1]
print(f"Correlación entre tamaño y tamaño_habitaciones: {corr_alta:.3f}")

if corr_alta > 0.8:
print("🚨 ALTA CORRELACIÓN: Puede causar problemas en el modelo")
📝 ¿QUÉ ESTAMOS HACIENDO?
Mostramos los riesgos de trabajar con muchas variables.

🎯 PELIGROS COMUNES:
Overfitting: Modelo memoriza el ruido en lugar de aprender patrones

Multicolinealidad: Variables redundantes que confunden al modelo

Complejidad innecesaria: Más variables no siempre es mejor

🔍 PARA QUÉ SIRVE
Aprender a reconocer cuándo estamos agregando variables que perjudican el modelo.

7. 🎯 SELECCIÓN DE VARIABLES INTELIGENTE
   python

# Técnica: Seleccionar solo las variables importantes

print("🎯 SELECCIÓN INTELIGENTE DE VARIABLES")

# Método 1: Usar los coeficientes del modelo

variables_importantes = coeficientes[coeficientes['Impacto_Absoluto'] > 1000]['Variable'].tolist()

print(f"Variables seleccionadas: {variables_importantes}")

# Entrenar modelo solo con variables importantes

X_train_sel = X_train[variables_importantes]
X_test_sel = X_test[variables_importantes]

modelo_seleccionado = LinearRegression()
modelo_seleccionado.fit(X_train_sel, y_train)

r2_seleccionado = modelo_seleccionado.score(X_test_sel, y_test)

print(f"\n📊 COMPARACIÓN FINAL:")
print(f"R² con TODAS las variables: {r2_multi:.3f}")
print(f"R² con variables SELECCIONADAS: {r2_seleccionado:.3f}")

# Método 2: Usar correlación con el target

correlacion_con_target = datos.corr()['precio'].abs().sort_values(ascending=False)
variables_correlacionadas = correlacion_con_target[correlacion_con_target > 0.1].index.tolist()
variables_correlacionadas.remove('precio') # Quitar el target

print(f"\n🔗 Variables con buena correlación con el precio:")
print(variables_correlacionadas)

# Visualizar selección

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Importancia por coeficientes

coef_importantes = coeficientes[coeficientes['Variable'].isin(variables_importantes)]
bars1 = axes[0].barh(coef_importantes['Variable'], coef_importantes['Impacto_Absoluto'])
axes[0].set_xlabel('Impacto Absoluto')
axes[0].set_title('Variables Seleccionadas por Impacto')
axes[0].grid(True, alpha=0.3, axis='x')

# Correlación con target

correlaciones = correlacion_con_target[variables_correlacionadas]
bars2 = axes[1].barh(correlaciones.index, correlaciones.values)
axes[1].set_xlabel('Correlación Absoluta con Precio')
axes[1].set_title('Variables por Correlación con Target')
axes[1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.show()
📝 ¿QUÉ ESTAMOS HACIENDO?
Seleccionamos solo las variables que realmente aportan valor.

🎯 ¿POR QUÉ SELECCIONAR VARIABLES?
Modelos más simples: Más fáciles de entender y mantener

Menos overfitting: Menos probabilidad de memorizar ruido

Más rápido: Menos variables = menos cómputo necesario

Más robusto: Generaliza mejor a nuevos datos

🔍 PARA QUÉ SIRVE
Encontrar el balance perfecto entre simplicidad y poder predictivo.

8. 🧪 EJEMPLO PRÁCTICO: PREDECIR CON NUEVOS DATOS
   python

# Predicción con nuevos datos

print("🧪 PREDICCIÓN CON NUEVOS DATOS")

# Crear una nueva casa

nueva_casa = pd.DataFrame({
'tamaño': [180],
'habitaciones': [4],
'antiguedad': [5],
'barrio_cod': [2],
'tiene_garage': [1],
'tiene_piscina': [0]
})

print("🏠 CARACTERÍSTICAS DE LA NUEVA CASA:")
for col, val in nueva_casa.iloc[0].items():
print(f" {col}: {val}")

# Preprocesar igual que los datos de entrenamiento

nueva_casa_scaled = scaler.transform(nueva_casa)

# Hacer predicción

precio_predicho = modelo_multi.predict(nueva_casa_scaled)[0]

print(f"\n💰 PRECIO PREDICHO: ${precio_predicho:,.0f}")

# Mostrar contribución de cada variable

contribuciones = modelo*multi.coef* \* nueva_casa_scaled[0]
contribucion_df = pd.DataFrame({
'Variable': X.columns,
'Contribución': contribuciones
}).sort_values('Contribución', key=abs, ascending=False)

print("\n📊 CONTRIBUCIÓN DE CADA VARIABLE AL PRECIO:")
for \_, row in contribucion_df.iterrows():
signo = "+" if row['Contribución'] > 0 else ""
print(f" {row['Variable']}: {signo}${row['Contribución']:,.0f}")

print(f" Precio base: ${modelo*multi.intercept*:,.0f}")
print(f" TOTAL: ${precio_predicho:,.0f}")
📝 ¿QUÉ ESTAMOS HACIENDO?
Usamos el modelo entrenado para predecir el precio de una casa nueva.

🎯 ¿CÓMO INTERPRETAR?
Cada variable contribuye positivamente o negativamente al precio final.

🔍 PARA QUÉ SIRVE
Entender no solo EL PRECIO, sino POR QUÉ el modelo da ese precio.

9. 📋 RESUMEN: FLUJO DE TRABAJO CON MÚLTIPLES VARIABLES
   markdown

# 🎯 RESUMEN: PASOS CON MÚLTIPLES VARIABLES

## 1. 🔍 ANÁLISIS EXPLORATORIO

- **Qué**: Estudiar cada variable individualmente
- **Por qué**: Entender relaciones y detectar problemas
- **Para qué**: Seleccionar variables prometedoras

## 2. 🏗️ PREPARACIÓN DE DATOS

- **Qué**: Estandarizar y dividir datos
- **Por qué**: Variables en diferentes escalas pueden distorsionar el modelo
- **Para qué**: Entrenamiento más estable y eficiente

## 3. 🧠 ENTRENAMIENTO DEL MODELO

- **Qué**: Entrenar con todas las variables
- **Por qué**: Ver el poder predictivo máximo
- **Para qué**: Establecer línea base de rendimiento

## 4. 📈 EVALUACIÓN Y ANÁLISIS

- **Qué**: Analizar coeficientes y rendimiento
- **Por qué**: Identificar variables importantes
- **Para qué**: Tomar decisiones informadas sobre qué variables mantener

## 5. 🎯 SELECCIÓN DE VARIABLES

- **Qué**: Elegir solo las variables más importantes
- **Por qué**: Evitar overfitting y simplificar el modelo
- **Para qué**: Modelo más robusto y interpretable

## 6. 🚀 MODELO FINAL

- **Qué**: Entrenar modelo final con variables seleccionadas
- **Por qué**: Balance óptimo entre simplicidad y poder predictivo
- **Para qué**: Implementación en producción

## 💡 REGLA DE ORO:

**"Empieza simple, añade complejidad solo si mejora significativamente el modelo"** 10. 🏆 CONCLUSIÓN FINAL
python

# Resumen visual del proceso

print("🎉 ¡PROCESO COMPLETADO!")

# Crear resumen visual

fig, ax = plt.subplots(figsize=(12, 8))

etapas = [
"1. Análisis Exploratorio\n(Entender datos)",
"2. Preparación de Datos\n(Estandarizar, dividir)",
"3. Entrenamiento Modelo\n(Todas las variables)",
"4. Evaluación y Análisis\n(Coeficientes, métricas)",
"5. Selección Variables\n(Mantener solo importantes)",
"6. Modelo Final\n(Optimizado y simple)"
]

colores = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']

for i, (etapa, color) in enumerate(zip(etapas, colores)):
y_pos = len(etapas) - i
ax.barh(y_pos, 1, color=color, alpha=0.8, edgecolor='black')
ax.text(0.5, y_pos, etapa, ha='center', va='center',
fontweight='bold', fontsize=11)

ax.set_xlim(0, 1)
ax.set_ylim(0, len(etapas) + 1)
ax.set_title('🚀 FLUJO DE TRABAJO CON MÚLTIPLES VARIABLES',
fontsize=14, fontweight='bold', pad=20)
ax.axis('off')

plt.tight_layout()
plt.show()

print("""
✅ LO QUE APRENDIMOS HOY:

🔍 ANÁLISIS:

- Cómo explorar múltiples variables
- Matrices de correlación
- Identificar relaciones

🏗️ CONSTRUCCIÓN:

- Preparar datos para múltiples variables
- Estandarización importante
- Entrenar modelos múltiples

📊 EVALUACIÓN:

- Interpretar coeficientes
- Identificar variables importantes
- Detectar overfitting

🎯 OPTIMIZACIÓN:

- Selección inteligente de variables
- Balance simplicidad-poder predictivo
- Modelos más robustos

💡 RECUERDA:
"Más variables no siempre es mejor"
"La simplicidad es la máxima sofisticación"
"Siempre pregunta: ¿Esta variable realmente ayuda?"
""")
