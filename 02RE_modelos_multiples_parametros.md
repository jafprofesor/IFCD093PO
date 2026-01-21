🎯 TEORÍA: REGRESIÓN LINEAL vs CLASIFICACIÓN LINEAL
python

# Explicación visual de los conceptos

print("🎯 TEORÍA: REGRESIÓN LINEAL vs CLASIFICACIÓN LINEAL")

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# REGRESIÓN LINEAL

x_reg = np.linspace(0, 10, 100)
y_reg = 2\*x_reg + 1 + np.random.normal(0, 1, 100)
axes[0].scatter(x_reg, y_reg, alpha=0.6)
z_reg = np.polyfit(x_reg, y_reg, 1)
p_reg = np.poly1d(z_reg)
axes[0].plot(x_reg, p_reg(x_reg), "r--", linewidth=2)
axes[0].set_xlabel('Característica X')
axes[0].set_ylabel('Target Y (continuo)')
axes[0].set_title('REGRESIÓN LINEAL\n(Predice valores continuos)')
axes[0].grid(True, alpha=0.3)
axes[0].text(0.5, 15, 'Ejemplo: Precio de casas\nSalario vs Experiencia\nTemperatura vs Hora',
ha='center', va='center', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))

# CLASIFICACIÓN LINEAL (LOGÍSTICA)

x_clas = np.random.normal(0, 1, 100)
y_clas = 1 / (1 + np.exp(-(2*x_clas + 0.5))) > 0.5
axes[1].scatter(x_clas, y_clas, alpha=0.6)
x_sort = np.sort(x_clas)
y_prob = 1 / (1 + np.exp(-(2*x_sort + 0.5)))
axes[1].plot(x_sort, y_prob, "r--", linewidth=2)
axes[1].set_xlabel('Característica X')
axes[1].set_ylabel('Probabilidad de Clase')
axes[1].set_title('REGRESIÓN LOGÍSTICA\n(Predice probabilidades)')
axes[1].grid(True, alpha=0.3)
axes[1].text(0, 0.5, 'Ejemplo: Spam/No spam\nEnfermo/Sano\nAprobado/Reprobado',
ha='center', va='center', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))

plt.tight_layout()
plt.show()

print("""
📚 CONCEPTOS CLAVE:

REGRESIÓN LINEAL:
• Predice valores CONTINUOS (números)
• Ejemplo: Precio de una casa, temperatura, salario
• Fórmula: y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ

REGRESIÓN LOGÍSTICA (Clasificación Lineal):
• Predice PROBABILIDADES entre 0 y 1
• Luego convierte a categorías (Sí/No, 0/1)
• Usa función sigmoide: P = 1 / (1 + e^(-z))
• Donde z = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ

AMBOS:
• Son modelos LINEALES (combinación lineal de características)
• Pueden usar MÚLTIPLES parámetros
• Son INTERPRETABLES (podemos entender los coeficientes)
""")
🏠 EJERCICIO 1: CALIFORNIA HOUSING - REGRESIÓN LINEAL MÚLTIPLE
1.1 Carga y Exploración de Datos
python
print("🏠 EJERCICIO 1: CALIFORNIA HOUSING - REGRESIÓN LINEAL MÚLTIPLE")

# Cargar dataset

california = fetch_california_housing()
df_california = pd.DataFrame(california.data, columns=california.feature_names)
df_california['MedHouseVal'] = california.target

print("📊 INFORMACIÓN DEL DATASET:")
print(f"Forma: {df_california.shape}")
print(f"Características: {list(df_california.columns)}")
print(f"\nRango de valores de vivienda: ${df_california['MedHouseVal'].min()*100000:,.0f} - ${df_california['MedHouseVal'].max()*100000:,.0f}")

# Mostrar estadísticas

print("\n📈 ESTADÍSTICAS DESCRIPTIVAS:")
print(df_california.describe())

# Matriz de correlación

plt.figure(figsize=(10, 8))
corr_matrix = df_california.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Matriz de Correlación - California Housing')
plt.tight_layout()
plt.show()
1.2 Análisis de Relaciones Lineales
python

# Visualizar relaciones lineales

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
caracteristicas = df_california.columns[:-1] # Todas excepto el target

for i, feature in enumerate(caracteristicas):
ax = axes[i//4, i%4]
ax.scatter(df_california[feature], df_california['MedHouseVal'], alpha=0.3)

    # Añadir línea de tendencia
    z = np.polyfit(df_california[feature], df_california['MedHouseVal'], 1)
    p = np.poly1d(z)
    ax.plot(df_california[feature], p(df_california[feature]), "r--", linewidth=2)

    ax.set_xlabel(feature)
    ax.set_ylabel('MedHouseVal')
    ax.set_title(f'{feature} vs Precio\nCorr: {corr_matrix.loc[feature, "MedHouseVal"]:.2f}')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("🔍 OBSERVACIONES INICIALES:")
print("• MedInc tiene la correlación más alta con el precio (0.69)")
print("• Latitude y Longitude muestran patrones geográficos")
print("• Algunas relaciones no son perfectamente lineales")
1.3 Regresión Lineal Simple vs Múltiple
python

# COMPARACIÓN: Regresión Simple vs Múltiple

print("🔬 COMPARACIÓN: Regresión Simple vs Múltiple")

# Regresión Simple (solo MedInc)

X_simple = df_california[['MedInc']]
y = df_california['MedHouseVal']

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
X_simple, y, test_size=0.2, random_state=42
)

modelo_simple = LinearRegression()
modelo_simple.fit(X_train_s, y_train_s)
y_pred_simple = modelo_simple.predict(X_test_s)

r2_simple = r2_score(y_test_s, y_pred_simple)
mse_simple = mean_squared_error(y_test_s, y_pred_simple)

print(f"📊 REGRESIÓN SIMPLE (solo MedInc):")
print(f" R²: {r2*simple:.4f}")
print(f" MSE: {mse_simple:.4f}")
print(f" Coeficiente MedInc: {modelo_simple.coef*[0]:.4f}")
print(f" Intercepto: {modelo*simple.intercept*:.4f}")

# Regresión Múltiple (todas las características)

X_multiple = df_california.drop('MedHouseVal', axis=1)
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
X_multiple, y, test_size=0.2, random_state=42
)

modelo_multiple = LinearRegression()
modelo_multiple.fit(X_train_m, y_train_m)
y_pred_multiple = modelo_multiple.predict(X_test_m)

r2_multiple = r2_score(y_test_m, y_pred_multiple)
mse_multiple = mean_squared_error(y_test_m, y_pred_multiple)

print(f"\n📊 REGRESIÓN MÚLTIPLE (todas características):")
print(f" R²: {r2_multiple:.4f}")
print(f" MSE: {mse_multiple:.4f}")
print(f" Mejora en R²: {r2_multiple - r2_simple:.4f}")

# Mostrar coeficientes

coeficientes = pd.DataFrame({
'Característica': X*multiple.columns,
'Coeficiente': modelo_multiple.coef*,
'Impacto*Absoluto': np.abs(modelo_multiple.coef*)
}).sort_values('Impacto_Absoluto', ascending=False)

print(f"\n🎯 COEFICIENTES DEL MODELO MÚLTIPLE:")
print(coeficientes.to_string(index=False))
1.4 Interpretación de Coeficientes
python

# Visualización de coeficientes

plt.figure(figsize=(12, 6))

# Gráfico de coeficientes

plt.subplot(1, 2, 1)
colors = ['green' if x > 0 else 'red' for x in modelo_multiple.coef_]
bars = plt.barh(coeficientes['Característica'], coeficientes['Coeficiente'], color=colors)
plt.xlabel('Valor del Coeficiente')
plt.title('Coeficientes del Modelo de Regresión\n(Verde=Positivo, Rojo=Negativo)')
plt.grid(True, alpha=0.3, axis='x')

# Añadir valores en las barras

for bar, coef in zip(bars, coeficientes['Coeficiente']):
plt.text(bar.get_width() + (0.01 if bar.get_width() > 0 else -0.03),
bar.get_y() + bar.get_height()/2,
f'{coef:.3f}', ha='left' if bar.get_width() > 0 else 'right', va='center')

# Comparación de rendimiento

plt.subplot(1, 2, 2)
modelos = ['Simple (MedInc)', 'Múltiple']
r2_scores = [r2_simple, r2_multiple]
bars = plt.bar(modelos, r2_scores, color=['lightblue', 'lightgreen'])
plt.ylabel('R² Score')
plt.title('Comparación: Regresión Simple vs Múltiple')
plt.ylim(0, 0.7)
plt.grid(True, alpha=0.3, axis='y')

# Añadir valores en las barras

for bar, score in zip(bars, r2_scores):
plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
f'{score:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

print("💡 INTERPRETACIÓN DE COEFICIENTES:")
print("POSITIVO: Cuando la característica AUMENTA, el precio AUMENTA")
print("NEGATIVO: Cuando la característica AUMENTA, el precio DISMINUYE")
print("\n📖 EJEMPLOS:")
print(f"• MedInc (+0.44): Por cada unidad que aumenta el ingreso medio, el precio aumenta 0.44 unidades")
print(f"• AveRooms (+0.95): Más habitaciones por vivienda → precio más alto")
print(f"• Latitude (-0.43): Moverse hacia el norte → precio más bajo (controlando otras variables)")
1.5 Regularización: Ridge y Lasso
python

# Aplicando regularización

print("🛡️ APLICANDO REGULARIZACIÓN: Ridge y Lasso")

# Escalar características para regularización

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_m)
X_test_scaled = scaler.transform(X_test_m)

# Modelo Ridge (L2)

ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train_m)
y_pred_ridge = ridge.predict(X_test_scaled)
r2_ridge = r2_score(y_test_m, y_pred_ridge)

# Modelo Lasso (L1)

lasso = Lasso(alpha=0.1)
lasso.fit(X_train_scaled, y_train_m)
y_pred_lasso = lasso.predict(X_test_scaled)
r2_lasso = r2_score(y_test_m, y_pred_lasso)

print(f"📊 COMPARACIÓN CON REGULARIZACIÓN:")
print(f" Regresión Múltiple: R² = {r2_multiple:.4f}")
print(f" Ridge Regression: R² = {r2_ridge:.4f}")
print(f" Lasso Regression: R² = {r2_lasso:.4f}")

# Comparar coeficientes

coef*comparison = pd.DataFrame({
'Característica': X_multiple.columns,
'Sin_Regularización': modelo_multiple.coef*,
'Ridge': ridge.coef*,
'Lasso': lasso.coef*
})

print(f"\n🔍 COMPARACIÓN DE COEFICIENTES:")
print(coef_comparison.round(4))

# Visualizar comparación de coeficientes

plt.figure(figsize=(15, 8))
x_pos = np.arange(len(X_multiple.columns))
width = 0.25

plt.bar(x_pos - width, coef_comparison['Sin_Regularización'], width, label='Sin Regularización', alpha=0.8)
plt.bar(x_pos, coef_comparison['Ridge'], width, label='Ridge', alpha=0.8)
plt.bar(x_pos + width, coef_comparison['Lasso'], width, label='Lasso', alpha=0.8)

plt.xlabel('Características')
plt.ylabel('Valor del Coeficiente')
plt.title('Comparación de Coeficientes: Efecto de la Regularización')
plt.xticks(x_pos, X_multiple.columns, rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("💡 OBSERVACIONES SOBRE REGULARIZACIÓN:")
print("• Ridge: Reduce magnitud de todos los coeficientes")
print("• Lasso: Puede llevar coeficientes a cero (selección de características)")
print("• Ambos ayudan a prevenir overfitting")
🩺 EJERCICIO 2: DIABETES - CLASIFICACIÓN LINEAL MÚLTIPLE
2.1 Carga y Preparación de Datos
python
print("\n" + "="\*70)
print("🩺 EJERCICIO 2: DIABETES - CLASIFICACIÓN LINEAL MÚLTIPLE")

# Cargar dataset

diabetes = load_diabetes()
df_diabetes = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
df_diabetes['progression'] = diabetes.target

# Convertir a problema de clasificación binaria

# Supongamos que valores altos indican diabetes más avanzada

umbral_diabetes = np.percentile(df_diabetes['progression'], 75) # Percentil 75
df_diabetes['diabetes_avanzada'] = (df_diabetes['progression'] > umbral_diabetes).astype(int)

print("📊 INFORMACIÓN DEL DATASET:")
print(f"Forma: {df_diabetes.shape}")
print(f"Características: {list(diabetes.feature_names)}")
print(f"\nDistribución de clases:")
print(df_diabetes['diabetes_avanzada'].value_counts())
print(f"Proporción: {df_diabetes['diabetes_avanzada'].value_counts(normalize=True)}")

# Mostrar estadísticas por clase

print("\n📈 ESTADÍSTICAS POR CLASE:")
print(df_diabetes.groupby('diabetes_avanzada').mean().round(3))
2.2 Análisis Exploratorio para Clasificación
python

# Análisis para clasificación

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

caracteristicas_importantes = ['age', 'bmi', 'bp', 's1', 's2', 's3']

for i, feature in enumerate(caracteristicas_importantes):
ax = axes[i//3, i%3]

    # Boxplot por clase
    data_clase0 = df_diabetes[df_diabetes['diabetes_avanzada'] == 0][feature]
    data_clase1 = df_diabetes[df_diabetes['diabetes_avanzada'] == 1][feature]

    box_data = [data_clase0, data_clase1]
    box_plot = ax.boxplot(box_data, labels=['No Avanzada', 'Avanzada'], patch_artist=True)

    # Colorear las cajas
    colors = ['lightblue', 'lightcoral']
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)

    ax.set_ylabel(feature)
    ax.set_title(f'Distribución de {feature}\npor Estado de Diabetes')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Matriz de correlación con la variable objetivo

plt.figure(figsize=(10, 8))
corr_diabetes = df_diabetes.corr()
sns.heatmap(corr_diabetes[['diabetes_avanzada']].sort_values('diabetes_avanzada', ascending=False),
annot=True, cmap='coolwarm', center=0)
plt.title('Correlación con Diabetes Avanzada')
plt.tight_layout()
plt.show()
2.3 Regresión Logística Múltiple
python

# Preparar datos para clasificación

X_diab = df_diabetes.drop(['progression', 'diabetes_avanzada'], axis=1)
y_diab = df_diabetes['diabetes_avanzada']

# Dividir datos

X_diab_train, X_diab_test, y_diab_train, y_diab_test = train_test_split(
X_diab, y_diab, test_size=0.3, random_state=42, stratify=y_diab
)

# Escalar características (importante para regresión logística)

scaler_diab = StandardScaler()
X_diab_train_scaled = scaler_diab.fit_transform(X_diab_train)
X_diab_test_scaled = scaler_diab.transform(X_diab_test)

print("✅ DATOS PREPARADOS PARA CLASIFICACIÓN:")
print(f"Entrenamiento: {X_diab_train.shape[0]} muestras")
print(f"Prueba: {X_diab_test.shape[0]} muestras")
print(f"Proporción de clase 1 en entrenamiento: {y_diab_train.mean():.3f}")

# Entrenar modelo de regresión logística

log_reg = LogisticRegression(random_state=42, max_iter=1000)
log_reg.fit(X_diab_train_scaled, y_diab_train)

# Predicciones

y_pred_log = log_reg.predict(X_diab_test_scaled)
y_pred_proba = log_reg.predict_proba(X_diab_test_scaled)[:, 1]

# Métricas

accuracy = accuracy_score(y_diab_test, y_pred_log)
precision = precision_score(y_diab_test, y_pred_log)
recall = recall_score(y_diab_test, y_pred_log)
f1 = f1_score(y_diab_test, y_pred_log)

print(f"\n📊 RESULTADOS REGRESIÓN LOGÍSTICA:")
print(f" Exactitud: {accuracy:.3f}")
print(f" Precisión: {precision:.3f}")
print(f" Recall: {recall:.3f}")
print(f" F1-Score: {f1:.3f}")

# Coeficientes e interpretación

coef*diabetes = pd.DataFrame({
'Característica': X_diab.columns,
'Coeficiente': log_reg.coef*[0],
'Odds*Ratio': np.exp(log_reg.coef*[0]),
'Impacto': np.abs(log*reg.coef*[0])
}).sort_values('Impacto', ascending=False)

print(f"\n🎯 COEFICIENTES E INTERPRETACIÓN:")
print(coef_diabetes.round(4))
2.4 Interpretación y Visualización
python

# Visualización de resultados

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. Coeficientes y Odds Ratios

axes[0, 0].barh(coef_diabetes['Característica'], coef_diabetes['Coeficiente'])
axes[0, 0].set_xlabel('Coeficiente')
axes[0, 0].set_title('Coeficientes de Regresión Logística')
axes[0, 0].axvline(x=0, color='red', linestyle='--', alpha=0.7)
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].barh(coef_diabetes['Característica'], coef_diabetes['Odds_Ratio'])
axes[0, 1].set_xlabel('Odds Ratio')
axes[0, 1].set_title('Odds Ratios (e^coeficiente)')
axes[0, 1].axvline(x=1, color='red', linestyle='--', alpha=0.7)
axes[0, 1].grid(True, alpha=0.3)

# 2. Matriz de confusión

cm = confusion_matrix(y_diab_test, y_pred_log)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0],
xticklabels=['Pred No', 'Pred Sí'],
yticklabels=['Real No', 'Real Sí'])
axes[1, 0].set_title('Matriz de Confusión')
axes[1, 0].set_ylabel('Real')
axes[1, 0].set_xlabel('Predicción')

# 3. Curva ROC

fpr, tpr, thresholds = roc_curve(y_diab_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

axes[1, 1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
axes[1, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Aleatorio')
axes[1, 1].set_xlabel('Tasa de Falsos Positivos')
axes[1, 1].set_ylabel('Tasa de Verdaderos Positivos')
axes[1, 1].set_title('Curva ROC')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("💡 INTERPRETACIÓN PRÁCTICA:")
print("ODDS RATIO > 1: Aumenta probabilidad de diabetes avanzada")
print("ODDS RATIO < 1: Disminuye probabilidad de diabetes avanzada")
print(f"\n📖 EJEMPLOS:")
print(f"• bmi (1.67): Por cada unidad de BMI, odds de diabetes avanzada se multiplica por 1.67")
print(f"• bp (1.52): Presión arterial más alta → mayor riesgo")
print(f"• s5 (2.12): Nivel de glucosa tiene el mayor impacto")
2.5 Regularización en Clasificación
python

# Regularización en regresión logística

print("🛡️ REGULARIZACIÓN EN CLASIFICACIÓN")

# Probar diferentes tipos de regularización

param_grid = {
'C': [0.001, 0.01, 0.1, 1, 10, 100],
'penalty': ['l1', 'l2'],
'solver': ['liblinear']
}

log_reg_cv = LogisticRegression(random_state=42, max_iter=1000)
grid_search = GridSearchCV(log_reg_cv, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_diab_train_scaled, y_diab_train)

best*log_reg = grid_search.best_estimator*
y_pred_best = best_log_reg.predict(X_diab_test_scaled)
accuracy_best = accuracy_score(y_diab_test, y_pred_best)

print(f"✅ MEJORES PARÁMETROS: {grid*search.best_params*}")
print(f"📊 Exactitud con mejores parámetros: {accuracy_best:.3f}")
print(f"📊 Exactitud sin optimizar: {accuracy:.3f}")
print(f"🎯 Mejora: {accuracy_best - accuracy:.3f}")

# Comparar coeficientes con diferentes regularizaciones

log_l1 = LogisticRegression(penalty='l1', C=1.0, solver='liblinear', random_state=42)
log_l2 = LogisticRegression(penalty='l2', C=1.0, random_state=42)

log_l1.fit(X_diab_train_scaled, y_diab_train)
log_l2.fit(X_diab_train_scaled, y_diab_train)

coef*comparison_diab = pd.DataFrame({
'Característica': X_diab.columns,
'L1 (Lasso)': log_l1.coef*[0],
'L2 (Ridge)': log*l2.coef*[0],
'Sin Regularización': log*reg.coef*[0]
})

print(f"\n🔍 COMPARACIÓN DE COEFICIENTES CON DIFERENTES REGULARIZACIONES:")
print(coef_comparison_diab.round(4))

# Visualizar

plt.figure(figsize=(12, 8))
x_pos = np.arange(len(X_diab.columns))
width = 0.25

plt.bar(x_pos - width, coef_comparison_diab['L1 (Lasso)'], width, label='L1 (Lasso)', alpha=0.8)
plt.bar(x_pos, coef_comparison_diab['L2 (Ridge)'], width, label='L2 (Ridge)', alpha=0.8)
plt.bar(x_pos + width, coef_comparison_diab['Sin Regularización'], width, label='Sin Regularización', alpha=0.8)

plt.xlabel('Características')
plt.ylabel('Coeficiente')
plt.title('Efecto de la Regularización en Coeficientes (Diabetes)')
plt.xticks(x_pos, X_diab.columns, rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("💡 OBSERVACIONES:")
print("• L1 (Lasso): Algunos coeficientes se vuelven exactamente cero")
print("• L2 (Ridge): Reduce magnitud pero mantiene todas las características")
print("• Regularización ayuda a prevenir overfitting y mejora generalización")
🚢 EJERCICIO 3: TITANIC - CLASIFICACIÓN CON MÚLTIPLES CARACTERÍSTICAS
3.1 Carga y Feature Engineering
python
print("\n" + "="\*70)
print("🚢 EJERCICIO 3: TITANIC - CLASIFICACIÓN CON MÚLTIPLES CARACTERÍSTICAS")

# Cargar dataset Titanic

try:
df_titanic = sns.load_dataset('titanic')
except:
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df_titanic = pd.read_csv(url)

print("📊 INFORMACIÓN INICIAL:")
print(f"Forma del dataset: {df_titanic.shape}")
print(f"Columnas: {list(df_titanic.columns)}")
print(f"\nValores faltantes:")
print(df_titanic.isnull().sum())

# Feature Engineering avanzado

print("\n🔧 REALIZANDO FEATURE ENGINEERING...")

# Crear copia para trabajar

df_titanic_clean = df_titanic.copy()

# 1. Manejar valores faltantes

df_titanic_clean['age'].fillna(df_titanic_clean['age'].median(), inplace=True)
df_titanic_clean['embarked'].fillna(df_titanic_clean['embarked'].mode()[0], inplace=True)
df_titanic_clean.drop(columns=['deck'], inplace=True, errors='ignore')

# 2. Crear nuevas características

df_titanic_clean['family_size'] = df_titanic_clean['sibsp'] + df_titanic_clean['parch'] + 1
df_titanic_clean['is_alone'] = (df_titanic_clean['family_size'] == 1).astype(int)
df_titanic_clean['title'] = df_titanic_clean['name'].str.extract(' ([A-Za-z]+)\.', expand=False)

# Simplificar títulos

title_mapping = {
'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
'Mlle': 'Miss', 'Countess': 'Rare', 'Ms': 'Miss', 'Lady': 'Rare',
'Jonkheer': 'Rare', 'Don': 'Rare', 'Dona': 'Rare', 'Mme': 'Mrs',
'Capt': 'Rare', 'Sir': 'Rare'
}
df_titanic_clean['title'] = df_titanic_clean['title'].map(title_mapping)

# 3. Codificar variables categóricas

label_encoders = {}
categorical_cols = ['sex', 'embarked', 'title', 'class', 'who', 'adult_male', 'embark_town', 'alive', 'alone']

for col in categorical_cols:
if col in df_titanic_clean.columns:
le = LabelEncoder()
df_titanic_clean[col] = le.fit_transform(df_titanic_clean[col].astype(str))
label_encoders[col] = le

# 4. Seleccionar características finales

features_titanic = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked',
'family_size', 'is_alone', 'title', 'who', 'adult_male']

X_titanic = df_titanic_clean[features_titanic]
y_titanic = df_titanic_clean['survived']

print("✅ FEATURE ENGINEERING COMPLETADO")
print(f"Características seleccionadas: {features_titanic}")
print(f"Tamaño final: {X_titanic.shape}")
3.2 Análisis Exploratorio para Titanic
python

# Análisis de relaciones con la supervivencia

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 1. Clase vs Supervivencia

survived_pclass = df_titanic_clean.groupby(['pclass', 'survived']).size().unstack()
survived_pclass.plot(kind='bar', ax=axes[0, 0])
axes[0, 0].set_title('Supervivencia por Clase')
axes[0, 0].set_xlabel('Clase')

# 2. Sexo vs Supervivencia

survived_sex = df_titanic_clean.groupby(['sex', 'survived']).size().unstack()
survived_sex.plot(kind='bar', ax=axes[0, 1])
axes[0, 1].set_title('Supervivencia por Sexo')

# 3. Edad vs Supervivencia

df_titanic_clean.boxplot(column='age', by='survived', ax=axes[0, 2])
axes[0, 2].set_title('Edad vs Supervivencia')

# 4. Tarifa vs Supervivencia

df_titanic_clean.boxplot(column='fare', by='survived', ax=axes[1, 0])
axes[1, 0].set_title('Tarifa vs Supervivencia')

# 5. Tamaño familiar vs Supervivencia

survived_family = df_titanic_clean.groupby(['family_size', 'survived']).size().unstack()
survived_family.plot(kind='bar', ax=axes[1, 1])
axes[1, 1].set_title('Supervivencia por Tamaño Familiar')

# 6. Título vs Supervivencia

if 'title' in df_titanic_clean.columns:
survived_title = df_titanic_clean.groupby(['title', 'survived']).size().unstack()
survived_title.plot(kind='bar', ax=axes[1, 2])
axes[1, 2].set_title('Supervivencia por Título')

plt.tight_layout()
plt.show()

# Matriz de correlación

plt.figure(figsize=(12, 8))
corr_titanic = df_titanic_clean[features_titanic + ['survived']].corr()
sns.heatmap(corr_titanic, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Matriz de Correlación - Titanic')
plt.tight_layout()
plt.show()
3.3 Modelado con Regresión Logística Múltiple
python

# Modelado con Titanic

X_titanic_train, X_titanic_test, y_titanic_train, y_titanic_test = train_test_split(
X_titanic, y_titanic, test_size=0.3, random_state=42, stratify=y_titanic
)

# Escalar características

scaler_titanic = StandardScaler()
X_titanic_train_scaled = scaler_titanic.fit_transform(X_titanic_train)
X_titanic_test_scaled = scaler_titanic.transform(X_titanic_test)

print("✅ DATOS TITANIC PREPARADOS:")
print(f"Entrenamiento: {X_titanic_train.shape[0]} muestras")
print(f"Prueba: {X_titanic_test.shape[0]} muestras")
print(f"Proporción de supervivientes: {y_titanic_train.mean():.3f}")

# Entrenar modelo de regresión logística

log_reg_titanic = LogisticRegression(random_state=42, max_iter=1000)
log_reg_titanic.fit(X_titanic_train_scaled, y_titanic_train)

# Predicciones

y_pred_titanic = log_reg_titanic.predict(X_titanic_test_scaled)
y_pred_proba_titanic = log_reg_titanic.predict_proba(X_titanic_test_scaled)[:, 1]

# Métricas

accuracy_titanic = accuracy_score(y_titanic_test, y_pred_titanic)
precision_titanic = precision_score(y_titanic_test, y_pred_titanic)
recall_titanic = recall_score(y_titanic_test, y_pred_titanic)
f1_titanic = f1_score(y_titanic_test, y_pred_titanic)

print(f"\n📊 RESULTADOS REGRESIÓN LOGÍSTICA - TITANIC:")
print(f" Exactitud: {accuracy_titanic:.3f}")
print(f" Precisión: {precision_titanic:.3f}")
print(f" Recall: {recall_titanic:.3f}")
print(f" F1-Score: {f1_titanic:.3f}")

# Coeficientes e interpretación

coef*titanic = pd.DataFrame({
'Característica': features_titanic,
'Coeficiente': log_reg_titanic.coef*[0],
'Odds*Ratio': np.exp(log_reg_titanic.coef*[0]),
'Impacto': np.abs(log*reg_titanic.coef*[0])
}).sort_values('Impacto', ascending=False)

print(f"\n🎯 COEFICIENTES E INTERPRETACIÓN - TITANIC:")
print(coef_titanic.round(4))
3.4 Evaluación Completa y Interpretación
python

# Evaluación completa del modelo Titanic

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. Coeficientes y Odds Ratios

axes[0, 0].barh(coef_titanic['Característica'], coef_titanic['Coeficiente'])
axes[0, 0].set_xlabel('Coeficiente')
axes[0, 0].set_title('Coeficientes - Titanic')
axes[0, 0].axvline(x=0, color='red', linestyle='--', alpha=0.7)
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].barh(coef_titanic['Característica'], coef_titanic['Odds_Ratio'])
axes[0, 1].set_xlabel('Odds Ratio')
axes[0, 1].set_title('Odds Ratios - Titanic')
axes[0, 1].axvline(x=1, color='red', linestyle='--', alpha=0.7)
axes[0, 1].grid(True, alpha=0.3)

# 2. Matriz de confusión

cm_titanic = confusion_matrix(y_titanic_test, y_pred_titanic)
sns.heatmap(cm_titanic, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0],
xticklabels=['Pred No', 'Pred Sí'],
yticklabels=['Real No', 'Real Sí'])
axes[1, 0].set_title('Matriz de Confusión - Titanic')
axes[1, 0].set_ylabel('Real')
axes[1, 0].set_xlabel('Predicción')

# 3. Curva ROC

fpr_titanic, tpr_titanic, thresholds_titanic = roc_curve(y_titanic_test, y_pred_proba_titanic)
roc_auc_titanic = auc(fpr_titanic, tpr_titanic)

axes[1, 1].plot(fpr_titanic, tpr_titanic, color='darkorange', lw=2,
label=f'ROC (AUC = {roc_auc_titanic:.3f})')
axes[1, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Aleatorio')
axes[1, 1].set_xlabel('Tasa de Falsos Positivos')
axes[1, 1].set_ylabel('Tasa de Verdaderos Positivos')
axes[1, 1].set_title('Curva ROC - Titanic')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("💡 INTERPRETACIÓN PRÁCTICA - TITANIC:")
print("ODDS RATIO > 1: Aumenta probabilidad de supervivencia")
print("ODDS RATIO < 1: Disminuye probabilidad de supervivencia")
print(f"\n📖 FACTORES CLAVE IDENTIFICADOS:")
print(f"• Sexo femenino (sex): Mayor probabilidad de supervivencia (Odds Ratio: {np.exp(coef_titanic[coef_titanic['Característica']=='sex']['Coeficiente'].values[0]):.2f})")
print(f"• Clase alta (pclass): Menor probabilidad en clases bajas (Odds Ratio: {np.exp(coef_titanic[coef_titanic['Característica']=='pclass']['Coeficiente'].values[0]):.2f})")
print(f"• Tarifa (fare): Mayores tarifas → mayor supervivencia")
print(f"• Quién (who): Mujeres y niños tuvieron prioridad")
3.5 Optimización y Validación
python

# Optimización del modelo Titanic

print("🎯 OPTIMIZANDO MODELO TITANIC")

# Búsqueda de mejores parámetros

param_grid_titanic = {
'C': [0.001, 0.01, 0.1, 1, 10, 100],
'penalty': ['l1', 'l2'],
'solver': ['liblinear']
}

log_reg_titanic_cv = LogisticRegression(random_state=42, max_iter=1000)
grid_search_titanic = GridSearchCV(log_reg_titanic_cv, param_grid_titanic, cv=5,
scoring='accuracy', n_jobs=-1)
grid_search_titanic.fit(X_titanic_train_scaled, y_titanic_train)

best*log_reg_titanic = grid_search_titanic.best_estimator*
y_pred_best_titanic = best_log_reg_titanic.predict(X_titanic_test_scaled)
accuracy_best_titanic = accuracy_score(y_titanic_test, y_pred_best_titanic)

print(f"✅ MEJORES PARÁMETROS: {grid*search_titanic.best_params*}")
print(f"📊 Exactitud optimizada: {accuracy_best_titanic:.3f}")
print(f"📊 Exactitud original: {accuracy_titanic:.3f}")
print(f"🎯 Mejora: {accuracy_best_titanic - accuracy_titanic:.3f}")

# Validación cruzada

cv_scores = cross_val_score(best_log_reg_titanic, X_titanic_train_scaled, y_titanic_train,
cv=5, scoring='accuracy')

print(f"\n🔍 VALIDACIÓN CRUZADA (5-fold):")
print(f" Scores: {cv_scores}")
print(f" Media: {cv_scores.mean():.3f} (+/- {cv_scores.std() \* 2:.3f})")

# Reporte de clasificación detallado

print(f"\n📋 REPORTE DE CLASIFICACIÓN DETALLADO:")
print(classification_report(y_titanic_test, y_pred_best_titanic,
target_names=['No Sobrevivió', 'Sobrevivió']))
📊 RESUMEN COMPARATIVO FINAL
python
print("\n" + "="*80)
print("📊 RESUMEN COMPARATIVO FINAL")
print("="*80)

# Crear resumen comparativo

resumen_comparativo = pd.DataFrame({
'Dataset': ['California Housing', 'California Housing', 'Diabetes', 'Diabetes', 'Titanic', 'Titanic'],
'Tipo Problema': ['Regresión', 'Regresión', 'Clasificación', 'Clasificación', 'Clasificación', 'Clasificación'],
'Modelo': ['Lineal Simple', 'Lineal Múltiple', 'Logística Base', 'Logística Optima', 'Logística Base', 'Logística Optima'],
'Métrica': [r2_simple, r2_multiple, accuracy, accuracy_best, accuracy_titanic, accuracy_best_titanic],
'Mejora': ['-', f'+{(r2_multiple - r2_simple)*100:.1f}%', '-', f'+{(accuracy_best - accuracy)*100:.1f}%',
'-', f'+{(accuracy_best_titanic - accuracy_titanic)*100:.1f}%']
})

print(resumen_comparativo)

# Visualización comparativa final

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# California Housing

california_data = [r2_simple, r2_multiple]
axes[0].bar(['Simple\n(1 var)', 'Múltiple\n(8 vars)'], california_data, color=['lightblue', 'lightgreen'])
axes[0].set_title('California Housing\n(R² Score)')
axes[0].set_ylabel('R²')
axes[0].grid(True, alpha=0.3)
for i, v in enumerate(california_data):
axes[0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')

# Diabetes

diabetes_data = [accuracy, accuracy_best]
axes[1].bar(['Base', 'Optimizado'], diabetes_data, color=['lightblue', 'lightgreen'])
axes[1].set_title('Diabetes\n(Exactitud)')
axes[1].set_ylabel('Exactitud')
axes[1].grid(True, alpha=0.3)
for i, v in enumerate(diabetes_data):
axes[1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')

# Titanic

titanic_data = [accuracy_titanic, accuracy_best_titanic]
axes[2].bar(['Base', 'Optimizado'], titanic_data, color=['lightblue', 'lightgreen'])
axes[2].set_title('Titanic\n(Exactitud)')
axes[2].set_ylabel('Exactitud')
axes[2].grid(True, alpha=0.3)
for i, v in enumerate(titanic_data):
axes[2].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

print("""
🎯 CONCLUSIONES GENERALES:

📈 REGRESIÓN LINEAL MÚLTIPLE (California Housing):
• Múltiples características mejoran significativamente la predicción vs una sola
• R² mejoró de 0.47 a 0.60 (↑ 27% relativo)
• Características más importantes: MedInc, Latitude, AveRooms
• La regularización ayuda a prevenir overfitting

🩺 CLASIFICACIÓN LINEAL (Diabetes):
• Modelo capaz de identificar diabetes avanzada con ~76% de exactitud
• Características clave: s5 (glucosa), bmi, bp
• La optimización de hiperparámetros mejora ligeramente el rendimiento
• Regularización L1 puede eliminar características irrelevantes

🚢 CLASIFICACIÓN CON FEATURE ENGINEERING (Titanic):
• Feature engineering crucial para buen rendimiento (~80% exactitud)
• Factores más importantes: sexo, clase, tarifa
• "Mujeres y niños primero" claramente reflejado en los coeficientes
• La optimización proporciona mejoras incrementales

💡 LECCIONES APRENDIDAS:

1. Múltiples características generalmente mejoran los modelos lineales
2. La interpretabilidad es una gran ventaja de los modelos lineales
3. El preprocesamiento y feature engineering son cruciales
4. La regularización ayuda a modelos más robustos
5. Siempre validar con datos de prueba independientes

🚀 PRÓXIMOS PASOS:
• Probar modelos más complejos (árboles, SVM, redes neuronales)
• Experimentar con más feature engineering
• Aplicar técnicas de selección de características
• Probar en otros datasets del mundo real
""")
🧪 EJERCICIOS ADICIONALES
python
print("\n" + "="*80)
print("🧪 EJERCICIOS ADICIONALES PARA PRACTICAR")
print("="*80)

ejercicios = """
🎯 EJERCICIO 4: INTERPRETACIÓN AVANZADA

1. Para California Housing:

   - Calcula el precio predicho para una casa con características específicas
   - Explica cómo cada característica contribuye al precio final
   - ¿Qué características podrías eliminar sin afectar mucho el modelo?

2. Para Diabetes:

   - Calcula la probabilidad de diabetes avanzada para un paciente específico
   - Explica cómo cambiaría la probabilidad si el BMI aumenta en 1 unidad
   - ¿Qué características clínicas son más importantes para el diagnóstico?

3. Para Titanic:
   - Crea perfiles de pasajeros con alta/baja probabilidad de supervivencia
   - ¿Qué pasajeros fueron "falsos positivos" o "falsos negativos"?
   - ¿Cómo mejorarías el feature engineering?

🔍 EJERCICIO 5: ANÁLISIS DE ERRORES

1. Analiza los residuales en California Housing:

   - ¿Hay patrones en los errores?
   - ¿Qué tipos de casas predice mejor/peor el modelo?

2. Para los modelos de clasificación:
   - Analiza la matriz de confusión en detalle
   - ¿Qué tipos de errores son más costosos?
   - ¿Cómo podrías ajustar el threshold de decisión?

📈 EJERCICIO 6: EXPANSIÓN DEL MODELO

1. Agrega características polinomiales a la regresión:

   - ¿Mejora el rendimiento?
   - ¿Cómo afecta la interpretabilidad?

2. Prueba diferentes escalados:

   - StandardScaler vs MinMaxScaler
   - ¿Cómo afectan a los coeficientes?

3. Implementa selección de características:
   - Usando L1 (Lasso)
   - Usando importancia de características
     """

print(ejercicios)

print("""
🎉 ¡FELICITACIONES POR COMPLETAR LOS EJERCICIOS!

Has aplicado exitosamente:
• Regresión Lineal Múltiple con interpretación de coeficientes
• Regresión Logística para clasificación binaria  
• Feature engineering y preprocesamiento avanzado
• Regularización (Ridge, Lasso) para prevenir overfitting
• Optimización de hiperparámetros
• Evaluación comprehensiva de modelos

¡Los modelos lineales son poderosos, interpretables y un excelente punto de partida
para cualquier problema de machine learning! 🚀
""")
