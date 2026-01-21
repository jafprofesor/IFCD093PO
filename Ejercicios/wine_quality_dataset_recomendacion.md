# Wine Quality Dataset - El más simple para comenzar

## ¿Por qué Wine Quality es ideal para principiantes?

### ✅ **Ventajas para estudiantes:**

1. **Pocos datos faltantes** (~0.1%)
   - No complica con estrategias de imputación complejas
   - Los estudiantes pueden enfocarse en técnicas básicas

2. **Variables numéricas principalmente**
   - Solo 1 variable categórica (wine_type: red/white)
   - No requiere manejo complejo de encoding

3. **Distribución clara del target**
   - Variables target numéricas (calidad: 3-9)
   - Fácil de entender para regresión o clasificación

4. **Datasets de tamaño moderado**
   - ~1,500-5,000 registros (no abrumador)
   - Carga rápida para ejecutar ejemplos

5. **Outliers manejables**
   - Outliers claros y fáciles de identificar
   - Técnicas básicas de manejo son suficientes

6. **Dominio familiar**
   - Todos conocen el contexto del vino
   - Fácil interpretar las variables

---

## 🗂️ **Plan de Preprocesamiento Paso a Paso**

### **Paso 1: Exploración Inicial (Muy Simple)**
```python
# Cargar y explorar
df = pd.read_csv('winequality-red.csv')
print(f"Forma: {df.shape}")
print(df.info())
print(df.describe())
```

### **Paso 2: Manejo de Datos Faltantes (Mínimo)**
```python
# Verificar valores faltantes
print(df.isnull().sum())
# Probablemente solo algunos, fácil de manejar
```

### **Paso 3: Detección de Outliers (Básica)**
```python
# Usar IQR para detectar outliers simples
Q1 = df['fixed acidity'].quantile(0.25)
Q3 = df['fixed acidity'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['fixed acidity'] < Q1 - 1.5*IQR) | 
              (df['fixed acidity'] > Q3 + 1.5*IQR)]
```

### **Paso 4: Feature Engineering (Sencillo)**
```python
# Crear categorías de calidad
df['quality_category'] = df['quality'].apply(
    lambda x: 'low' if x <= 5 else 'medium' if x <= 7 else 'high'
)
```

### **Paso 5: Escalado Simple**
```python
# StandardScaler para todas las variables numéricas
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df.select_dtypes(include=[np.number]))
```

### **Paso 6: Preparación Final**
```python
# Combinar datasets si es necesario
df['wine_type'] = 0  # 0 para red, 1 para white
```

---

## 📋 **Ejemplo de Código Completo**

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Cargar datos
df_red = pd.read_csv('winequality-red.csv', sep=';')
df_white = pd.read_csv('winequality-white.csv', sep=';')

# 2. Combinar datasets
df_red['wine_type'] = 'red'
df_white['wine_type'] = 'white'
df = pd.concat([df_red, df_white], ignore_index=True)

# 3. Exploración básica
print(f"Dataset shape: {df.shape}")
print(f"Missing values: {df.isnull().sum().sum()}")
print(f"Quality distribution:")
print(df['quality'].value_counts().sort_index())

# 4. Detección simple de outliers
def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[column] < lower) | (df[column] > upper)]

# 5. Feature engineering básico
df['quality_category'] = df['quality'].apply(
    lambda x: 'low' if x <= 5 else 'medium' if x <= 7 else 'high'
)

# 6. Encoding simple
df['wine_type_encoded'] = (df['wine_type'] == 'white').astype(int)

# 7. Escalado
scaler = StandardScaler()
numerical_features = df.select_dtypes(include=[np.number]).columns
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# 8. Preparar para modelado
X = df.drop(['quality', 'wine_type', 'quality_category'], axis=1)
y = df['quality']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Ready to model: ✓")
```

---

## 📊 **Comparación de Complejidad con Otros Datasets**

| Dataset | Datos Faltantes | Categóricas | Outliers | Complejidad |
|---------|----------------|-------------|----------|-------------|
| **Wine Quality** | ~0.1% | 1 simple | Moderado | ⭐ Fácil |
| Heart Disease | ~20% | 3-4 complejas | Alto | ⭐⭐⭐ Medio |
| Ames Housing | ~30% | 15+ complejas | Alto | ⭐⭐⭐⭐ Difícil |
| Bank Marketing | ~5% | 5+ con many levels | Moderado | ⭐⭐⭐ Medio |
| Adult Census | ~5% | 6+ con '?' | Moderado | ⭐⭐ Medio |

---

## 🎯 **Secuencia de Aprendizaje Sugerida**

### **Semana 1: Wine Quality**
- Técnicas básicas de limpieza
- Manejo simple de outliers
- Escalado estándar
- Primer modelo (Random Forest)

### **Semana 2: Adult Census**
- Manejo de strings sucios
- Datos faltantes más complejos
- Encoding de variables categóricas
- Feature engineering

### **Semana 3: Heart Disease**
- Imputación avanzada
- Outliers más complejos
- Variables categóricas más desafiantes
- Modelos más sofisticados

### **Semana 4+: Datasets más complejos**
- Ames Housing
- Google Play Store
- Airbnb Listings

---

## 💡 **Consejos para el Instructor**

1. **Empezar simple**: Wine Quality permite enfocarse en conceptos básicos
2. **Progresión gradual**: Cada dataset añade un nivel de dificultad
3. **Explicar el "por qué"**: En cada paso, explicar la razón detrás de cada técnica
4. **Práctica con código**: Incluir ejercicios prácticos en cada notebook
5. **Comparar resultados**: Mostrar cómo mejora el rendimiento después de cada paso

## 🎓 **Resultados Esperados**

Los estudiantes aprenderán:
- ✅ Carga y exploración básica de datos
- ✅ Técnicas fundamentales de limpieza
- ✅ Manejo básico de outliers
- ✅ Escalado de variables
- ✅ Preparación de datasets
- ✅ Pipeline reproducible
- ✅ Evaluación de resultados

**Todo esto sin frustrarse con problemas complejos de datos sucios o faltantes.**