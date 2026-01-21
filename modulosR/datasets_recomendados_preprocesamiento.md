# Datasets Recomendados para Práctica de Preprocesamiento

## 1. **Heart Disease Dataset (Cleveland)**

**Fuente:** UCI Machine Learning Repository  
**Desafíos de preprocesamiento:**

- **Datos faltantes:** Varias columnas con valores perdidos (~20%)
- **Variables categóricas:** Chest pain type, thalassemia, etc.
- **Outliers:** Presión arterial sistólica y diastólica
- **Escalado:** Rango muy diferente entre variables (edad vs thalach)
- **URL:** https://archive.ics.uci.edu/dataset/45/heart+disease

**Indicaciones específicas:**

- Identificar patrones de datos faltantes (MCAR, MAR, MNAR)
- Aplicar diferentes estrategias de imputación
- Crear variables dummy para características categóricas
- Usar RobustScaler por presencia de outliers

---

## 2. **Ames Housing Dataset**

**Fuente:** Kaggle - House Prices  
**Desafíos de preprocesamiento:**

- **Alta dimensionalidad:** 80+ características
- **Datos categóricos:** many unique values (Neighborhood, SaleType)
- **Datos faltantes:** Múltiples patrones complejos
- **Skewness:** Variables target y predictores sesgados
- **URL:** https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data

**Indicaciones específicas:**

- Feature engineering: combinar características relacionadas
- Tratamiento especial para variables con muchos niveles categóricos
- Box-Cox transformation para variables sesgadas
- PCA para reducción de dimensionalidad

---

## 3. **Bank Marketing Dataset**

**Fuente:** UCI ML Repository - Bank Marketing  
**Desafíos de preprocesamiento:**

- **Desbalance de clases:** ~11% de respuestas positivas
- **Variables temporales:** Fechas que requieren ingeniería especial
- **Datos faltantes:** En 'pdays' (missing = no contactado antes)
- **Variables categóricas ordinales:** 'education', 'job'
- **URL:** https://archive.ics.uci.edu/dataset/222/bank+marketing

**Indicaciones específicas:**

- Crear variables derivadas de fechas (día de semana, mes)
- Manejar 'unknown' como categoría válida
- Técnicas para datos desbalanceados
- Encoding ordinal para variables ordinales

---

## 4. **Adult Census Income Dataset**

**Fuente:** UCI ML Repository  
**Desafíos de preprocesamiento:**

- **Espacios en blanco:** Strings con espacios adicionales
- **Inconsistencias:** Múltiples representaciones del mismo valor
- **Variables categóricas con many levels:** 'native-country'
- **Datos faltantes:** Representados como '?' string
- **URL:** https://archive.ics.uci.edu/dataset/2/adult

**Indicaciones específicas:**

- Limpiar strings (trim, case normalization)
- Group rare categories
- Crear variables binarias para características clave
- Analizar interacciones entre variables

---

## 5. **Wine Quality Dataset (Red + White)**

**Fuente:** UCI ML Repository  
**Desafíos de preprocesamiento:**

- **Dos datasets relacionados:** Red wine + White wine
- **Datos faltantes:** Muy pocos (~0.1%)
- **Outliers:** Valores extremos en todas las características
- **Multiclass classification:** 6-10 clases de calidad
- **URL:** https://archive.ics.uci.edu/dataset/186/wine+quality

**Indicaciones específicas:**

- Combinar datasets y crear variable 'wine_type'
- Técnicas robustas para outliers (Isolation Forest)
- Feature scaling por la naturaleza de las variables químicas
- Análisis de correlaciones múltiples

---

## 6. **Online Shoppers Purchasing Intention Dataset**

**Fuente:** UCI ML Repository  
**Desafíos de preprocesamiento:**

- **Datos mixtos:** Numéricos, categóricos, booleanos
- **Datos temporales:** Session times, bounce rates
- **Skewness:** Distribución muy sesgada de revenue-related features
- **Región/country:** Variable geográfica con many levels
- **URL:** https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset

**Indicaciones específicas:**

- Log transformation para variables sesgadas
- Categorización de variables continuas
- Análisis de comportamiento temporal
- Encoding geográfico por región

---

## 7. **Forest Cover Type Dataset**

**Fuente:** UCI ML Repository  
**Desafíos de preprocesamiento:**

- **Alta dimensionalidad:** 54 características
- **Variables categóricas binarias:** Aspect, soil types
- **Correlación alta:** Algunas variables muy correlacionadas
- **Desbalance:** Ciertos tipos de bosque menos frecuentes
- **URL:** https://archive.ics.uci.edu/dataset/54/covertype

**Indicaciones específicas:**

- Análisis de componentes principales
- Selección de características
- Handling de binary categorical variables
- Normalización por escala geográfica

---

## 8. **Google Play Store Apps Dataset**

**Fuente:** Kaggle  
**Desafíos de preprocesamiento:**

- **Datos sucios:** Precios con símbolos, reviews con texto
- **Inconsistencias:** Múltiples versiones de same app
- **Datos faltantes:** Ratings, número de installs
- **Tipos de datos mixtos:** Reviews (numeric + text)
- **URL:** https://www.kaggle.com/datasets/lava18/google-play-store-apps

**Indicaciones específicas:**

- Parsing de strings para extraer valores numéricos
- Feature engineering de texto (reviews)
- Manejo de duplicados
- Transformación de categorías (Free vs Paid)

---

## 9. **Airbnb Listings Dataset**

**Fuente:** Inside Airbnb  
**Desafíos de preprocesamiento:**

- **Geolocalización:** Coordenadas lat/lon
- **Datos faltantes:** Host-related information
- **Variables categóricas:** Neighbourhood groups, room types
- **Precios:** Outliers extremos, missing values
- **URL:** http://insideairbnb.com/get-the-data.html

**Indicaciones específicas:**

- Feature engineering de coordenadas geográficas
- Imputación basada en neighbourhood
- Treatment de outliers extremos en precios
- One-hot encoding para tipos de habitación

---

## 10. **Hotel Booking Dataset**

**Fuente:** Kaggle - Hotel Booking Demand  
**Desafíos de preprocesamiento:**

- **Datos temporales:** Fechas de arrival/departure
- **Customers recurrentes:** Booking history patterns
- **Cancelaciones:** Variable target muy desbalanceada
- **Group bookings:** Variables especiales para grupos
- **URL:** https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demandhttps://www.kaggle.com/datasets/mojtaba142/hotel-booking

**Indicaciones específicas:**

- Feature engineering de fechas
- Análisis de patrones de booking
- Técnicas para severe class imbalance
- Crear variables derivadas (lead time, stay duration)

---

## Recomendaciones Generales para Preprocesamiento:

### 🔧 **Herramientas principales:**

```python
# Imports recomendados
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif
```

### 📊 **Pipeline recomendado:**

1. **Exploración inicial:** info(), describe(), missing_data analysis
2. **Limpieza:** Fix data types, remove duplicates, handle inconsistencies
3. **Análisis exploratorio:** Visualizar distributions, correlations
4. **Feature engineering:** Create new features, transform existing ones
5. **Handling missing values:** Choose appropriate imputation strategy
6. **Encoding:** Transform categorical variables
7. **Scaling:** Normalize/standardize numerical features
8. **Feature selection:** Remove irrelevant/redundant features

### ⚠️ **Consejos importantes:**

- Siempre visualizar los datos antes de preprocesar
- Documentar todas las transformaciones aplicadas
- Crear pipelines reproducibles
- Validar que no hay data leakage
- Mantener el mismo preprocesamiento para train/test sets

### 🎯 **Métricas de evaluación del preprocesamiento:**

- Porcentaje de datos faltantes manejados
- Número de features creadas/eliminadas
- Correlaciones eliminadas
- Balance entre train/test mantenido
- Performance mejorada después del preprocesamiento
