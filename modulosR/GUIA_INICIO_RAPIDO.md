# 🚀 GUÍA DE INICIO RÁPIDO

## Machine Learning con Python (IFCD093PO) - Versión Final

**¡Comienza tu viaje en Machine Learning en 5 pasos!**

---

## ⚡ Inicio Rápido (5 minutos)

### Paso 1: Verifica Python

```powershell
python --version
# Debe ser Python 3.8 o superior
```

### Paso 2: Crea un entorno virtual

```powershell
# Navega a la carpeta del curso
cd c:\Users\Jaf\Desktop\ML\IFCD093PO_final

# Crea el entorno
python -m venv venv

# Actívalo
.\venv\Scripts\activate
```

### Paso 3: Instala las librerías

```powershell
pip install -r requirements.txt
```

### Paso 4: Inicia Jupyter Notebook

```powershell
jupyter notebook
```

### Paso 5: ¡Abre el primer notebook!

En el navegador que se abre, haz clic en:

- `00_fundamentos_matematicos.ipynb`

---

## 📚 Ruta de Aprendizaje Recomendada

### 🟢 Nivel Principiante Absoluto (Semanas 1-4)

Comienza aquí si nunca has programado:

1. **Semana 1-2:** `00_fundamentos_matematicos.ipynb` (12h)

   - Álgebra y estadística básica
   - No te preocupes si parece difícil
   - Regresa cuando lo necesites

2. **Semana 2-4:** `01_introduccion_python.ipynb` (10h)
   - Python desde cero
   - Practica TODOS los ejercicios
   - Experimenta con el código

### 🟡 Nivel con Algo de Python (Semanas 1-6)

Si ya sabes Python básico:

1. **Semana 1:** Repaso rápido de `00_fundamentos_matematicos.ipynb`
2. **Semana 1-2:** `01_introduccion_python.ipynb` (enfócate en ejercicios)
3. **Semana 2-4:** `02_librerias_fundamentales.ipynb` (12h)
4. **Semana 5-6:** `03_introduccion_machine_learning.ipynb` (8h)

### 🔴 Nivel Intermedio (Semanas 1-12)

Si dominas Python y librerías:

Salta directo a:

- `03_introduccion_machine_learning.ipynb`
- Módulo 2 completo (Supervisado)
- Módulo 3 completo (No Supervisado)

---

## 🎯 Objetivos por Módulo

### Módulo 0: Fundamentos Matemáticos

**Al terminar podrás:**

- ✅ Entender ecuaciones lineales
- ✅ Calcular media, mediana, desviación estándar
- ✅ Trabajar con vectores y matrices
- ✅ Comprender el descenso de gradiente

### Módulo 1: Python y Herramientas

**Al terminar podrás:**

- ✅ Programar en Python con confianza
- ✅ Manipular datos con Pandas
- ✅ Hacer cálculos con NumPy
- ✅ Crear visualizaciones con Matplotlib
- ✅ Entender el flujo de ML

### Módulo 2: Aprendizaje Supervisado

**Al terminar podrás:**

- ✅ Construir modelos de clasificación
- ✅ Crear modelos de regresión
- ✅ Evaluar modelos correctamente
- ✅ Aplicar regularización
- ✅ Usar Random Forest y XGBoost

### Módulo 3: Aprendizaje No Supervisado

**Al terminar podrás:**

- ✅ Segmentar clientes con clustering
- ✅ Reducir dimensionalidad de datos
- ✅ Detectar anomalías
- ✅ Hacer análisis de cesta de compra

### Módulo 4: Proyectos Integradores

**Al terminar podrás:**

- ✅ Resolver problemas completos de ML
- ✅ Crear pipelines end-to-end
- ✅ Presentar resultados profesionalmente

---

## 💡 Consejos para el Éxito

### 📖 Cómo Estudiar Efectivamente

1. **No saltes capítulos** - Van en orden de dificultad
2. **Haz TODOS los ejercicios** - La práctica es esencial
3. **Experimenta** - Modifica el código y observa qué pasa
4. **Toma notas** - Añade tus propias celdas markdown
5. **Repite si es necesario** - Algunos conceptos necesitan tiempo

### ⏰ Gestión del Tiempo

**Si estudias 2 horas al día:**

- Módulo 0: 1 semana
- Módulo 1: 2-3 semanas
- Módulo 2: 4-5 semanas
- Módulo 3: 2-3 semanas
- Módulo 4: 1 semana
- **Total: ~3 meses**

**Si estudias 4 horas al día:**

- Módulo 0: 3 días
- Módulo 1: 1 semana
- Módulo 2: 2 semanas
- Módulo 3: 1.5 semanas
- Módulo 4: 2-3 días
- **Total: ~6 semanas**

### 🎓 Mejores Prácticas

#### Durante el Estudio:

- ✅ Ejecuta cada celda de código
- ✅ Lee los comentarios cuidadosamente
- ✅ Observa las visualizaciones
- ✅ Prueba valores diferentes
- ✅ Pregunta "¿por qué?" constantemente

#### Después de Cada Notebook:

- ✅ Revisa el resumen
- ✅ Haz los ejercicios adicionales
- ✅ Crea un proyecto mini personal
- ✅ Explica el concepto a alguien más

#### Si Te Atascas:

1. Vuelve a leer la explicación
2. Ejecuta el código paso a paso
3. Busca en la documentación oficial
4. Google el error específico
5. Pregunta en foros (Stack Overflow)
6. Toma un descanso y regresa después

---

## 🔧 Solución de Problemas Comunes

### Problema: "ModuleNotFoundError: No module named 'numpy'"

**Solución:**

```powershell
# Asegúrate de tener el entorno activado
.\venv\Scripts\activate

# Instala las librerías
pip install -r requirements.txt
```

### Problema: "Jupyter no se abre en el navegador"

**Solución:**

```powershell
# Copia la URL que aparece en la terminal
# Pégala manualmente en tu navegador
# Busca algo como: http://localhost:8888/?token=...
```

### Problema: "El código da error pero en el notebook funciona"

**Solución:**

- Reinicia el kernel: Menú → Kernel → Restart & Clear Output
- Ejecuta todas las celdas desde el principio

### Problema: "Los gráficos no se ven"

**Solución:**

```python
# Agrega al inicio del notebook
%matplotlib inline
```

### Problema: "Mi computadora va muy lenta"

**Solución:**

- Cierra otros programas
- Trabaja con datasets más pequeños al principio
- Considera usar Google Colab (gratis, en la nube)

---

## 📚 Recursos Complementarios

### Documentación Oficial:

- [Python](https://docs.python.org/3/)
- [NumPy](https://numpy.org/doc/)
- [Pandas](https://pandas.pydata.org/docs/)
- [Scikit-learn](https://scikit-learn.org/stable/)
- [Matplotlib](https://matplotlib.org/stable/contents.html)

### Tutoriales Interactivos:

- [Kaggle Learn](https://www.kaggle.com/learn) - Micro-cursos gratis
- [Google Colab](https://colab.research.google.com/) - Notebooks en la nube
- [DataCamp](https://www.datacamp.com/) - Cursos interactivos (algunos gratis)

### Videos Recomendados:

- [StatQuest](https://www.youtube.com/c/joshstarmer) - Explicaciones visuales
- [3Blue1Brown](https://www.youtube.com/c/3blue1brown) - Matemáticas visuales
- [Sentdex](https://www.youtube.com/c/sentdex) - Python y ML práctico

### Comunidades:

- [Stack Overflow](https://stackoverflow.com/) - Preguntas técnicas
- [Reddit r/learnmachinelearning](https://www.reddit.com/r/learnmachinelearning/)
- [Kaggle Forums](https://www.kaggle.com/discussions)

---

## 🎯 Plan de Estudio Sugerido (12 Semanas)

### Semanas 1-2: Fundamentos

- Lunes-Miércoles: Matemáticas (Módulo 0)
- Jueves-Viernes: Python básico (Módulo 1, parte 1)
- Fin de semana: Práctica y revisión

### Semanas 3-4: Herramientas

- Lunes-Jueves: Librerías (Módulo 1, partes 2-3)
- Viernes: Repaso general
- Fin de semana: Mini-proyecto personal

### Semanas 5-8: Aprendizaje Supervisado

- Cada semana: 1-2 notebooks del Módulo 2
- Foco en entender primero, optimizar después
- Fin de semana: Ejercicios extra

### Semanas 9-11: Aprendizaje No Supervisado

- Cada semana: 2 notebooks del Módulo 3
- Experimenta con diferentes algoritmos
- Compara resultados

### Semana 12: Proyectos Finales

- Completa los 4 proyectos integradores
- Crea tu portfolio personal
- ¡Celebra tu logro! 🎉

---

## ✅ Checklist de Progreso

### Módulo 0: Fundamentos Matemáticos

- [ ] Completado `00_fundamentos_matematicos.ipynb`
- [ ] Resueltos todos los ejercicios
- [ ] Entiendo el descenso de gradiente

### Módulo 1: Introducción

- [ ] Completado `01_introduccion_python.ipynb`
- [ ] Completado `02_librerias_fundamentales.ipynb`
- [ ] Completado `03_introduccion_machine_learning.ipynb`
- [ ] Puedo manipular DataFrames con confianza
- [ ] Hice un proyecto de ML simple

### Módulo 2: Supervisado

- [ ] Completados los 7 notebooks
- [ ] Entiendo todas las métricas
- [ ] Puedo construir un modelo de clasificación
- [ ] Puedo construir un modelo de regresión
- [ ] Sé cuándo usar regularización

### Módulo 3: No Supervisado

- [ ] Completados los 6 notebooks
- [ ] Puedo hacer clustering
- [ ] Entiendo PCA
- [ ] Sé interpretar resultados sin etiquetas

### Módulo 4: Proyectos

- [ ] Completados los 4 proyectos
- [ ] Tengo un portfolio de proyectos
- [ ] Puedo explicar mis soluciones
- [ ] ¡CERTIFICADO! 🎓

---

## 🎉 ¡Felicidades por Comenzar!

**Recuerda:**

- Machine Learning es un viaje, no un destino
- Todos empezamos sin saber nada
- La práctica constante es la clave
- Los errores son oportunidades de aprendizaje
- ¡Disfruta el proceso!

---

**¿Listo? ¡Abre `00_fundamentos_matematicos.ipynb` y comencemos! 🚀**
