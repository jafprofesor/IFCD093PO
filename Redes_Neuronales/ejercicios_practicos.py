#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠 EJERCICIOS PRÁCTICOS - Redes Neuronales con Scikit-Learn

Este archivo contiene ejercicios prácticos progresivos para consolidar
los conocimientos sobre redes neuronales adquiridos en el notebook principal.
Cada ejercicio incluye un objetivo claro, descripción del dataset,
y pasos detallados para su implementación.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import load_iris, load_digits, load_wine, make_regression
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


def ejercicio_1_clasificacion_vinos():
    """
    🎯 EJERCICIO 1: Clasificación de Vinos
    
    Objetivo: Clasificar tipos de vino usando una red neuronal.
    
    Dataset: Wine dataset de scikit-learn
    - 178 muestras, 13 características
    - 3 clases de vino
    - Problema: Clasificación multiclase
    """
    print("=" * 60)
    print("🍷 EJERCICIO 1: CLASIFICACIÓN DE VINOS")
    print("=" * 60)
    
    # Cargar datos
    wine = load_wine()
    X, y = wine.data, wine.target
    
    print(f"Dataset Wine:")
    print(f"- {X.shape[0]} muestras, {X.shape[1]} características")
    print(f"- {len(np.unique(y))} clases: {wine.target_names}")
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Normalizar
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Entrenar modelo
    mlp = MLPClassifier(
        hidden_layer_sizes=(50, 30),
        activation='relu',
        solver='adam',
        alpha=0.01,
        learning_rate_init=0.001,
        max_iter=1000,
        random_state=42
    )
    
    print("\nEntrenando red neuronal...")
    mlp.fit(X_train_scaled, y_train)
    
    # Evaluar
    y_pred = mlp.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n📊 RESULTADOS:")
    print(f"Precisión: {accuracy:.3f} ({accuracy*100:.1f}%)")
    print(f"Iteraciones: {mlp.n_iter_}")
    
    # Reporte detallado
    print(f"\n📋 Reporte por clase:")
    print(classification_report(y_test, y_pred, target_names=wine.target_names))
    
    return accuracy


def ejercicio_2_optimizacion_grid_search():
    """
    🔧 EJERCICIO 2: Optimización con Grid Search
    
    Objetivo: Encontrar los mejores hiperparámetros para el dataset Digits.
    
    Dataset: Digits dataset
    - 1797 muestras, 64 características (8x8 píxeles)
    - 10 clases (dígitos 0-9)
    - Problema: Clasificación multiclase compleja
    """
    print("\n" + "=" * 60)
    print("🔢 EJERCICIO 2: OPTIMIZACIÓN CON GRID SEARCH")
    print("=" * 60)
    
    # Cargar datos
    digits = load_digits()
    X, y = digits.data, digits.target
    
    print(f"Dataset Digits:")
    print(f"- {X.shape[0]} muestras, {X.shape[1]} características")
    print(f"- {len(np.unique(y))} clases (dígitos 0-9)")
    
    # Dividir y normalizar
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Definir grilla de parámetros (versión simplificada)
    param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (128, 64)],
        'activation': ['relu', 'tanh'],
        'alpha': [0.001, 0.01]
    }
    
    print(f"\nProbando {len(param_grid['hidden_layer_sizes']) * len(param_grid['activation']) * len(param_grid['alpha'])} combinaciones...")
    
    # Grid Search
    mlp = MLPClassifier(solver='adam', max_iter=300, random_state=42)
    grid_search = GridSearchCV(
        mlp, param_grid, cv=3, scoring='accuracy', n_jobs=-1
    )
    
    grid_search.fit(X_train_scaled, y_train)
    
    print(f"\n🎯 Mejores parámetros:")
    for param, value in grid_search.best_params_.items():
        print(f"  - {param}: {value}")
    
    # Evaluar modelo optimizado
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n📊 RESULTADOS:")
    print(f"Precisión con parámetros por defecto: {grid_search.best_score_:.3f}")
    print(f"Precisión en test set: {accuracy:.3f}")
    
    return accuracy


def ejercicio_3_analisis_overfitting():
    """
    📊 EJERCICIO 3: Análisis de Overfitting
    
    Objetivo: Experimentar con diferentes configuraciones para inducir y detectar overfitting.
    
    Enfoque: Comparar modelos simples vs complejos
    """
    print("\n" + "=" * 60)
    print("📈 EJERCICIO 3: ANÁLISIS DE OVERFITTING")
    print("=" * 60)
    
    # Usar dataset Iris para experimentación rápida
    iris = load_iris()
    X, y = iris.data, iris.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Configuraciones para experimentar
    configs = [
        {"name": "Modelo Simple", "hidden_layer_sizes": (5,), "alpha": 0.1, "early_stopping": True},
        {"name": "Modelo Balanceado", "hidden_layer_sizes": (20,), "alpha": 0.01, "early_stopping": True},
        {"name": "Modelo Complejo", "hidden_layer_sizes": (100, 50), "alpha": 0.0001, "early_stopping": False}
    ]
    
    results = []
    
    for config in configs:
        print(f"\n🔧 Probando {config['name']}...")
        
        mlp = MLPClassifier(
            hidden_layer_sizes=config['hidden_layer_sizes'],
            activation='relu',
            solver='adam',
            alpha=config['alpha'],
            learning_rate_init=0.01,
            max_iter=500,
            early_stopping=config['early_stopping'],
            random_state=42
        )
        
        mlp.fit(X_train_scaled, y_train)
        
        train_pred = mlp.predict(X_train_scaled)
        test_pred = mlp.predict(X_test_scaled)
        
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        overfitting = train_acc - test_acc
        
        results.append({
            'name': config['name'],
            'train_acc': train_acc,
            'test_acc': test_acc,
            'overfitting': overfitting,
            'iterations': mlp.n_iter_
        })
        
        print(f"  - Entrenamiento: {train_acc:.3f}")
        print(f"  - Prueba: {test_acc:.3f}")
        print(f"  - Overfitting: {overfitting:.3f}")
        print(f"  - Iteraciones: {mlp.n_iter_}")
    
    print(f"\n📊 RESUMEN DE EXPERIMENTO:")
    for result in results:
        status = "🟢" if abs(result['overfitting']) < 0.05 else "🟡" if abs(result['overfitting']) < 0.15 else "🔴"
        print(f"{status} {result['name']}: Overfitting {result['overfitting']:.3f}")
    
    return results


def ejercicio_4_regresion_precios():
    """
    🏠 EJERCICIO 4: Regresión para Predicción de Precios
    
    Objetivo: Crear un sistema de predicción de precios usando MLPRegressor.
    
    Dataset: Generación sintética similar a Boston Housing
    - 506 muestras, 13 características
    - Problema: Regresión (precio continuo)
    """
    print("\n" + "=" * 60)
    print("🏠 EJERCICIO 4: REGRESIÓN - PREDICCIÓN DE PRECIOS")
    print("=" * 60)
    
    # Generar dataset sintético
    np.random.seed(42)
    X, y = make_regression(
        n_samples=506, n_features=13, noise=15, random_state=42
    )
    
    print(f"Dataset sintético:")
    print(f"- {X.shape[0]} muestras, {X.shape[1]} características")
    print(f"- Rango de precios: ${y.min():.1f}k - ${y.max():.1f}k")
    print(f"- Precio promedio: ${y.mean():.1f}k")
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Normalizar
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Entrenar regresor
    mlp_reg = MLPRegressor(
        hidden_layer_sizes=(100, 50),
        activation='relu',
        solver='adam',
        alpha=0.01,
        learning_rate_init=0.001,
        max_iter=1000,
        early_stopping=True,
        random_state=42
    )
    
    print("\nEntrenando regresor neuronal...")
    mlp_reg.fit(X_train_scaled, y_train)
    
    # Predicciones
    y_pred_train = mlp_reg.predict(X_train_scaled)
    y_pred_test = mlp_reg.predict(X_test_scaled)
    
    # Métricas
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_r2 = r2_score(y_test, y_pred_test)
    
    print(f"\n📊 RESULTADOS:")
    print(f"RMSE: ${np.sqrt(test_mse):.1f}k (error promedio)")
    print(f"MAE: ${test_mae:.1f}k (error mediano)")
    print(f"R²: {test_r2:.3f} ({test_r2*100:.1f}% varianza explicada)")
    print(f"Iteraciones: {mlp_reg.n_iter_}")
    
    # Interpretación
    if test_r2 > 0.7:
        status = "🟢 Excelente"
    elif test_r2 > 0.5:
        status = "🟡 Bueno"
    else:
        status = "🔴 Mejorable"
    
    print(f"\n💡 INTERPRETACIÓN: {status} modelo de regresión")
    
    # Ejemplo de predicción
    print(f"\n🎯 EJEMPLO DE PREDICCIÓN:")
    sample_idx = 0
    pred_price = y_pred_test[sample_idx]
    actual_price = y_test[sample_idx]
    error = abs(pred_price - actual_price)
    
    print(f"Precio predicho: ${pred_price:.1f}k")
    print(f"Precio real: ${actual_price:.1f}k")
    print(f"Error: ${error:.1f}k ({error/abs(actual_price)*100:.1f}%)")
    
    return test_r2


def ejercicio_5_comparacion_algoritmos():
    """
    🏆 EJERCICIO 5: Comparación Completa de Algoritmos
    
    Objetivo: Comparar redes neuronales con otros algoritmos de ML.
    
    Dataset: Digits
    Algoritmos: Neural Network, Random Forest, SVM, Logistic Regression, etc.
    """
    print("\n" + "=" * 60)
    print("🏆 EJERCICIO 5: COMPARACIÓN DE ALGORITMOS")
    print("=" * 60)
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    
    # Cargar datos
    digits = load_digits()
    X, y = digits.data, digits.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Configurar modelos
    models = {
        'Neural Network': MLPClassifier(
            hidden_layer_sizes=(128, 64), activation='relu',
            solver='adam', alpha=0.001, max_iter=500, random_state=42
        ),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Naive Bayes': GaussianNB(),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5)
    }
    
    print(f"Comparando {len(models)} algoritmos...")
    
    results = {}
    for name, model in models.items():
        print(f"Entrenando {name}...")
        model.fit(X_train_scaled, y_train)
        
        test_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, test_pred)
        
        results[name] = accuracy
        print(f"  → Precisión: {accuracy:.3f}")
    
    # Ranking
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n🏆 RANKING FINAL:")
    for i, (name, accuracy) in enumerate(sorted_results, 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        print(f"{medal} {i}. {name}: {accuracy:.3f}")
    
    return results


def ejercicio_6_proyecto_recomendacion():
    """
    🎬 EJERCICIO 6: Proyecto - Sistema de Recomendación Simple
    
    Objetivo: Crear un sistema que prediga calificaciones de películas.
    
    Dataset: Sintético con características de películas
    - Género, año, duración, presupuesto, etc.
    - Objetivo: Rating (1-5 estrellas)
    """
    print("\n" + "=" * 60)
    print("🎬 EJERCICIO 6: SISTEMA DE RECOMENDACIÓN")
    print("=" * 60)
    
    # Generar dataset sintético de películas
    np.random.seed(42)
    n_movies = 1000
    
    genres = ['Drama', 'Comedy', 'Action', 'Horror', 'Sci-Fi']
    movie_data = []
    ratings = []
    
    for i in range(n_movies):
        # Características de la película
        genre = np.random.choice(genres)
        year = np.random.randint(1990, 2023)
        duration = np.random.randint(80, 180)
        budget = np.random.randint(1, 100) * 1000000  # En dólares
        imdb_score = np.random.uniform(1, 10)
        
        # Generar rating realista
        base_rating = 3.0
        
        # Ajustes por género
        if genre in ['Drama', 'Sci-Fi']:
            base_rating += 0.5
        
        # Ajustes por año
        if year > 2010:
            base_rating += 0.3
        
        # Ajustes por duración
        if 90 <= duration <= 130:
            base_rating += 0.2
        
        # Agregar ruido
        final_rating = base_rating + np.random.normal(0, 0.8)
        final_rating = np.clip(final_rating, 1, 5)  # Limitar entre 1 y 5
        
        # Características numéricas para el modelo
        genre_encoding = genres.index(genre)
        movie_data.append([genre_encoding, year, duration, budget/1000000, imdb_score])
        ratings.append(final_rating)
    
    X = np.array(movie_data)
    y = np.array(ratings)
    
    print(f"Dataset generado:")
    print(f"- {len(X)} películas")
    print(f"- Características: Género, Año, Duración, Presupuesto(M$), IMDb")
    print(f"- Objetivo: Rating (1-5 estrellas)")
    print(f"- Rango de ratings: {y.min():.1f} - {y.max():.1f}")
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Normalizar
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Entrenar regresor
    mlp_recommender = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        alpha=0.01,
        learning_rate_init=0.001,
        max_iter=1000,
        random_state=42
    )
    
    print("\nEntrenando sistema de recomendación...")
    mlp_recommender.fit(X_train_scaled, y_train)
    
    # Evaluar
    y_pred = mlp_recommender.predict(X_test_scaled)
    
    from sklearn.metrics import mean_absolute_error, r2_score
    
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n📊 RESULTADOS:")
    print(f"MAE: {mae:.2f} estrellas (error promedio)")
    print(f"R²: {r2:.3f} ({r2*100:.1f}% varianza explicada)")
    
    # Ejemplo de predicción
    print(f"\n🎯 EJEMPLO DE RECOMENDACIÓN:")
    sample_idx = 0
    genre_enc, year, duration, budget, imdb = X_test[sample_idx]
    genre_name = genres[int(genre_enc)]
    
    pred_rating = y_pred[sample_idx]
    actual_rating = y_test[sample_idx]
    
    print(f"Película: {genre_name}, {int(year)}, {int(duration)}min, ${budget:.0f}M")
    print(f"Rating predicho: {pred_rating:.1f} estrellas")
    print(f"Rating real: {actual_rating:.1f} estrellas")
    print(f"Error: {abs(pred_rating - actual_rating):.2f} estrellas")
    
    return r2


def crear_dashboard_comparativo():
    """
    📊 Crear un dashboard comparativo de todos los ejercicios
    """
    print("\n" + "=" * 60)
    print("📊 DASHBOARD COMPARATIVO FINAL")
    print("=" * 60)
    
    print("Ejecutando todos los ejercicios para crear comparativa...")
    
    resultados = {}
    
    try:
        print("\n1. Clasificación de Vinos...")
        resultados['Wine Classification'] = ejercicio_1_clasificacion_vinos()
    except Exception as e:
        print(f"Error en ejercicio 1: {e}")
        resultados['Wine Classification'] = 0.0
    
    try:
        print("\n2. Optimización Grid Search...")
        resultados['Grid Search Optimization'] = ejercicio_2_optimizacion_grid_search()
    except Exception as e:
        print(f"Error en ejercicio 2: {e}")
        resultados['Grid Search Optimization'] = 0.0
    
    try:
        print("\n3. Análisis Overfitting...")
        ejercicio_3_analisis_overfitting()
        resultados['Overfitting Analysis'] = "Completado"
    except Exception as e:
        print(f"Error en ejercicio 3: {e}")
        resultados['Overfitting Analysis'] = "Error"
    
    try:
        print("\n4. Regresión Precios...")
        resultados['Price Regression'] = ejercicio_4_regresion_precios()
    except Exception as e:
        print(f"Error en ejercicio 4: {e}")
        resultados['Price Regression'] = 0.0
    
    try:
        print("\n5. Comparación Algoritmos...")
        resultados['Algorithm Comparison'] = ejercicio_5_comparacion_algoritmos()
    except Exception as e:
        print(f"Error en ejercicio 5: {e}")
        resultados['Algorithm Comparison'] = "Error"
    
    try:
        print("\n6. Sistema Recomendación...")
        resultados['Recommendation System'] = ejercicio_6_proyecto_recomendacion()
    except Exception as e:
        print(f"Error en ejercicio 6: {e}")
        resultados['Recommendation System'] = 0.0
    
    # Resumen final
    print("\n" + "=" * 60)
    print("🎉 RESUMEN FINAL DE EJERCICIOS")
    print("=" * 60)
    
    for nombre, resultado in resultados.items():
        if isinstance(resultado, float):
            status = "✅" if resultado > 0.8 else "🟡" if resultado > 0.6 else "🔴"
            print(f"{status} {nombre}: {resultado:.3f}")
        else:
            print(f"✅ {nombre}: {resultado}")
    
    print("\n🏆 ¡Felicitaciones! Has completado todos los ejercicios prácticos.")
    print("💡 Recomendación: Revisa los resultados y experimenta con diferentes parámetros.")


if __name__ == "__main__":
    """
    🚀 EJECUTOR PRINCIPAL DE EJERCICIOS
    
    Puedes ejecutar ejercicios individuales o todos juntos.
    
    Uso:
    - python ejercicios_practicos.py                    # Ejecutar todos
    - python ejercicios_practicos.py 1                  # Solo ejercicio 1
    - python ejercicios_practicos.py dashboard          # Dashboard comparativo
    """
    import sys
    
    if len(sys.argv) > 1:
        ejercicio = sys.argv[1]
        
        if ejercicio == "1":
            ejercicio_1_clasificacion_vinos()
        elif ejercicio == "2":
            ejercicio_2_optimizacion_grid_search()
        elif ejercicio == "3":
            ejercicio_3_analisis_overfitting()
        elif ejercicio == "4":
            ejercicio_4_regresion_precios()
        elif ejercicio == "5":
            ejercicio_5_comparacion_algoritmos()
        elif ejercicio == "6":
            ejercicio_6_proyecto_recomendacion()
        elif ejercicio.lower() == "dashboard":
            crear_dashboard_comparativo()
        else:
            print("Ejercicio no válido. Use: 1, 2, 3, 4, 5, 6, o 'dashboard'")
    else:
        crear_dashboard_comparativo()
    
    print("\n🎓 ¡Gracias por practicar con redes neuronales!")
    print("📚 Continúa aprendiendo con proyectos propios.")
