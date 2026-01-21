"""
ChatBot de Python y Machine Learning usando ChatterBot
Instalación requerida: pip install chatterbot chatterbot-corpus
"""

from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer, ChatterBotCorpusTrainer
import warnings

# Suprimir advertencias de deprecación
warnings.filterwarnings('ignore')

# Crear el chatbot
chatbot = ChatBot(
    'PyMLBot',
    storage_adapter='chatterbot.storage.SQLStorageAdapter',
    database_uri='sqlite:///database.sqlite3',
    logic_adapters=[
        {
            'import_path': 'chatterbot.logic.BestMatch',
            'default_response': 'Lo siento, no tengo información sobre eso. ¿Puedes reformular tu pregunta?',
            'maximum_similarity_threshold': 0.70
        }
    ]
)

# Crear entrenadores
list_trainer = ListTrainer(chatbot)
corpus_trainer = ChatterBotCorpusTrainer(chatbot)

# ==================== ENTRENAMIENTO: PYTHON ====================
print("Entrenando con conocimientos de Python...")

python_conversations = [
    # Variables y tipos de datos
    "¿Qué es una variable en Python?",
    "Una variable es un contenedor para almacenar valores. Se crea cuando le asignas un valor, por ejemplo: x = 5",
    
    "¿Cuáles son los tipos de datos básicos en Python?",
    "Los tipos básicos son: int (enteros), float (decimales), str (cadenas), bool (booleanos), list (listas), tuple (tuplas), dict (diccionarios) y set (conjuntos)",
    
    # Listas
    "¿Qué es una lista en Python?",
    "Una lista es una colección ordenada y mutable de elementos. Se define con corchetes: mi_lista = [1, 2, 3, 'hola']",
    
    "¿Cómo agregar elementos a una lista?",
    "Puedes usar append() para agregar al final, insert() para una posición específica, o extend() para agregar múltiples elementos",
    
    # Diccionarios
    "¿Qué es un diccionario en Python?",
    "Un diccionario es una colección de pares clave-valor. Se define con llaves: mi_dict = {'nombre': 'Ana', 'edad': 25}",
    
    "¿Cómo acceder a valores de un diccionario?",
    "Puedes usar mi_dict['clave'] o mi_dict.get('clave'). El método get() es más seguro porque no genera error si la clave no existe",
    
    # Funciones
    "¿Qué es una función en Python?",
    "Una función es un bloque de código reutilizable que se define con def. Ejemplo: def saludar(nombre): return f'Hola {nombre}'",
    
    "¿Qué son los argumentos *args y **kwargs?",
    "*args permite pasar un número variable de argumentos posicionales, y **kwargs permite pasar argumentos con nombre como diccionario",
    
    # Clases
    "¿Qué es una clase en Python?",
    "Una clase es una plantilla para crear objetos. Define atributos y métodos. Se crea con la palabra clave class",
    
    "¿Qué es __init__ en Python?",
    "__init__ es el constructor de la clase, se ejecuta automáticamente cuando creas una instancia del objeto",
    
    # Módulos
    "¿Qué es un módulo en Python?",
    "Un módulo es un archivo .py que contiene código Python. Puedes importarlo con import para usar sus funciones y clases",
    
    "¿Cómo crear un módulo?",
    "Solo crea un archivo .py con funciones o clases, y luego impórtalo en otro archivo con import nombre_archivo",
    
    # Manejo de errores
    "¿Cómo manejar errores en Python?",
    "Usa try-except para capturar excepciones. Ejemplo: try: resultado = 10/0 except ZeroDivisionError: print('No se puede dividir por cero')",
    
    # Comprensiones
    "¿Qué es una list comprehension?",
    "Es una forma concisa de crear listas. Ejemplo: cuadrados = [x**2 for x in range(10)]",
    
    # Decoradores
    "¿Qué es un decorador en Python?",
    "Un decorador es una función que modifica el comportamiento de otra función. Se usa con @nombre_decorador antes de la función",
]

list_trainer.train(python_conversations)

# ==================== ENTRENAMIENTO: MACHINE LEARNING ====================
print("Entrenando con conocimientos de Machine Learning...")

ml_conversations = [
    # Conceptos básicos
    "¿Qué es Machine Learning?",
    "Machine Learning es una rama de la IA donde los algoritmos aprenden patrones de los datos sin ser programados explícitamente",
    
    "¿Cuáles son los tipos de Machine Learning?",
    "Los principales tipos son: Supervisado (con etiquetas), No Supervisado (sin etiquetas) y por Refuerzo (basado en recompensas)",
    
    "¿Qué es aprendizaje supervisado?",
    "Es cuando el modelo aprende de datos etiquetados. Ejemplos: clasificación y regresión",
    
    "¿Qué es aprendizaje no supervisado?",
    "Es cuando el modelo encuentra patrones en datos sin etiquetas. Ejemplos: clustering y reducción de dimensionalidad",
    
    # Algoritmos
    "¿Qué es regresión lineal?",
    "Es un algoritmo supervisado que predice valores continuos ajustando una línea a los datos. Fórmula: y = mx + b",
    
    "¿Qué es regresión logística?",
    "A pesar del nombre, es un algoritmo de clasificación que predice probabilidades usando la función sigmoide",
    
    "¿Qué son los árboles de decisión?",
    "Son modelos que toman decisiones dividiendo los datos en ramas basadas en características, formando una estructura de árbol",
    
    "¿Qué es Random Forest?",
    "Es un conjunto (ensemble) de múltiples árboles de decisión que votan para hacer predicciones más robustas",
    
    "¿Qué es K-Means?",
    "Es un algoritmo de clustering no supervisado que agrupa datos en K clusters basándose en similitud",
    
    "¿Qué es KNN?",
    "K-Nearest Neighbors clasifica un dato basándose en las K muestras más cercanas en el espacio de características",
    
    "¿Qué es SVM?",
    "Support Vector Machine busca el hiperplano óptimo que mejor separa las clases en el espacio de características",
    
    # Redes Neuronales
    "¿Qué es una red neuronal?",
    "Es un modelo inspirado en el cerebro, compuesto por capas de neuronas conectadas que aprenden patrones complejos",
    
    "¿Qué es Deep Learning?",
    "Es Machine Learning con redes neuronales profundas (muchas capas). Excelente para imágenes, texto y audio",
    
    # Evaluación
    "¿Qué es overfitting?",
    "Es cuando el modelo aprende demasiado de los datos de entrenamiento y no generaliza bien a datos nuevos",
    
    "¿Qué es underfitting?",
    "Es cuando el modelo es demasiado simple y no captura los patrones en los datos de entrenamiento",
    
    "¿Qué es la validación cruzada?",
    "Es una técnica que divide los datos en K partes, entrena en K-1 y valida en 1, rotando para evaluar mejor el modelo",
    
    "¿Qué es accuracy?",
    "Accuracy es la proporción de predicciones correctas sobre el total. Fórmula: (VP + VN) / Total",
    
    "¿Qué es precision?",
    "Precision mide cuántas predicciones positivas fueron correctas. Fórmula: VP / (VP + FP)",
    
    "¿Qué es recall?",
    "Recall mide cuántos positivos reales fueron detectados. Fórmula: VP / (VP + FN)",
    
    # Librerías
    "¿Qué es scikit-learn?",
    "Es la librería más popular de Python para Machine Learning clásico. Incluye algoritmos, preprocesamiento y métricas",
    
    "¿Qué es TensorFlow?",
    "Es una librería de Google para Deep Learning. Permite crear y entrenar redes neuronales complejas",
    
    "¿Qué es PyTorch?",
    "Es una librería de Facebook para Deep Learning, popular por su facilidad de uso y modo de ejecución dinámico",
    
    "¿Qué es pandas?",
    "Es una librería para manipulación y análisis de datos. Fundamental para preparar datos antes de entrenar modelos",
    
    "¿Qué es numpy?",
    "Es una librería para operaciones numéricas con arrays multidimensionales. Base de casi todas las librerías de ML",
    
    # Preprocesamiento
    "¿Qué es la normalización?",
    "Es escalar los datos a un rango específico (como 0-1) para que todas las características tengan la misma escala",
    
    "¿Qué es la estandarización?",
    "Es transformar los datos para que tengan media 0 y desviación estándar 1, usando z-score",
    
    "¿Qué es feature engineering?",
    "Es el proceso de crear nuevas características o transformar las existentes para mejorar el modelo",
]

list_trainer.train(ml_conversations)

# Entrenar con corpus en español (opcional)
print("Entrenando con corpus general...")
try:
    corpus_trainer.train("chatterbot.corpus.spanish")
except:
    print("No se pudo cargar el corpus en español")

print("\n" + "="*60)
print("¡ChatBot entrenado y listo!")
print("="*60)
print("\nEste bot puede responder preguntas sobre:")
print("  • Python (variables, funciones, clases, listas, etc.)")
print("  • Machine Learning (algoritmos, evaluación, librerías)")
print("\nEscribe 'salir' para terminar la conversación")
print("="*60 + "\n")

# ==================== CONVERSACIÓN ====================
def chat():
    """Función principal de conversación"""
    while True:
        try:
            user_input = input("Tú: ").strip()
            
            if user_input.lower() in ['salir', 'exit', 'quit', 'adios']:
                print("Bot: ¡Hasta luego! Espero haberte ayudado.")
                break
            
            if not user_input:
                continue
            
            response = chatbot.get_response(user_input)
            print(f"Bot: {response}\n")
            
        except KeyboardInterrupt:
            print("\n\nBot: ¡Hasta luego!")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue

# Iniciar la conversación
if __name__ == "__main__":
    chat()