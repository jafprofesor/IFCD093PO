// Elementos del DOM
const canvas = document.getElementById('dbscanCanvas');
const ctx = canvas.getContext('2d');
const epsilonSlider = document.getElementById('epsilon');
const minPointsSlider = document.getElementById('minPoints');
const epsilonValue = document.getElementById('epsilonValue');
const minPointsValue = document.getElementById('minPointsValue');
const generateBtn = document.getElementById('generateBtn');
const startBtn = document.getElementById('startBtn');
const pauseBtn = document.getElementById('pauseBtn');
const resetBtn = document.getElementById('resetBtn');
const explanationText = document.getElementById('explanationText');

// Variables globales
let points = [];
const canvasWidth = canvas.width;
const canvasHeight = canvas.height;
const centerX = canvasWidth / 2;
const centerY = canvasHeight / 2;
let epsilon = parseInt(epsilonSlider.value);
let minPoints = parseInt(minPointsSlider.value);
let clusters = [];
let noise = [];
let corePoints = [];
let borderPoints = [];
let animationId = null;
let isPaused = false;
let currentStep = 0;
let animationSpeed = 500; // ms entre pasos
let animationTimeout = null;

// Colores para visualización
const colors = [
    '#e74c3c', // rojo - núcleo
    '#3498db', // azul - frontera
    '#95a5a6'  // gris - ruido
];

// Actualizar valores de los sliders
epsilonSlider.addEventListener('input', () => {
    epsilon = parseInt(epsilonSlider.value);
    epsilonValue.textContent = epsilon;
    drawPoints();
});

minPointsSlider.addEventListener('input', () => {
    minPoints = parseInt(minPointsSlider.value);
    minPointsValue.textContent = minPoints;
    drawPoints();
});

// Generar dataset de anillos concéntricos
generateBtn.addEventListener('click', generateDataset);

// Controles de animación
startBtn.addEventListener('click', startAnimation);
pauseBtn.addEventListener('click', togglePause);
resetBtn.addEventListener('click', resetAnimation);

// Función para generar dataset de anillos concéntricos
function generateDataset() {
    resetAnimation();
    points = [];
    
    // Parámetros para los anillos
    const innerRadius = 80;
    const outerRadius = 200;
    const innerPoints = 100;
    const outerPoints = 200;
    const noise = 30;
    const jitter = 15;
    
    // Generar puntos para el anillo interior
    for (let i = 0; i < innerPoints; i++) {
        const angle = Math.random() * Math.PI * 2;
        const radius = innerRadius + (Math.random() * jitter * 2 - jitter);
        const x = centerX + Math.cos(angle) * radius;
        const y = centerY + Math.sin(angle) * radius;
        points.push({ x, y, cluster: -1 }); // -1 significa no asignado
    }
    
    // Generar puntos para el anillo exterior
    for (let i = 0; i < outerPoints; i++) {
        const angle = Math.random() * Math.PI * 2;
        const radius = outerRadius + (Math.random() * jitter * 2 - jitter);
        const x = centerX + Math.cos(angle) * radius;
        const y = centerY + Math.sin(angle) * radius;
        points.push({ x, y, cluster: -1 });
    }
    
    // Añadir puntos de ruido aleatorios
    for (let i = 0; i < noise; i++) {
        const x = Math.random() * canvasWidth;
        const y = Math.random() * canvasHeight;
        points.push({ x, y, cluster: -1 });
    }
    
    updateExplanation("Dataset generado: dos anillos concéntricos con ruido. Ajusta los parámetros epsilon (radio de vecindad) y minPoints (mínimo de vecinos) y pulsa 'Iniciar' para comenzar la simulación.");
    drawPoints();
    startBtn.disabled = false;
}

// Función para dibujar los puntos
function drawPoints() {
    ctx.clearRect(0, 0, canvasWidth, canvasHeight);
    
    // Dibujar círculo de epsilon alrededor del punto actual si estamos en animación
    if (currentStep > 0 && currentStep <= points.length) {
        const currentPoint = points[currentStep - 1];
        ctx.beginPath();
        ctx.arc(currentPoint.x, currentPoint.y, epsilon, 0, Math.PI * 2);
        ctx.strokeStyle = 'rgba(52, 152, 219, 0.3)';
        ctx.lineWidth = 2;
        ctx.stroke();
    }
    
    // Dibujar todos los puntos
    points.forEach((point, index) => {
        ctx.beginPath();
        ctx.arc(point.x, point.y, 5, 0, Math.PI * 2);
        
        // Colorear según el tipo de punto
        if (corePoints.includes(index)) {
            ctx.fillStyle = colors[0]; // Núcleo
        } else if (borderPoints.includes(index)) {
            ctx.fillStyle = colors[1]; // Frontera
        } else if (noise.includes(index)) {
            ctx.fillStyle = colors[2]; // Ruido
        } else {
            ctx.fillStyle = '#2c3e50'; // No clasificado
        }
        
        // Resaltar el punto actual en la animación
        if (index === currentStep - 1) {
            ctx.lineWidth = 2;
            ctx.strokeStyle = '#f39c12';
            ctx.stroke();
        }
        
        ctx.fill();
    });
}

// Implementación del algoritmo DBSCAN
function dbscan() {
    // Reiniciar variables
    clusters = [];
    noise = [];
    corePoints = [];
    borderPoints = [];
    let clusterIndex = 0;
    
    // Paso 1: Encontrar todos los puntos núcleo
    points.forEach((point, index) => {
        point.neighbors = findNeighbors(index);
        if (point.neighbors.length >= minPoints) {
            corePoints.push(index);
        }
    });
    
    // Paso 2: Formar clusters a partir de puntos núcleo
    corePoints.forEach((pointIndex) => {
        if (points[pointIndex].cluster === -1) {
            points[pointIndex].cluster = clusterIndex;
            let queue = [pointIndex];
            
            while (queue.length > 0) {
                let currentIndex = queue.shift();
                
                // Añadir vecinos no visitados al cluster
                points[currentIndex].neighbors.forEach((neighborIndex) => {
                    if (points[neighborIndex].cluster === -1) {
                        points[neighborIndex].cluster = clusterIndex;
                        
                        // Si es un punto núcleo, añadirlo a la cola
                        if (corePoints.includes(neighborIndex)) {
                            queue.push(neighborIndex);
                        } else {
                            borderPoints.push(neighborIndex);
                        }
                    }
                });
            }
            
            clusterIndex++;
        }
    });
    
    // Paso 3: Identificar puntos de ruido
    points.forEach((point, index) => {
        if (point.cluster === -1) {
            noise.push(index);
        }
    });
}

// Encontrar vecinos dentro de epsilon
function findNeighbors(pointIndex) {
    const neighbors = [];
    const point = points[pointIndex];
    
    points.forEach((otherPoint, otherIndex) => {
        if (pointIndex !== otherIndex) {
            const distance = Math.sqrt(
                Math.pow(point.x - otherPoint.x, 2) + 
                Math.pow(point.y - otherPoint.y, 2)
            );
            
            if (distance <= epsilon) {
                neighbors.push(otherIndex);
            }
        }
    });
    
    return neighbors;
}

// Animación paso a paso del algoritmo
function startAnimation() {
    if (points.length === 0) {
        updateExplanation("Primero debes generar un dataset.");
        return;
    }
    
    if (isPaused) {
        isPaused = false;
        pauseBtn.textContent = "Pausar";
        animateNextStep();
    } else {
        resetAnimation();
        startBtn.disabled = true;
        pauseBtn.disabled = false;
        
        // Reiniciar variables
        currentStep = 0;
        corePoints = [];
        borderPoints = [];
        noise = [];
        points.forEach(point => point.cluster = -1);
        
        updateExplanation("Iniciando simulación DBSCAN. Parámetros: Epsilon = " + epsilon + ", MinPoints = " + minPoints);
        animateNextStep();
    }
}

function animateNextStep() {
    if (isPaused) return;
    
    currentStep++;
    
    if (currentStep <= points.length) {
        // Fase 1: Identificar vecinos para cada punto
        const currentPoint = points[currentStep - 1];
        currentPoint.neighbors = findNeighbors(currentStep - 1);
        
        if (currentPoint.neighbors.length >= minPoints) {
            corePoints.push(currentStep - 1);
            updateExplanation(`Punto ${currentStep}: Es un punto NÚCLEO con ${currentPoint.neighbors.length} vecinos (>= ${minPoints}).`);
        } else if (currentPoint.neighbors.length > 0) {
            updateExplanation(`Punto ${currentStep}: No es un punto núcleo, tiene solo ${currentPoint.neighbors.length} vecinos (< ${minPoints}).`);
        } else {
            updateExplanation(`Punto ${currentStep}: No tiene vecinos dentro de epsilon (${epsilon}).`);
        }
        
        drawPoints();
        animationTimeout = setTimeout(animateNextStep, animationSpeed);
    } else if (currentStep === points.length + 1) {
        // Fase 2: Formar clusters
        updateExplanation("Formando clusters a partir de puntos núcleo...");
        let clusterIndex = 0;
        
        corePoints.forEach((pointIndex) => {
            if (points[pointIndex].cluster === -1) {
                points[pointIndex].cluster = clusterIndex;
                
                // Encontrar todos los puntos alcanzables desde este punto núcleo
                let queue = [pointIndex];
                while (queue.length > 0) {
                    let currentIndex = queue.shift();
                    
                    points[currentIndex].neighbors.forEach((neighborIndex) => {
                        if (points[neighborIndex].cluster === -1) {
                            points[neighborIndex].cluster = clusterIndex;
                            
                            if (corePoints.includes(neighborIndex)) {
                                queue.push(neighborIndex);
                            } else {
                                borderPoints.push(neighborIndex);
                            }
                        }
                    });
                }
                
                clusterIndex++;
            }
        });
        
        drawPoints();
        currentStep++;
        animationTimeout = setTimeout(animateNextStep, animationSpeed);
    } else if (currentStep === points.length + 2) {
        // Fase 3: Identificar puntos de ruido
        points.forEach((point, index) => {
            if (point.cluster === -1) {
                noise.push(index);
            }
        });
        
        updateExplanation(`Clasificación completada: ${corePoints.length} puntos núcleo, ${borderPoints.length} puntos frontera, ${noise.length} puntos de ruido.`);
        drawPoints();
        
        startBtn.disabled = false;
        pauseBtn.disabled = true;
    }
}

function togglePause() {
    isPaused = !isPaused;
    
    if (isPaused) {
        pauseBtn.textContent = "Continuar";
        clearTimeout(animationTimeout);
    } else {
        pauseBtn.textContent = "Pausar";
        animateNextStep();
    }
}

function resetAnimation() {
    clearTimeout(animationTimeout);
    isPaused = false;
    currentStep = 0;
    corePoints = [];
    borderPoints = [];
    noise = [];
    
    points.forEach(point => point.cluster = -1);
    
    startBtn.disabled = false;
    pauseBtn.disabled = true;
    pauseBtn.textContent = "Pausar";
    
    updateExplanation("Simulación reiniciada. Ajusta los parámetros y pulsa 'Iniciar' para comenzar.");
    drawPoints();
}

function updateExplanation(text) {
    explanationText.innerHTML = `<p>${text}</p>`;
}

// Inicialización
function init() {
    epsilonValue.textContent = epsilon;
    minPointsValue.textContent = minPoints;
    drawPoints();
}

// Iniciar cuando el DOM esté cargado
document.addEventListener('DOMContentLoaded', init);