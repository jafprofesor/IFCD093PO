// Referencias a elementos del DOM
const canvas = document.getElementById("kmeansCanvas");
const ctx = canvas.getContext("2d");
const kSlider = document.getElementById("kValue");
const kValueDisplay = document.getElementById("kValueDisplay");
const speedSlider = document.getElementById("animationSpeed");
const speedDisplay = document.getElementById("speedDisplay");
const generateBtn = document.getElementById("generateBtn");
const startBtn = document.getElementById("startBtn");
const pauseBtn = document.getElementById("pauseBtn");
const resetBtn = document.getElementById("resetBtn");
const explanationText = document.getElementById("explanationText");
const iterationCount = document.getElementById("iterationCount");
const convergenceStatus = document.getElementById("convergenceStatus");

// Deslizante de selección de dataset
const datasetTypeSlider = document.getElementById("datasetType");
const datasetTypeDisplay = document.getElementById("datasetTypeDisplay");

// Variables globales
let points = [];
const canvasWidth = canvas.width;
const canvasHeight = canvas.height;
const centerX = canvasWidth / 2;
const centerY = canvasHeight / 2;
let k = parseInt(kSlider.value);
let animationSpeed = parseInt(speedSlider.value);
let centroids = [];
let previousCentroids = [];
let clusters = [];
let currentDataset = "normal";
let animationId = null;
let isPaused = false;
let iteration = 0;
let hasConverged = false;
let animationTimeout = null;

// Colores para los clústeres
const clusterColors = [
  "#e74c3c",
  "#3498db",
  "#2ecc71",
  "#f39c12",
  "#9b59b6",
  "#1abc9c",
  "#d35400",
  "#34495e",
  "#7f8c8d",
  "#c0392b",
];

// Actualizar valores de los sliders
kSlider.addEventListener("input", () => {
  k = parseInt(kSlider.value);
  kValueDisplay.textContent = k;
});

// Evento para el deslizante de dataset
datasetTypeSlider.addEventListener("input", () => {
  const value = parseInt(datasetTypeSlider.value);
  let dataset;

  switch (value) {
    case 1:
      dataset = "normal";
      break;
    case 2:
      dataset = "moon";
      break;
    case 3:
      dataset = "circular";
      break;
    case 4:
      dataset = "spiral";
      break;
    case 5:
      dataset = "rings";
      break;
    case 6:
      dataset = "scattered";
      break;
    default:
      dataset = "normal";
  }

  setActiveDataset(dataset);
});

// Inicializar al cargar la página
window.addEventListener("load", () => {
  setActiveDataset("normal");
});

speedSlider.addEventListener("input", () => {
  animationSpeed = parseInt(speedSlider.value);
  speedDisplay.textContent = animationSpeed + " ms";
});

// Evento para el deslizante de selección de dataset
datasetTypeSlider.addEventListener("input", () => {
  const datasetValue = parseInt(datasetTypeSlider.value);
  let datasetType;
  let datasetName;

  switch (datasetValue) {
    case 1:
      datasetType = "normal";
      datasetName = "Clústeres Normales";
      break;
    case 2:
      datasetType = "moon";
      datasetName = "Clústeres en Media Luna";
      break;
    case 3:
      datasetType = "circular";
      datasetName = "Círculos Concéntricos";
      break;
    case 4:
      datasetType = "spiral";
      datasetName = "Espiral";
      break;
    case 5:
      datasetType = "rings";
      datasetName = "Anillos";
      break;
    case 6:
      datasetType = "scattered";
      datasetName = "Grupos Dispersos";
      break;
  }

  datasetTypeDisplay.textContent = datasetName;
  setActiveDataset(datasetType);
});

function setActiveDataset(dataset) {
  // Actualizar el valor del deslizante si es necesario
  let sliderValue;
  switch (dataset) {
    case "normal":
      sliderValue = 1;
      break;
    case "moon":
      sliderValue = 2;
      break;
    case "circular":
      sliderValue = 3;
      break;
    case "spiral":
      sliderValue = 4;
      break;
    case "rings":
      sliderValue = 5;
      break;
    case "scattered":
      sliderValue = 6;
      break;
  }

  // Solo actualizar si el valor es diferente para evitar bucles
  if (parseInt(datasetTypeSlider.value) !== sliderValue) {
    datasetTypeSlider.value = sliderValue;

    // Actualizar el texto del dataset
    const datasetName = getDatasetName();
    datasetTypeDisplay.textContent = datasetName;
  }

  currentDataset = dataset;
  resetAnimation();

  // Generar el dataset directamente aquí en lugar de llamar a generateDataset
  points = [];

  if (currentDataset === "normal") {
    generateNormalClusters();
  } else if (currentDataset === "moon") {
    generateMoonClusters();
  } else if (currentDataset === "circular") {
    generateCircularClusters();
  } else if (currentDataset === "spiral") {
    generateSpiralClusters();
  } else if (currentDataset === "rings") {
    generateRingsClusters();
  } else if (currentDataset === "scattered") {
    generateScatteredClusters();
  }

  drawPoints();
}

// Generar dataset según el tipo seleccionado
generateBtn.addEventListener("click", generateDataset);

// Controles de animación
startBtn.addEventListener("click", startAnimation);
pauseBtn.addEventListener("click", togglePause);
resetBtn.addEventListener("click", resetAnimation);

// Función para generar dataset según el tipo seleccionado
function generateDataset() {
  resetAnimation();
  points = [];

  if (currentDataset === "normal") {
    generateNormalClusters();
  } else if (currentDataset === "moon") {
    generateMoonClusters();
  } else if (currentDataset === "circular") {
    generateCircularClusters();
  } else if (currentDataset === "spiral") {
    generateSpiralClusters();
  } else if (currentDataset === "rings") {
    generateRingsClusters();
  } else if (currentDataset === "scattered") {
    generateScatteredClusters();
  }

  updateExplanation(
    `Dataset generado: ${getDatasetName()}. Ajusta el número de clústeres (K = ${k}) y pulsa 'Iniciar' para comenzar la simulación.`
  );
  drawPoints();
  startBtn.disabled = false;
}

function getDatasetName() {
  switch (currentDataset) {
    case "normal":
      return "Clústeres Normales";
    case "moon":
      return "Medias Lunas";
    case "circular":
      return "Círculos Concéntricos";
    case "spiral":
      return "Espiral";
    case "rings":
      return "Anillos";
    case "scattered":
      return "Grupos Dispersos";
    default:
      return "Clústeres Normales";
  }
}

// Generar clústeres normales (distribución gaussiana)
function generateNormalClusters() {
  const numClusters = Math.min(5, k); // Máximo 5 clústeres para este dataset
  const pointsPerCluster = 50;
  const clusterSpread = 40;
  const noise = 15; // Ruido adicional para hacer el dataset menos homogéneo

  // Posiciones de los centros de los clústeres
  const clusterCenters = [
    { x: centerX - 150, y: centerY - 150 },
    { x: centerX + 150, y: centerY - 150 },
    { x: centerX, y: centerY + 150 },
    { x: centerX - 200, y: centerY + 50 },
    { x: centerX + 200, y: centerY + 50 },
  ];

  for (let i = 0; i < numClusters; i++) {
    const center = clusterCenters[i];

    for (let j = 0; j < pointsPerCluster; j++) {
      // Distribución gaussiana alrededor del centro
      const x = center.x + gaussianRandom() * clusterSpread;
      const y = center.y + gaussianRandom() * clusterSpread;

      // Añadir ruido adicional a algunos puntos (20% de probabilidad)
      if (Math.random() < 0.2) {
        const nx = x + (Math.random() - 0.5) * noise * 2;
        const ny = y + (Math.random() - 0.5) * noise * 2;
        points.push({ x: nx, y: ny, cluster: -1 });
      } else {
        points.push({ x, y, cluster: -1 });
      }
    }
  }

  // Añadir algunos puntos de ruido completamente aleatorios
  const randomNoisePoints = 20;
  for (let i = 0; i < randomNoisePoints; i++) {
    const x = Math.random() * canvas.width;
    const y = Math.random() * canvas.height;
    points.push({ x, y, cluster: -1 });
  }
}

// Función para generar número aleatorio con distribución gaussiana
function gaussianRandom() {
  let u = 0,
    v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

// Generar clústeres en forma de media luna (estilo yin-yang)
function generateMoonClusters() {
  const numMoons = Math.min(2, k); // Máximo 2 medias lunas
  const pointsPerMoon = 150;

  // Parámetros para las medias lunas estilo yin-yang
  const radius = 150;
  const width = 40;
  const separation = 0; // Sin separación para que se toquen
  const noise = 10; // Ruido para hacer el dataset menos homogéneo

  // Añadir los puntos principales de las medias lunas
  for (let m = 0; m < numMoons; m++) {
    for (let i = 0; i < pointsPerMoon; i++) {
      // Ángulos para crear el patrón yin-yang
      const angle =
        m === 0
          ? Math.PI * (0.5 + 0.5 * Math.random()) // Primera media luna: 90° a 180°
          : Math.PI * 0.5 * Math.random(); // Segunda media luna: 0° a 90°

      const r = radius + width * (Math.random() - 0.5);

      let x, y;
      if (m === 0) {
        // Primera media luna (negra en yin-yang)
        x = centerX + r * Math.cos(angle);
        y = centerY + r * Math.sin(angle) + separation / 2;
        // Añadir ruido
        x += (Math.random() - 0.5) * noise;
        y += (Math.random() - 0.5) * noise;
      } else {
        // Segunda media luna (blanca en yin-yang)
        x = centerX + r * Math.cos(angle);
        y = centerY - r * Math.sin(angle) - separation / 2;
        // Añadir ruido
        x += (Math.random() - 0.5) * noise;
        y += (Math.random() - 0.5) * noise;
      }

      points.push({ x, y, cluster: -1 });
    }
  }

  // Añadir los "ojos" del yin-yang (pequeños círculos en el lado opuesto)
  const eyeRadius = 30;
  const eyePoints = 20;

  // Ojo en la parte negra (círculo blanco)
  const whiteEyeX = centerX - radius / 2;
  const whiteEyeY = centerY + radius / 2;

  // Ojo en la parte blanca (círculo negro)
  const blackEyeX = centerX + radius / 2;
  const blackEyeY = centerY - radius / 2;

  // Añadir puntos para los ojos
  for (let i = 0; i < eyePoints; i++) {
    const angle = Math.random() * 2 * Math.PI;
    const r = eyeRadius * Math.sqrt(Math.random());

    // Ojo blanco en la parte negra
    const wx = whiteEyeX + r * Math.cos(angle) + (Math.random() - 0.5) * noise;
    const wy = whiteEyeY + r * Math.sin(angle) + (Math.random() - 0.5) * noise;

    // Ojo negro en la parte blanca
    const bx = blackEyeX + r * Math.cos(angle) + (Math.random() - 0.5) * noise;
    const by = blackEyeY + r * Math.sin(angle) + (Math.random() - 0.5) * noise;

    points.push({ x: wx, y: wy, cluster: -1 });
    points.push({ x: bx, y: by, cluster: -1 });
  }
}

// Generar clústeres en forma de espiral
function generateSpiralClusters() {
  const numSpirals = Math.min(2, k); // Máximo 2 espirales
  const pointsPerSpiral = 150;

  for (let s = 0; s < numSpirals; s++) {
    const direction = s === 0 ? 1 : -1; // Dirección de la espiral (horario o antihorario)
    const offsetX = s === 0 ? -20 : 20; // Pequeño desplazamiento para separar las espirales

    for (let i = 0; i < pointsPerSpiral; i++) {
      // Parámetros de la espiral
      const t = (i / pointsPerSpiral) * 6 * Math.PI; // 3 vueltas completas
      const radius = 10 + t * 10; // Radio creciente

      // Coordenadas en espiral
      const x = centerX + offsetX + direction * radius * Math.cos(t);
      const y = centerY + radius * Math.sin(t);

      // Añadir ruido aleatorio
      const noise = 5;
      const nx = x + (Math.random() - 0.5) * noise;
      const ny = y + (Math.random() - 0.5) * noise;

      points.push({ x: nx, y: ny, cluster: -1 });
    }
  }
}

// Generar clústeres en forma de anillos (diferentes a los círculos concéntricos)
function generateRingsClusters() {
  const numRings = Math.min(3, k); // Máximo 3 anillos
  const pointsPerRing = 120;

  // Posiciones de los centros de los anillos
  const centers = [
    { x: centerX - 100, y: centerY - 100 },
    { x: centerX + 100, y: centerY - 50 },
    { x: centerX, y: centerY + 120 },
  ];

  // Radios para los anillos
  const radii = [70, 60, 80];

  for (let r = 0; r < numRings; r++) {
    const center = centers[r];
    const radius = radii[r];

    for (let i = 0; i < pointsPerRing; i++) {
      const angle = Math.random() * 2 * Math.PI;
      const noise = 10; // Grosor del anillo
      const r = radius + (Math.random() - 0.5) * noise;

      const x = center.x + r * Math.cos(angle);
      const y = center.y + r * Math.sin(angle);

      points.push({ x, y, cluster: -1 });
    }
  }
}

// Generar grupos dispersos (muy separados entre sí)
function generateScatteredClusters() {
  const numClusters = Math.min(6, k); // Máximo 6 grupos dispersos
  const pointsPerCluster = 40;

  // Posiciones muy separadas para los centros
  const centers = [
    { x: 100, y: 100 },
    { x: 500, y: 100 },
    { x: 100, y: 500 },
    { x: 500, y: 500 },
    { x: 150, y: 300 },
    { x: 450, y: 300 },
  ];

  for (let c = 0; c < numClusters; c++) {
    const center = centers[c];
    const spread = 30; // Dispersión pequeña dentro de cada grupo

    for (let i = 0; i < pointsPerCluster; i++) {
      const x = center.x + (Math.random() - 0.5) * spread;
      const y = center.y + (Math.random() - 0.5) * spread;

      points.push({ x, y, cluster: -1 });
    }
  }
}

// Generar clústeres circulares concéntricos
function generateCircularClusters() {
  const numCircles = Math.min(3, k); // Máximo 3 círculos concéntricos
  const pointsPerCircle = 120;
  const noise = 15; // Ruido para hacer el dataset menos homogéneo

  // Radios para los círculos concéntricos
  const radii = [80, 160, 240];
  const jitter = 15;

  for (let c = 0; c < numCircles; c++) {
    const radius = radii[c];

    for (let i = 0; i < pointsPerCircle; i++) {
      const angle = Math.random() * Math.PI * 2;
      const r = radius + (Math.random() * jitter * 2 - jitter);

      let x = centerX + Math.cos(angle) * r;
      let y = centerY + Math.sin(angle) * r;

      // Añadir ruido a algunos puntos (30% de probabilidad)
      if (Math.random() < 0.3) {
        x += (Math.random() - 0.5) * noise * 2;
        y += (Math.random() - 0.5) * noise * 2;
      }

      points.push({ x, y, cluster: -1 });
    }
  }

  // Añadir algunos puntos de ruido completamente aleatorios
  const randomNoisePoints = 25;
  for (let i = 0; i < randomNoisePoints; i++) {
    const x = centerX + (Math.random() - 0.5) * 400;
    const y = centerY + (Math.random() - 0.5) * 400;
    points.push({ x, y, cluster: -1 });
  }
}

// Inicializar centroides aleatoriamente
function initializeCentroids() {
  centroids = [];
  previousCentroids = [];

  // Seleccionar k puntos aleatorios como centroides iniciales
  const pointIndices = Array.from({ length: points.length }, (_, i) => i);

  // Mezclar los índices
  for (let i = pointIndices.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [pointIndices[i], pointIndices[j]] = [pointIndices[j], pointIndices[i]];
  }

  // Tomar los primeros k índices
  for (let i = 0; i < k; i++) {
    if (i < pointIndices.length) {
      const point = points[pointIndices[i]];
      centroids.push({ x: point.x, y: point.y });
      previousCentroids.push({ x: point.x, y: point.y });
    } else {
      // Si no hay suficientes puntos, crear centroides aleatorios
      centroids.push({
        x: Math.random() * canvasWidth,
        y: Math.random() * canvasHeight,
      });
      previousCentroids.push({
        x: Math.random() * canvasWidth,
        y: Math.random() * canvasHeight,
      });
    }
  }
}

// Asignar cada punto al centroide más cercano
function assignPointsToClusters() {
  // Reiniciar clusters
  clusters = Array(k)
    .fill()
    .map(() => []);

  // Asignar cada punto al centroide más cercano
  points.forEach((point, index) => {
    let minDist = Infinity;
    let closestCentroid = 0;

    centroids.forEach((centroid, i) => {
      const dist = distance(point, centroid);
      if (dist < minDist) {
        minDist = dist;
        closestCentroid = i;
      }
    });

    point.cluster = closestCentroid;
    clusters[closestCentroid].push(index);
  });
}

// Calcular la distancia euclidiana entre dos puntos
function distance(a, b) {
  return Math.sqrt(Math.pow(a.x - b.x, 2) + Math.pow(a.y - b.y, 2));
}

// Actualizar la posición de los centroides
function updateCentroids() {
  // Guardar centroides anteriores
  previousCentroids = centroids.map((c) => ({ x: c.x, y: c.y }));

  // Calcular nuevos centroides
  let maxMove = 0;

  clusters.forEach((cluster, i) => {
    if (cluster.length > 0) {
      let sumX = 0,
        sumY = 0;

      cluster.forEach((pointIndex) => {
        sumX += points[pointIndex].x;
        sumY += points[pointIndex].y;
      });

      const newX = sumX / cluster.length;
      const newY = sumY / cluster.length;

      // Calcular cuánto se movió el centroide
      const moveDistance = distance(
        { x: newX, y: newY },
        { x: centroids[i].x, y: centroids[i].y }
      );

      maxMove = Math.max(maxMove, moveDistance);

      // Actualizar centroide
      centroids[i] = { x: newX, y: newY };
    }
  });

  // Comprobar convergencia (si los centroides se mueven menos de 1 píxel)
  hasConverged = maxMove < 1;
  return maxMove;
}

// Función para dibujar los puntos y centroides
function drawPoints() {
  ctx.clearRect(0, 0, canvasWidth, canvasHeight);

  // Si estamos en la fase de asignación, dibujar líneas de distancia punto-centroide
  if (window.showDistances) {
    points.forEach((point) => {
      if (point.cluster !== -1) {
        const centroid = centroids[point.cluster];
        ctx.beginPath();
        ctx.moveTo(point.x, point.y);
        ctx.lineTo(centroid.x, centroid.y);
        ctx.strokeStyle = clusterColors[point.cluster % clusterColors.length];
        ctx.lineWidth = 1.5;
        ctx.globalAlpha = 0.5;
        ctx.stroke();
        ctx.globalAlpha = 1.0;
      }
    });
  }
  // Dibujar puntos
  points.forEach((point) => {
    ctx.beginPath();
    ctx.arc(point.x, point.y, 5, 0, Math.PI * 2);
    if (point.cluster !== -1) {
      ctx.fillStyle = clusterColors[point.cluster % clusterColors.length];
    } else {
      ctx.fillStyle = "#2c3e50";
    }
    ctx.fill();
  });

  // Dibujar centroides anteriores (línea punteada)
  previousCentroids.forEach((centroid, i) => {
    ctx.beginPath();
    ctx.arc(centroid.x, centroid.y, 8, 0, Math.PI * 2);
    ctx.strokeStyle = clusterColors[i % clusterColors.length];
    ctx.setLineDash([2, 2]);
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.setLineDash([]);
  });

  // Dibujar centroides actuales
  centroids.forEach((centroid, i) => {
    ctx.beginPath();
    ctx.arc(centroid.x, centroid.y, 8, 0, Math.PI * 2);
    ctx.fillStyle = clusterColors[i % clusterColors.length];
    ctx.fill();
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 2;
    ctx.stroke();
  });
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

    // Inicializar variables
    iteration = 0;
    hasConverged = false;

    // Inicializar centroides
    initializeCentroids();

    updateExplanation(
      `Iniciando K-means con K = ${k}. Fase 1: Inicialización de centroides aleatoriamente.`
    );
    iterationCount.textContent = `Iteración: ${iteration}`;
    convergenceStatus.textContent = `Estado: Inicializando`;

    drawPoints();
    animationTimeout = setTimeout(animateNextStep, animationSpeed);
  }
}

function animateNextStep() {
  if (isPaused) return;

  if (!hasConverged) {
    iteration++;
    iterationCount.textContent = `Iteración: ${iteration}`;

    // Fase 1: Asignar puntos a clústeres
    if (iteration === 1) {
      assignPointsToClusters();
      window.showDistances = true;
      updateExplanation(
        `Iteración ${iteration}: Asignando cada punto al centroide más cercano. Se muestran las distancias.`
      );
      convergenceStatus.textContent = `Estado: Asignando puntos`;
      drawPoints();
      // Mostrar distancias solo un instante
      setTimeout(() => {
        window.showDistances = false;
        drawPoints();
        // Continuar animación tras mostrar distancias
        animationTimeout = setTimeout(animateNextStep, animationSpeed);
      }, Math.max(400, animationSpeed * 0.7));
      return;
    }
    // Fase 2: Actualizar centroides
    else {
      const maxMove = updateCentroids();
      updateExplanation(
        `Iteración ${iteration}: Recalculando posición de centroides. Movimiento máximo: ${maxMove.toFixed(
          2
        )} píxeles.`
      );
      if (hasConverged) {
        convergenceStatus.textContent = `Estado: Convergencia alcanzada`;
        updateExplanation(
          `¡Convergencia alcanzada en ${iteration} iteraciones! Los centroides se han estabilizado.`
        );
        if (currentDataset === "normal" && k >= 5) {
          updateExplanation(
            `¡Convergencia alcanzada en ${iteration} iteraciones! K-means funciona bien con este tipo de clústeres convexos.`
          );
        } else if (currentDataset === "moon") {
          updateExplanation(
            `¡Convergencia alcanzada en ${iteration} iteraciones! Observa que K-means tiene dificultades con formas no convexas como las medias lunas.`
          );
        } else if (currentDataset === "circular") {
          updateExplanation(
            `¡Convergencia alcanzada en ${iteration} iteraciones! K-means no puede separar correctamente círculos concéntricos debido a su métrica de distancia euclidiana.`
          );
        }
        startBtn.disabled = false;
        pauseBtn.disabled = true;
      } else {
        convergenceStatus.textContent = `Estado: Actualizando centroides`;
        assignPointsToClusters();
      }
    }
    drawPoints();
    if (!hasConverged) {
      animationTimeout = setTimeout(animateNextStep, animationSpeed);
    }
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
  iteration = 0;
  hasConverged = false;
  centroids = [];
  previousCentroids = [];

  points.forEach((point) => (point.cluster = -1));

  startBtn.disabled = false;
  pauseBtn.disabled = true;
  pauseBtn.textContent = "Pausar";

  iterationCount.textContent = `Iteración: 0`;
  convergenceStatus.textContent = `Estado: No iniciado`;

  updateExplanation(
    "Simulación reiniciada. Ajusta los parámetros y pulsa 'Iniciar' para comenzar."
  );
  drawPoints();
}

function updateExplanation(text) {
  explanationText.innerHTML = `<p>${text}</p>`;
}

// Inicialización
function init() {
  kValueDisplay.textContent = k;
  speedDisplay.textContent = animationSpeed + " ms";
  // Establecer el dataset inicial
  setActiveDataset("normal");
  // Actualizar el texto del dataset
  datasetTypeDisplay.textContent = "Clústeres Normales";
  drawPoints();
}

// Iniciar cuando el DOM esté cargado
document.addEventListener("DOMContentLoaded", init);
