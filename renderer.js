// ==========================================
// CONFIGURACIÓN
// ==========================================
const video = document.getElementById('webcam');
const startBtn = document.getElementById('start-btn');
const videoContainer = document.getElementById('video-container');
const resultText = document.getElementById('result-text');
const statusText = document.getElementById('status-text');
const debugText = document.getElementById('debug-text');
const listaRespuestas = document.getElementById('lista-respuestas');

let modelo = null;
let clases = [];
let handLandmarker = null;
let lastVideoTime = -1;
let lastSpokenWord = "";
let lastSpokenTime = 0;
let contadorEstable = 0;
let ultimaPrediccion = "";
let rafId = null;
let lastFrameAt = 0;
let historialPredicciones = [];
let lastDisplayedWord = "";
let lastDetectionTime = 0;

const REPETICIONES_NECESARIAS = 14;  // Ni muy rápido, ni muy lento
const UMBRAL_CONFIANZA = 0.70;       // 70% de seguridad (evita confusiones)       
const FRAME_INTERVAL_MS = 66;
const MAX_HISTORIAL = 8;
const DETECCION_MANTENER_MS = 1200;

// ==========================================
// EMOJIS (AQUÍ ESTÁ LA MAGIA)
// ==========================================
const emojisSenas = {
    "Hola": "👋", "Buenos Dias": "☀️", "Buenas Tardes": "🌤️", "Buenas Noches": "🌙",
    "Como estas": "😊", "Bien": "👍", "Mal": "👎", "Regular": "😐", "Y tu": "🤔",
    "Adios": "👋", "Gracias": "🙏", "Por favor": "🤲", "De nada": "😊",
    "Mi nombres es": "👤", "Si": "✅", "No": "❌", "Agua": "💧", "Comer": "🍎", "Ayuda": "🆘"
};
function obtenerEmoji(sena) { return emojisSenas[sena] || "✋"; }

// ==========================================
// VOZ
// ==========================================
const synth = window.speechSynthesis;
let voices = [];
function cargarVoces() { voices = synth.getVoices(); }
cargarVoces();
if (speechSynthesis.onvoiceschanged !== undefined) {
    speechSynthesis.onvoiceschanged = cargarVoces;
}
function hablar(texto) {
    if (!texto || !synth) return;
    if (synth.speaking) synth.cancel();
    const utterance = new SpeechSynthesisUtterance(texto);
    utterance.lang = 'es-ES';
    utterance.rate = 0.9;
    const vozEspanol = voices.find(v => v.lang.includes('es'));
    if (vozEspanol) utterance.voice = vozEspanol;
    synth.speak(utterance);
}

// ==========================================
// BASE DE DATOS
// ==========================================
const baseDeDatos = {
    "Hola": {
        respuestas: [
            { texto: "Hola, ¿qué tal?", sena: "Hola" },
            { texto: "Buenos días", sena: "Buenos Dias" },
            { texto: "Buenas tardes", sena: "Buenas Tardes" }
        ]
    },
    "Como estas": {
        respuestas: [
            { texto: "Estoy bien, gracias", sena: "Bien" },
            { texto: "Estoy regular", sena: "Regular" },
            { texto: "Estoy mal", sena: "Mal" },
            { texto: "¿Y tú cómo estás?", sena: "Y tu" }
        ]
    },
    "Mi nombres es": {
        respuestas: [
            { texto: "Mi nombre es Gabriel", sena: "Mi nombres es" },
            { texto: "Mi nombre es Erika", sena: "Mi nombres es" }
        ]
    },
    "Buenos Dias": {
        respuestas: [
            { texto: "Buenos días", sena: "Buenos Dias" },
            { texto: "Hola", sena: "Hola" }
        ]
    },
    "Adios": {
        respuestas: [
            { texto: "Adiós", sena: "Adios" },
            { texto: "Nos vemos", sena: "Adios" },
            { texto: "Hasta luego", sena: "Adios" }
        ]
    }
};

// ==========================================
// INICIAR (COMPATIBLE CON PC Y TELÉFONO)
// ==========================================
async function iniciarTodo() {
    statusText.textContent = "Cargando modelo...";
    startBtn.disabled = true;
    startBtn.textContent = "⏳ Cargando...";
    historialPredicciones = [];
    contadorEstable = 0;
    ultimaPrediccion = "";
    lastDisplayedWord = "";

    try {
        let paquete;

        if (typeof require !== 'undefined') {
            const fs = require('fs');
            const path = require('path');
            const rutaCompleta = path.join(__dirname, 'modelo', 'modelo_completo.json');
            const contenido = fs.readFileSync(rutaCompleta, 'utf-8');
            paquete = JSON.parse(contenido);
        } else {
            const respuesta = await fetch("modelo_completo.json");
            if (!respuesta.ok) throw new Error("No se pudo cargar el modelo");
            paquete = await respuesta.json();
        }

        clases = paquete.clases || [];
        if (!clases.length) throw new Error("El modelo no tiene clases");

        const binaryString = atob(paquete.pesos);
        const bytes = new Uint8Array(binaryString.length);
        for (let i = 0; i < binaryString.length; i++) {
            bytes[i] = binaryString.charCodeAt(i);
        }
        const weightData = bytes.buffer;

        try {
            await tf.setBackend("webgl");
        } catch (error) {
            console.warn("TensorFlow.js WebGL no disponible, usando backend predeterminado", error);
        }
        await tf.ready();

        modelo = await tf.loadLayersModel(tf.io.fromMemory({
            modelTopology: paquete.modelo,
            weightSpecs: paquete.pesosEspecificaciones,
            weightData: weightData
        }));

        const vision = await window.FilesetResolver.forVisionTasks(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
        );

        try {
            handLandmarker = await window.HandLandmarker.createFromOptions(vision, {
                baseOptions: {
                    modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
                    delegate: "GPU"
                },
                runningMode: "VIDEO",
                numHands: 2
            });
        } catch (error) {
            console.warn("GPU no disponible, usando CPU", error);
            handLandmarker = await window.HandLandmarker.createFromOptions(vision, {
                baseOptions: {
                    modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
                    delegate: "CPU"
                },
                runningMode: "VIDEO",
                numHands: 2
            });
        }

        if (video.srcObject) {
            const tracks = video.srcObject.getTracks();
            tracks.forEach(track => track.stop());
        }

        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            throw new Error("Este dispositivo o navegador no permite acceder a la cámara");
        }

        let stream;
        try {
            stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    facingMode: "user",
                    width: { ideal: 320 },
                    height: { ideal: 240 },
                    frameRate: { ideal: 30, max: 30 }
                },
                audio: false
            });
        } catch (error) {
            console.warn("No se pudo iniciar la cámara con resolución optimizada", error);
            stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        }

        video.srcObject = stream;
        video.muted = true;
        video.playsInline = true;
        await video.play();
        lastVideoTime = -1;
        lastFrameAt = 0;
        rafId = requestAnimationFrame(predecirWebcam);

        videoContainer.classList.remove('hidden');
        startBtn.style.display = 'none';
        statusText.textContent = "Cámara activa";

        mostrarRespuestas("__generales__");

    } catch (error) {
        console.error("Error:", error);
        statusText.textContent = "Error: " + (error?.message || "desconocido");
        startBtn.disabled = false;
        startBtn.textContent = "🎥 Encender Cámara";
        if (video.srcObject) {
            video.srcObject.getTracks().forEach(track => track.stop());
            video.srcObject = null;
        }
    }
}

// ==========================================
// RESPUESTAS
// ==========================================
function mostrarRespuestas(pregunta) {
    listaRespuestas.innerHTML = '';
    let entrada = null;
    for (const key in baseDeDatos) {
        if (key.toLowerCase() === pregunta.toLowerCase()) {
            entrada = baseDeDatos[key];
            break;
        }
    }

    const respuestasAMostrar = entrada ? entrada.respuestas : [
        { texto: "Sí", sena: "Si" },
        { texto: "No", sena: "No" },
        { texto: "Gracias", sena: "Gracias" },
        { texto: "Por favor", sena: "Por favor" },
        { texto: "De nada", sena: "De nada" },
        { texto: "Hola", sena: "Hola" }
    ];

    respuestasAMostrar.forEach(resp => {
        const badge = document.createElement('div');
        badge.className = 'respuesta-btn bg-white rounded-2xl p-4 cursor-pointer pop flex items-center gap-4';
        
        const emoji = obtenerEmoji(resp.sena);

        badge.innerHTML = `
            <div class="w-12 h-12 rounded-full bg-blue-100 flex items-center justify-center text-2xl">
                ${emoji}
            </div>
            <div class="flex-1">
                <div class="font-bold text-gray-800 text-lg">"${resp.texto}"</div>
                <div class="text-xs text-blue-500 mt-1">Seña: <span class="font-semibold">${resp.sena}</span></div>
            </div>
        `;
        badge.addEventListener('click', () => hablar(resp.texto));
        listaRespuestas.appendChild(badge);
    });
}

// ==========================================
// NORMALIZAR Y PREDECIR
// ==========================================
function normalizeLandmarks(landmarks) {
    const wrist = landmarks[0];
    return landmarks.map(p => ({
        x: p.x - wrist.x,
        y: p.y - wrist.y,
        z: p.z - wrist.z
    }));
}

function extraerVectorDosManos(resultados) {
    let vector = [];
    if (resultados.landmarks && resultados.landmarks.length >= 1) {
        const n = normalizeLandmarks(resultados.landmarks[0]);
        vector = n.flatMap(p => [p.x, p.y, p.z]);
    } else {
        vector = new Array(63).fill(0);
    }
    if (resultados.landmarks && resultados.landmarks.length >= 2) {
        const n = normalizeLandmarks(resultados.landmarks[1]);
        vector = vector.concat(n.flatMap(p => [p.x, p.y, p.z]));
    } else {
        vector = vector.concat(new Array(63).fill(0));
    }
    return vector;
}

async function predecirWebcam() {
    if (!handLandmarker || !modelo || !video) return;
    if (video.readyState < 2) {
        rafId = requestAnimationFrame(predecirWebcam);
        return;
    }

    const ahora = performance.now();
    if (ahora - lastFrameAt < FRAME_INTERVAL_MS || video.currentTime === lastVideoTime) {
        rafId = requestAnimationFrame(predecirWebcam);
        return;
    }

    lastFrameAt = ahora;
    lastVideoTime = video.currentTime;

    try {
        const results = handLandmarker.detectForVideo(video, ahora);
        if (results && results.landmarks && results.landmarks.length > 0) {
            const vector = extraerVectorDosManos(results);
            const input = tf.tensor2d([vector], [1, vector.length]);
            let prediccion;
            let datos;
            try {
                prediccion = modelo.predict(input);
                if (Array.isArray(prediccion)) {
                    throw new Error("El modelo debe devolver una sola salida");
                }
                datos = await prediccion.data();
            } finally {
                input.dispose();
                if (Array.isArray(prediccion)) {
                    prediccion.forEach(tensor => tensor.dispose());
                } else if (prediccion) {
                    prediccion.dispose();
                }
            }

            let maxIdx = 0;
            let maxProb = datos[0] || 0;
            for (let i = 1; i < datos.length; i++) {
                if ((datos[i] || 0) > maxProb) {
                    maxProb = datos[i];
                    maxIdx = i;
                }
            }

            debugText.textContent = clases.map((c, i) => `${c}: ${(datos[i] * 100).toFixed(0)}%`).join(" | ");
            if (maxProb >= UMBRAL_CONFIANZA) {
                historialPredicciones.push(clases[maxIdx] || "---");
                if (historialPredicciones.length > MAX_HISTORIAL) historialPredicciones.shift();
            } else {
                historialPredicciones = [];
            }
        } else {
            historialPredicciones = [];
            debugText.textContent = "Sin manos detectadas";
        }

        const conteo = new Map();
        for (const palabra of historialPredicciones) {
            conteo.set(palabra, (conteo.get(palabra) || 0) + 1);
        }

        let detectedWord = "---";
        let maxRepeticiones = 0;
        for (const [palabra, cantidad] of conteo) {
            if (cantidad > maxRepeticiones) {
                detectedWord = palabra;
                maxRepeticiones = cantidad;
            }
        }

        if (maxRepeticiones >= REPETICIONES_NECESARIAS) {
            ultimaPrediccion = detectedWord;
            contadorEstable = maxRepeticiones;
            lastDetectionTime = Date.now();
        } else if (Date.now() - lastDetectionTime < DETECCION_MANTENER_MS && ultimaPrediccion) {
            detectedWord = ultimaPrediccion;
        } else {
            ultimaPrediccion = "";
            contadorEstable = 0;
        }

        resultText.textContent = detectedWord;

        if (detectedWord !== "---" && detectedWord !== lastDisplayedWord) {
            mostrarRespuestas(detectedWord);
            const tarjeta = document.getElementById('tarjeta-resultado');
            if (tarjeta) {
                tarjeta.classList.remove('detectado-pulse');
                void tarjeta.offsetWidth;
                tarjeta.classList.add('detectado-pulse');
            }
            lastDisplayedWord = detectedWord;
        } else if (detectedWord === "---") {
            lastDisplayedWord = "";
        }

        const now = Date.now();
        if (detectedWord !== "---" && detectedWord !== lastSpokenWord && now - lastSpokenTime > 2000) {
            hablar(detectedWord);
            lastSpokenWord = detectedWord;
            lastSpokenTime = now;
        } else if (detectedWord === "---") {
            lastSpokenWord = "";
        }
    } catch (error) {
        console.error("Error durante la detección de señas:", error);
        statusText.textContent = "Error al analizar la cámara";
        historialPredicciones = [];
        resultText.textContent = "---";
    } finally {
        rafId = requestAnimationFrame(predecirWebcam);
    }
}

startBtn.addEventListener('click', iniciarTodo);