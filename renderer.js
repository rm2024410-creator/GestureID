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

// 👇 AJUSTADO PARA QUE LA IA RESPONDA AL INSTANTE 👇
// 5 repeticiones es casi medio segundo. 0.50 significa que si la IA está 50% segura, ya responde.
const REPETICIONES_NECESARIAS = 3;  
const UMBRAL_CONFIANZA = 0.20;       

// ==========================================
// EMOJIS
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
    "Hola": { respuestas: [{ texto: "Hola, ¿qué tal?", sena: "Hola" }, { texto: "Buenos días", sena: "Buenos Dias" }, { texto: "Buenas tardes", sena: "Buenas Tardes" }] },
    "Como estas": { respuestas: [{ texto: "Estoy bien, gracias", sena: "Bien" }, { texto: "Estoy regular", sena: "Regular" }, { texto: "Estoy mal", sena: "Mal" }, { texto: "¿Y tú cómo estás?", sena: "Y tu" }] },
    "Mi nombres es": { respuestas: [{ texto: "Mi nombre es Gabriel", sena: "Mi nombres es" }, { texto: "Mi nombre es Erika", sena: "Mi nombres es" }] },
    "Buenos Dias": { respuestas: [{ texto: "Buenos días", sena: "Buenos Dias" }, { texto: "Hola", sena: "Hola" }] },
    "Adios": { respuestas: [{ texto: "Adiós", sena: "Adios" }, { texto: "Nos vemos", sena: "Adios" }, { texto: "Hasta luego", sena: "Adios" }] }
};

// ==========================================
// INICIAR
// ==========================================
async function iniciarTodo() {
    statusText.textContent = "Cargando modelo...";
    startBtn.disabled = true;
    startBtn.textContent = "⏳ Cargando...";

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
            paquete = await respuesta.json();
        }

        clases = paquete.clases;
        statusText.textContent = "Modelo listo";

        const binaryString = atob(paquete.pesos);
        const bytes = new Uint8Array(binaryString.length);
        for (let i = 0; i < binaryString.length; i++) {
            bytes[i] = binaryString.charCodeAt(i);
        }
        const weightData = bytes.buffer;

        modelo = await tf.loadLayersModel(tf.io.fromMemory({
            modelTopology: paquete.modelo,
            weightSpecs: paquete.pesosEspecificaciones,
            weightData: weightData
        }));

        const vision = await window.FilesetResolver.forVisionTasks(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
        );
        handLandmarker = await window.HandLandmarker.createFromOptions(vision, {
            baseOptions: {
                modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
                delegate: "GPU"
            },
            runningMode: "VIDEO",
            numHands: 2
        });

        const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        video.srcObject = stream;
        video.addEventListener('loadeddata', predecirWebcam);

        videoContainer.classList.remove('hidden');
        startBtn.style.display = 'none';
        statusText.textContent = "Modelo listo";

        mostrarRespuestas("__generales__");

    } catch (error) {
        console.error("Error:", error);
        statusText.textContent = "Error: " + error.message;
        startBtn.disabled = false;
        startBtn.textContent = "🎥 Encender Cámara";
    }
}

// ==========================================
// RESPUESTAS (La IA decide qué mostrar)
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

    // Si la IA detecta la palabra, mostramos sus respuestas. Si no, un mensaje neutral.
    const respuestasAMostrar = entrada ? entrada.respuestas : [
        { texto: "Haz una seña para responder", sena: "---" }
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
// PREDICCIÓN (Lógica de la IA)
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
    if (!handLandmarker || !modelo) return;

    let startTimeMs = performance.now();
    if (video.currentTime !== lastVideoTime) {
        lastVideoTime = video.currentTime;
        const results = handLandmarker.detectForVideo(video, startTimeMs);

        let detectedWord = "---";

        if (results.landmarks && results.landmarks.length > 0) {
            const vector = extraerVectorDosManos(results);
            const input = tf.tensor2d([vector]);
            const prediccion = modelo.predict(input);
            const datos = await prediccion.data();

            let maxIdx = 0;
            let maxProb = datos[0];
            for (let i = 1; i < datos.length; i++) {
                if (datos[i] > maxProb) {
                    maxProb = datos[i];
                    maxIdx = i;
                }
            }

            const palabra = clases[maxIdx];
            const debugStr = clases.map((c, i) => `${c}: ${(datos[i] * 100).toFixed(0)}%`).join(" | ");
            debugText.textContent = debugStr;

            // 👇 LA IA DECIDE INMEDIATAMENTE 👇
            if (maxProb > UMBRAL_CONFIANZA) {
                if (palabra === ultimaPrediccion) {
                    contadorEstable++;
                } else {
                    contadorEstable = 1;
                    ultimaPrediccion = palabra;
                }
                if (contadorEstable >= REPETICIONES_NECESARIAS) {
                    detectedWord = palabra;
                }
            } else {
                contadorEstable = 0;
                ultimaPrediccion = "";
            }

            input.dispose();
            prediccion.dispose();
        }

        if (detectedWord === "---") {
            resultText.textContent = "---";
            ultimaPrediccion = ""; 
            contadorEstable = 0;
        } else {
            resultText.textContent = detectedWord;
            mostrarRespuestas(detectedWord);

            const tarjeta = document.getElementById('tarjeta-resultado');
            if (tarjeta && detectedWord !== ultimaPrediccion) {
                tarjeta.classList.remove('detectado-pulse');
                void tarjeta.offsetWidth;
                tarjeta.classList.add('detectado-pulse');

                try {
                    const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
                    const oscillator = audioCtx.createOscillator();
                    const gainNode = audioCtx.createGain();
                    oscillator.connect(gainNode);
                    gainNode.connect(audioCtx.destination);
                    oscillator.type = 'sine';
                    oscillator.frequency.setValueAtTime(880, audioCtx.currentTime);
                    gainNode.gain.setValueAtTime(0.1, audioCtx.currentTime);
                    gainNode.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.5);
                    oscillator.start();
                    oscillator.stop(audioCtx.currentTime + 0.5);
                } catch(e) {}
            }
        }

        const now = Date.now();
        // 1.5 segundos de espera para no sobreponer el audio
        if (detectedWord !== "---" && detectedWord !== lastSpokenWord && (now - lastSpokenTime > 1500)) {
            hablar(detectedWord);
            lastSpokenWord = detectedWord;
            lastSpokenTime = now;
            setTimeout(() => { lastSpokenWord = ""; }, 2000);
        } else if (detectedWord === "---") {
            lastSpokenWord = "";
        }
    }
    requestAnimationFrame(predecirWebcam);
}

startBtn.addEventListener('click', iniciarTodo);