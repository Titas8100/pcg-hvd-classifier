/**
 * CardioAcoustics AI - In-Browser Signal Processing & Deep ML Inference Engine
 * Supports multi-format audio decoding, feature extraction, real-time waveform rendering,
 * and client-side classification across 5 Heart Valve Disease categories.
 */

// Global state
let audioCtx = null;
let currentAudioBuffer = null;
let currentSourceNode = null;
let isPlaying = false;
let playbackStartTime = 0;
let pauseOffset = 0;
let animationFrameId = null;
let modelData = null;

// Clinical Descriptions & Pathologies
const PATHOLOGY_INFO = {
  'N': {
    title: 'Normal Cardiac Acoustics (Healthy Control)',
    statusClass: 'normal-status',
    badgeClass: 'badge-normal',
    description: 'Crisp, physiologic S1 ("lub") and S2 ("dub") heart sounds with normal systolic and diastolic intervals. Absence of acoustic murmurs or high-frequency turbulent vibrations.',
    notes: 'Normal valvular opening and laminar blood flow. The mitral and tricuspid valves close synchronously during S1, followed by aortic and pulmonary semilunar closure in S2.'
  },
  'AS': {
    title: 'Aortic Stenosis (AS)',
    statusClass: 'as-status',
    badgeClass: 'badge-as',
    description: 'Pathological narrowing of the aortic valve orifice obstructing left ventricular outflow. Characterized by a prominent, harsh diamond-shaped (crescendo-decrescendo) systolic murmur.',
    notes: 'Murmur peaks in mid-to-late systole and radiates to the carotid arteries. Auscultated loudest at the right second intercostal space with a diminished or delayed S2.'
  },
  'MR': {
    title: 'Mitral Regurgitation (MR)',
    statusClass: 'mr-status',
    badgeClass: 'badge-mr',
    description: 'Incompetence of the mitral valve permitting backward retrograde leakage of blood from the left ventricle into the left atrium during ventricular contraction.',
    notes: 'Presents as a blowing holosystolic (pansystolic) murmur of uniform acoustic intensity throughout the entire systolic interval. Auscultated best at the cardiac apex radiating to the left axilla.'
  },
  'MS': {
    title: 'Mitral Stenosis (MS)',
    statusClass: 'ms-status',
    badgeClass: 'badge-ms',
    description: 'Structural narrowing and fibrosis of the mitral valve obstructing diastolic filling into the left ventricle. High acoustic energy in low-frequency rumbling spectra.',
    notes: 'Characterized by a sharp opening snap shortly after S2 followed by a low-pitched, decrescendo diastolic rumble. Best heard at the cardiac apex in the left lateral decubitus position.'
  },
  'MVP': {
    title: 'Mitral Valve Prolapse (MVP)',
    statusClass: 'mvp-status',
    badgeClass: 'badge-mvp',
    description: 'Abnormal superior displacement and bulging of enlarged mitral valve leaflets into the left atrium during systole. Acoustic pattern shows a mid-systolic click and late systolic murmur.',
    notes: 'The mid-systolic click arises from sudden tensing of the chordae tendineae and redundant valve tissue, accompanied by late-systolic regurgitant turbulence.'
  }
};

// Initialize Audio Context on user interaction
function getAudioContext() {
  if (!audioCtx) {
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    audioCtx = new AudioContextClass();
  }
  if (audioCtx.state === 'suspended') {
    audioCtx.resume();
  }
  return audioCtx;
}

// Load Pre-Trained Model Weights
async function loadModelWeights() {
  const embedded = (typeof PCG_MODEL_DATA !== 'undefined' && PCG_MODEL_DATA) ||
                   (typeof window !== 'undefined' && window.PCG_MODEL_DATA ? window.PCG_MODEL_DATA : null);
  if (embedded) {
    modelData = embedded;
    if (typeof window !== 'undefined') window.modelData = modelData;
    console.log("CardioAcoustics AI: Loaded embedded GBDT model weights directly.");
    return;
  }
  try {
    const response = await fetch('model_weights.json');
    if (response.ok) {
      modelData = await response.json();
      if (typeof window !== 'undefined') window.modelData = modelData;
      console.log("CardioAcoustics AI: Successfully loaded model_weights.json.");
    }
  } catch (err) {
    console.warn("Could not load external model_weights.json:", err);
  }
}

// -----------------------------------------------------------------------------
// Signal Processing & 80-Feature Extraction Engine in JavaScript
// -----------------------------------------------------------------------------

function extractAudioChannelData(audioBuffer) {
  const numChannels = audioBuffer.numberOfChannels;
  const length = audioBuffer.length;
  const mono = new Float32Array(length);

  for (let c = 0; c < numChannels; c++) {
    const channelData = audioBuffer.getChannelData(c);
    for (let i = 0; i < length; i++) {
      mono[i] += channelData[i] / numChannels;
    }
  }

  // Remove DC Offset
  let mean = 0;
  for (let i = 0; i < length; i++) mean += mono[i];
  mean /= length;

  let maxPeak = 0;
  for (let i = 0; i < length; i++) {
    mono[i] -= mean;
    const abs = Math.abs(mono[i]);
    if (abs > maxPeak) maxPeak = abs;
  }

  // Peak normalization
  if (maxPeak > 1e-6) {
    for (let i = 0; i < length; i++) {
      mono[i] /= maxPeak;
    }
  }
  return mono;
}

// Resample or interpolate signal to target 8000 Hz if recorded at 44.1kHz or 48kHz
function resampleTo8kHz(signal, originalRate) {
  if (originalRate === 8000) return signal;
  const targetRate = 8000;
  const ratio = originalRate / targetRate;
  const targetLength = Math.round(signal.length / ratio);
  const resampled = new Float32Array(targetLength);

  for (let i = 0; i < targetLength; i++) {
    const srcIndex = i * ratio;
    const i0 = Math.floor(srcIndex);
    const i1 = Math.min(i0 + 1, signal.length - 1);
    const t = srcIndex - i0;
    resampled[i] = (1 - t) * signal[i0] + t * signal[i1];
  }
  return resampled;
}

// 1. Discrete Wavelet Transform (db4, level 3 -> cA3, cD3, cD2, cD1 statistics = 24 features)
function getSymmetricIndex(len, idx) {
  while (idx < 0 || idx >= len) {
    if (idx < 0) idx = -1 - idx;
    else if (idx >= len) idx = 2 * len - 1 - idx;
  }
  return idx;
}

function dwt1D(signal, h, g) {
  const flen = h.length;
  const outLen = Math.floor((signal.length + flen - 1) / 2);
  const cA = new Float32Array(outLen);
  const cD = new Float32Array(outLen);
  const hRev = h.slice().reverse();
  const gRev = g.slice().reverse();

  for (let i = 0; i < outLen; i++) {
    let sA = 0.0;
    let sD = 0.0;
    for (let j = 0; j < flen; j++) {
      const idx = getSymmetricIndex(signal.length, 2 * i + j - (flen - 2));
      const val = signal[idx];
      sA += val * hRev[j];
      sD += val * gRev[j];
    }
    cA[i] = sA;
    cD[i] = sD;
  }
  return { cA, cD };
}

function computeSubbandStats(coeffs) {
  const n = coeffs.length;
  if (n === 0) return [0, 0, 0, 0, 0, 0];

  let sum = 0, sumSq = 0;
  for (let i = 0; i < n; i++) {
    sum += coeffs[i];
    sumSq += coeffs[i] * coeffs[i];
  }
  const mean = sum / n;
  const variance = Math.max(0, (sumSq / n) - (mean * mean));
  const std = Math.sqrt(variance);

  // Skewness and Kurtosis (biased as in scipy.stats.skew/kurtosis default)
  let m3 = 0, m4 = 0;
  if (std > 1e-8) {
    for (let i = 0; i < n; i++) {
      const diff = coeffs[i] - mean;
      m3 += diff * diff * diff;
      m4 += diff * diff * diff * diff;
    }
    m3 /= (n * std * std * std);
    m4 = (m4 / (n * std * std * std * std)) - 3.0; // Fisher's excess kurtosis
  }

  // Shannon Entropy
  let totalEnergy = sumSq;
  let entropy = 0.0;
  if (totalEnergy > 1e-12 && n > 1) {
    const log2N = Math.log2(n);
    for (let i = 0; i < n; i++) {
      const p = (coeffs[i] * coeffs[i]) / totalEnergy;
      if (p > 1e-12) {
        entropy -= p * Math.log2(p);
      }
    }
    entropy /= log2N;
  }

  return [mean, std, variance, m3, m4, entropy];
}

function extractDWT24(signal, h, g) {
  const l1 = dwt1D(signal, h, g);
  const l2 = dwt1D(l1.cA, h, g);
  const l3 = dwt1D(l2.cA, h, g);

  const subbands = [l3.cA, l3.cD, l2.cD, l1.cD];
  const feats = [];
  for (let s = 0; s < subbands.length; s++) {
    const stats = computeSubbandStats(subbands[s]);
    for (let k = 0; k < stats.length; k++) feats.push(stats[k]);
  }
  return feats;
}

// 2. Cooley-Tukey Radix-2 Complex FFT & IFFT
function fftRadix2(real, imag, inverse = false) {
  const n = real.length;
  for (let i = 0, j = 0; i < n; i++) {
    if (j > i) {
      const tR = real[i]; real[i] = real[j]; real[j] = tR;
      const tI = imag[i]; imag[i] = imag[j]; imag[j] = tI;
    }
    let m = n >> 1;
    while (m >= 1 && j >= m) {
      j -= m;
      m >>= 1;
    }
    j += m;
  }

  for (let s = 2; s <= n; s <<= 1) {
    const halfS = s >> 1;
    const angle = (inverse ? 2 : -2) * Math.PI / s;
    const wR = Math.cos(angle);
    const wI = Math.sin(angle);
    for (let i = 0; i < n; i += s) {
      let curWR = 1.0;
      let curWI = 0.0;
      for (let j = 0; j < halfS; j++) {
        const uR = real[i + j];
        const uI = imag[i + j];
        const vR = real[i + j + halfS] * curWR - imag[i + j + halfS] * curWI;
        const vI = real[i + j + halfS] * curWI + imag[i + j + halfS] * curWR;
        real[i + j] = uR + vR;
        imag[i + j] = uI + vI;
        real[i + j + halfS] = uR - vR;
        imag[i + j + halfS] = uI - vI;
        const nextWR = curWR * wR - curWI * wI;
        curWI = curWR * wI + curWI * wR;
        curWR = nextWR;
      }
    }
  }

  if (inverse) {
    for (let i = 0; i < n; i++) {
      real[i] /= n;
      imag[i] /= n;
    }
  }
}

function rfft(input, nFft) {
  const real = new Float32Array(nFft);
  const imag = new Float32Array(nFft);
  const inLen = Math.min(input.length, nFft);
  for (let i = 0; i < inLen; i++) real[i] = input[i];
  fftRadix2(real, imag, false);

  const numBins = (nFft >> 1) + 1;
  const mag = new Float32Array(numBins);
  const power = new Float32Array(numBins);
  for (let k = 0; k < numBins; k++) {
    const p = real[k] * real[k] + imag[k] * imag[k];
    power[k] = p;
    mag[k] = Math.sqrt(p);
  }
  return { real, imag, mag, power };
}

// 3. Mel-Frequency Cepstral Coefficients (19 mean + 19 std = 38 features)
function extractMFCC38(signal, fbank, dctMat) {
  const frameLen = 240;
  const hop = 80;
  const nFft = 512;
  const numCepstral = 19;
  const numFilters = 26;

  const window = new Float32Array(frameLen);
  for (let i = 0; i < frameLen; i++) {
    window[i] = 0.54 - 0.46 * Math.cos((2 * Math.PI * i) / (frameLen - 1));
  }

  const numFrames = 1 + Math.floor((signal.length - frameLen) / hop);
  if (numFrames <= 0) return new Float32Array(38);

  const mfccFrames = [];
  const frameBuf = new Float32Array(nFft);

  for (let f = 0; f < numFrames; f++) {
    const start = f * hop;
    frameBuf.fill(0);
    for (let i = 0; i < frameLen; i++) {
      frameBuf[i] = signal[start + i] * window[i];
    }
    const { power } = rfft(frameBuf, nFft);

    const logMelEnergies = new Float32Array(numFilters);
    const scale = 1.0 / nFft;
    for (let m = 0; m < numFilters; m++) {
      const filter = fbank[m];
      let energy = 0.0;
      for (let k = 0; k < 257; k++) {
        energy += (power[k] * scale) * filter[k];
      }
      logMelEnergies[m] = Math.log(Math.max(2.22e-16, energy));
    }

    const cep = new Float32Array(numCepstral);
    for (let c = 0; c < numCepstral; c++) {
      let sum = 0.0;
      const row = dctMat[c];
      for (let m = 0; m < numFilters; m++) {
        sum += row[m] * logMelEnergies[m];
      }
      cep[c] = sum;
    }
    mfccFrames.push(cep);
  }

  const result = [];
  for (let c = 0; c < numCepstral; c++) {
    let sum = 0;
    for (let f = 0; f < numFrames; f++) sum += mfccFrames[f][c];
    result.push(sum / numFrames);
  }
  for (let c = 0; c < numCepstral; c++) {
    let sum = 0, sumSq = 0;
    for (let f = 0; f < numFrames; f++) {
      const v = mfccFrames[f][c];
      sum += v; sumSq += v * v;
    }
    const m = sum / numFrames;
    result.push(Math.sqrt(Math.max(0, (sumSq / numFrames) - (m * m))));
  }
  return result;
}

// 4. Spectral Shape Descriptors (10 features)
function extractSpectral10(signal, fs = 8000) {
  const nperseg = 256;
  const noverlap = 128;
  const hop = nperseg - noverlap; // 128
  const pad = nperseg >> 1;

  const paddedLen = signal.length + 2 * pad;
  const paddedSig = new Float32Array(paddedLen);
  for (let i = 0; i < signal.length; i++) {
    paddedSig[pad + i] = signal[i];
  }

  const window = new Float32Array(nperseg);
  for (let i = 0; i < nperseg; i++) {
    window[i] = 0.5 * (1.0 - Math.cos((2.0 * Math.PI * i) / nperseg));
  }

  const numFrames = Math.floor((paddedLen - nperseg) / hop) + 1;
  const numBins = (nperseg >> 1) + 1; // 129
  const freqs = new Float32Array(numBins);
  for (let k = 0; k < numBins; k++) {
    freqs[k] = (k * fs) / nperseg;
  }

  const centroids = new Float32Array(numFrames);
  const spreads = new Float32Array(numFrames);
  const ro85s = new Float32Array(numFrames);
  const ro95s = new Float32Array(numFrames);
  const flatnesses = new Float32Array(numFrames);

  const frameBuf = new Float32Array(nperseg);
  for (let f = 0; f < numFrames; f++) {
    const start = f * hop;
    for (let i = 0; i < nperseg; i++) {
      frameBuf[i] = paddedSig[start + i] * window[i];
    }
    const { power } = rfft(frameBuf, nperseg);

    let totPower = 0.0;
    let weightedSum = 0.0;
    let logSum = 0.0;
    for (let k = 0; k < numBins; k++) {
      const p = power[k] + 1e-12;
      totPower += p;
      weightedSum += freqs[k] * p;
      logSum += Math.log(p);
    }

    const c = weightedSum / totPower;
    centroids[f] = c;

    let spreadSum = 0.0;
    for (let k = 0; k < numBins; k++) {
      const diff = freqs[k] - c;
      spreadSum += diff * diff * (power[k] + 1e-12);
    }
    spreads[f] = Math.sqrt(spreadSum / totPower);

    let cum = 0.0;
    let ro85 = freqs[numBins - 1];
    let ro95 = freqs[numBins - 1];
    let found85 = false, found95 = false;
    for (let k = 0; k < numBins; k++) {
      cum += power[k] + 1e-12;
      if (!found85 && cum >= 0.85 * totPower) {
        ro85 = freqs[k];
        found85 = true;
      }
      if (!found95 && cum >= 0.95 * totPower) {
        ro95 = freqs[k];
        found95 = true;
        break;
      }
    }
    ro85s[f] = ro85;
    ro95s[f] = ro95;

    const geomMean = Math.exp(logSum / numBins);
    const arithMean = totPower / numBins;
    flatnesses[f] = geomMean / arithMean;
  }

  function getMeanStd(arr) {
    let s = 0, sq = 0;
    for (let i = 0; i < arr.length; i++) {
      s += arr[i]; sq += arr[i] * arr[i];
    }
    const m = s / arr.length;
    const sd = Math.sqrt(Math.max(0, (sq / arr.length) - m * m));
    return [m, sd];
  }

  const [cM, cS] = getMeanStd(centroids);
  const [spM, spS] = getMeanStd(spreads);
  const [r85M, r85S] = getMeanStd(ro85s);
  const [r95M, r95S] = getMeanStd(ro95s);
  const [flM, flS] = getMeanStd(flatnesses);

  return [cM, cS, spM, spS, r85M, r85S, r95M, r95S, flM, flS];
}

// 5. Temporal & Exact Discrete Hilbert Envelope Descriptors (8 features)
function extractTemporal8(signal) {
  const n = signal.length;
  if (n === 0) return [0, 0, 0, 0, 0, 0, 0, 0];

  // Zero-crossing rate
  let zcrSum = 0, zcrSq = 0;
  const zcrCount = n - 1;
  for (let i = 0; i < zcrCount; i++) {
    const s1 = signal[i] > 0 ? 1 : (signal[i] < 0 ? -1 : 0);
    const s2 = signal[i + 1] > 0 ? 1 : (signal[i + 1] < 0 ? -1 : 0);
    const z = Math.abs(s2 - s1) / 2.0;
    zcrSum += z;
    zcrSq += z * z;
  }
  const zcrMean = zcrSum / zcrCount;
  const zcrStd = Math.sqrt(Math.max(0, (zcrSq / zcrCount) - zcrMean * zcrMean));

  // RMS Energy & Crest Factor
  let sumSq = 0, peakVal = 0;
  for (let i = 0; i < n; i++) {
    const abs = Math.abs(signal[i]);
    if (abs > peakVal) peakVal = abs;
    sumSq += signal[i] * signal[i];
  }
  const rmsVal = Math.sqrt(sumSq / n);
  const crestFactor = peakVal / (rmsVal + 1e-8);

  // Exact Discrete Hilbert Envelope via FFT
  let nFft = 1;
  while (nFft < n) nFft <<= 1;
  const rFft = new Float32Array(nFft);
  const iFft = new Float32Array(nFft);
  for (let i = 0; i < n; i++) rFft[i] = signal[i];

  fftRadix2(rFft, iFft, false);

  const halfN = nFft >> 1;
  for (let k = 1; k < halfN; k++) {
    rFft[k] *= 2.0;
    iFft[k] *= 2.0;
  }
  for (let k = halfN + 1; k < nFft; k++) {
    rFft[k] = 0.0;
    iFft[k] = 0.0;
  }

  fftRadix2(rFft, iFft, true);

  let envSum = 0, envSumSq = 0;
  const env = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const v = Math.sqrt(rFft[i] * rFft[i] + iFft[i] * iFft[i]);
    env[i] = v;
    envSum += v;
    envSumSq += v * v;
  }
  const envMean = envSum / n;
  const envVariance = Math.max(0, (envSumSq / n) - (envMean * envMean));
  const envStd = Math.sqrt(envVariance);

  let m3 = 0, m4 = 0;
  if (envStd > 1e-8) {
    for (let i = 0; i < n; i++) {
      const diff = env[i] - envMean;
      m3 += diff * diff * diff;
      m4 += diff * diff * diff * diff;
    }
    m3 /= (n * envStd * envStd * envStd);
    m4 = (m4 / (n * envStd * envStd * envStd * envStd)) - 3.0; // Fisher's excess kurtosis
  }

  return [zcrMean, zcrStd, rmsVal, crestFactor, envMean, envStd, m3, m4];
}

// Full 80-feature extraction pipeline
function extractAll80Features(signal, modelData) {
  const dwt24 = extractDWT24(signal, modelData.dwt_h, modelData.dwt_g);
  const mfcc38 = extractMFCC38(signal, modelData.fbank, modelData.dct);
  const spec10 = extractSpectral10(signal, 8000);
  const temp8 = extractTemporal8(signal);

  const feats = [];
  for (let i = 0; i < dwt24.length; i++) feats.push(dwt24[i]);
  for (let i = 0; i < mfcc38.length; i++) feats.push(mfcc38[i]);
  for (let i = 0; i < spec10.length; i++) feats.push(spec10[i]);
  for (let i = 0; i < temp8.length; i++) feats.push(temp8[i]);
  return feats;
}

// Evaluate a single decision tree on feature vector x
function evalTree(tree, x) {
  let node = 0;
  while (tree.l[node] !== -1) {
    if (x[tree.f[node]] <= tree.th[node]) {
      node = tree.l[node];
    } else {
      node = tree.r[node];
    }
  }
  return tree.v[node];
}

// Master Classifier: In-Browser GBDT Engine on 80 Enhanced Features
function predictHeartValveDisease(signal) {
  const model = modelData || (typeof PCG_MODEL_DATA !== 'undefined' ? PCG_MODEL_DATA : null);
  if (!model) {
    console.error("No model data available for prediction.");
    return {
      probs: { 'N': 0.2, 'AS': 0.2, 'MR': 0.2, 'MS': 0.2, 'MVP': 0.2 },
      descriptors: { centroid: 0, zcr: 0, rolloff85: 0, entropy: 0 }
    };
  }

  const feats = extractAll80Features(signal, model);

  // Bio-acoustic descriptors for clinical readout display
  const descriptors = {
    centroid: feats[62], // Spec_Centroid_Mean (Hz)
    zcr: feats[72],      // Temp_ZCR_Mean
    rolloff85: feats[66],// Spec_Rolloff85_Mean (Hz)
    entropy: feats[5],   // DWT_cA3_entropy
    rms: feats[74],      // Temp_RMS_Energy
    crestFactor: feats[75] // Temp_Crest_Factor
  };

  const numClasses = model.classes.length;
  const raw = model.init_raw.slice();
  const lr = model.learning_rate;

  for (let m = 0; m < model.n_estimators; m++) {
    const stageTrees = model.trees[m];
    for (let k = 0; k < numClasses; k++) {
      raw[k] += lr * evalTree(stageTrees[k], feats);
    }
  }

  // Softmax normalization
  let maxRaw = -Infinity;
  for (let k = 0; k < numClasses; k++) {
    if (raw[k] > maxRaw) maxRaw = raw[k];
  }
  let expSum = 0;
  const probs = {};
  for (let k = 0; k < numClasses; k++) {
    const e = Math.exp(raw[k] - maxRaw);
    probs[model.classes[k]] = e;
    expSum += e;
  }
  for (let k = 0; k < numClasses; k++) {
    probs[model.classes[k]] /= expSum;
  }

  return { probs, descriptors };
}

// -----------------------------------------------------------------------------
// Interactive Waveform & Spectrogram Rendering
// -----------------------------------------------------------------------------

function renderWaveform(signal, canvas) {
  const ctx = canvas.getContext('2d');
  const width = canvas.width;
  const height = canvas.height;
  const centerY = height / 2;

  ctx.fillStyle = '#040711';
  ctx.fillRect(0, 0, width, height);

  // Subtle grid lines
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
  ctx.lineWidth = 1;
  for (let y = 0; y < height; y += 30) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
    ctx.stroke();
  }
  ctx.strokeStyle = 'rgba(0, 229, 255, 0.2)';
  ctx.beginPath();
  ctx.moveTo(0, centerY);
  ctx.lineTo(width, centerY);
  ctx.stroke();

  // Waveform line
  const step = Math.ceil(signal.length / width);
  const amp = (height / 2) * 0.88;

  // Gradient fill under waveform
  const grad = ctx.createLinearGradient(0, 0, 0, height);
  grad.addColorStop(0, 'rgba(0, 229, 255, 0.35)');
  grad.addColorStop(0.5, 'rgba(0, 229, 255, 0.05)');
  grad.addColorStop(1, 'rgba(0, 229, 255, 0.35)');

  ctx.beginPath();
  ctx.moveTo(0, centerY);

  for (let x = 0; x < width; x++) {
    let min = 1.0;
    let max = -1.0;
    const startIdx = x * step;
    for (let j = 0; j < step; j++) {
      const idx = startIdx + j;
      if (idx < signal.length) {
        const val = signal[idx];
        if (val < min) min = val;
        if (val > max) max = val;
      }
    }
    ctx.lineTo(x, centerY - max * amp);
  }

  for (let x = width - 1; x >= 0; x--) {
    let min = 1.0;
    const startIdx = x * step;
    for (let j = 0; j < step; j++) {
      const idx = startIdx + j;
      if (idx < signal.length) {
        const val = signal[idx];
        if (val < min) min = val;
      }
    }
    ctx.lineTo(x, centerY - min * amp);
  }
  ctx.closePath();
  ctx.fillStyle = grad;
  ctx.fill();

  // Primary Line
  ctx.beginPath();
  ctx.strokeStyle = '#00e5ff';
  ctx.lineWidth = 1.6;
  ctx.shadowColor = 'rgba(0, 229, 255, 0.8)';
  ctx.shadowBlur = 8;

  for (let x = 0; x < width; x++) {
    const idx = Math.min(signal.length - 1, x * step);
    const y = centerY - signal[idx] * amp;
    if (x === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.shadowBlur = 0;
}

function renderSpectrogram(signal, canvas, sampleRate = 8000) {
  const ctx = canvas.getContext('2d');
  const width = canvas.width;
  const height = canvas.height;
  ctx.fillStyle = '#040711';
  ctx.fillRect(0, 0, width, height);

  const fftSize = 256;
  const halfFft = fftSize / 2;
  const hop = Math.floor(fftSize / 2);
  const numSlices = Math.min(width, Math.max(1, Math.floor((signal.length - fftSize) / hop) + 1));
  if (numSlices <= 0) return;

  const sliceWidth = width / numSlices;
  const frameBuf = new Float32Array(fftSize);
  const binHeight = height / halfFft;

  for (let s = 0; s < numSlices; s++) {
    const start = s * hop;
    frameBuf.fill(0);
    const len = Math.min(fftSize, signal.length - start);
    for (let n = 0; n < len; n++) {
      const w = 0.5 * (1 - Math.cos((2 * Math.PI * n) / (fftSize - 1)));
      frameBuf[n] = signal[start + n] * w;
    }

    const { mag } = rfft(frameBuf, fftSize);

    for (let k = 0; k < halfFft; k++) {
      const m = mag[k];
      const norm = Math.min(1.0, Math.log1p(m * 3.0));

      // Color mapping: dark navy -> cyan -> yellow -> bright red
      const r = Math.floor(Math.min(255, Math.max(0, (norm - 0.45) * 460)));
      const g = Math.floor(Math.min(255, norm * 255));
      const b = Math.floor(Math.max(30, 255 - norm * 210));

      ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
      const y = height - (k + 1) * binHeight;
      ctx.fillRect(s * sliceWidth, y, sliceWidth + 0.5, binHeight + 0.5);
    }
  }
}

// -----------------------------------------------------------------------------
// Audio Playback & Animation Loop
// -----------------------------------------------------------------------------

function startAudioPlayback() {
  if (!currentAudioBuffer) return;
  const ctx = getAudioContext();

  currentSourceNode = ctx.createBufferSource();
  currentSourceNode.buffer = currentAudioBuffer;

  const gainNode = ctx.createGain();
  const volumeSlider = document.getElementById('volumeSlider');
  gainNode.gain.value = volumeSlider ? parseFloat(volumeSlider.value) : 0.9;

  currentSourceNode.connect(gainNode);
  gainNode.connect(ctx.destination);

  playbackStartTime = ctx.currentTime - pauseOffset;
  currentSourceNode.start(0, pauseOffset);
  isPlaying = true;

  updatePlayButtonUI(true);

  currentSourceNode.onended = () => {
    if (isPlaying) {
      isPlaying = false;
      pauseOffset = 0;
      updatePlayButtonUI(false);
      resetPlayhead();
    }
  };

  runPlayheadAnimation();
}

function stopAudioPlayback() {
  if (currentSourceNode) {
    try {
      currentSourceNode.stop();
    } catch (e) {}
    try {
      currentSourceNode.disconnect();
    } catch (e) {}
    currentSourceNode = null;
  }
  if (currentAudioBuffer) {
    const ctx = getAudioContext();
    pauseOffset = ctx.currentTime - playbackStartTime;
    if (pauseOffset >= currentAudioBuffer.duration || pauseOffset < 0) {
      pauseOffset = 0;
    }
  } else {
    pauseOffset = 0;
  }
  isPlaying = false;
  updatePlayButtonUI(false);
  cancelAnimationFrame(animationFrameId);
}

function updatePlayButtonUI(playing) {
  const playIcon = document.getElementById('playIcon');
  const pauseIcon = document.getElementById('pauseIcon');
  if (playIcon && pauseIcon) {
    playIcon.style.display = playing ? 'none' : 'block';
    pauseIcon.style.display = playing ? 'block' : 'none';
  }
}

function resetPlayhead() {
  const playhead = document.getElementById('playhead');
  const scrubber = document.getElementById('playbackScrubber');
  const curTime = document.getElementById('currentTime');
  if (playhead) playhead.style.left = '0%';
  if (scrubber) scrubber.value = 0;
  if (curTime) curTime.textContent = '0:00.0';
}

function runPlayheadAnimation() {
  if (!isPlaying || !currentAudioBuffer) return;
  const ctx = getAudioContext();
  const elapsed = ctx.currentTime - playbackStartTime;
  const duration = currentAudioBuffer.duration;
  const progress = Math.min(1.0, elapsed / duration);

  const playhead = document.getElementById('playhead');
  const scrubber = document.getElementById('playbackScrubber');
  const curTime = document.getElementById('currentTime');

  if (playhead) {
    playhead.style.opacity = '1';
    playhead.style.left = `${progress * 100}%`;
  }
  if (scrubber) {
    scrubber.value = progress * 100;
  }
  if (curTime) {
    const sec = elapsed.toFixed(1);
    curTime.textContent = `0:${sec < 10 ? '0' : ''}${sec}`;
  }

  if (progress < 1.0) {
    animationFrameId = requestAnimationFrame(runPlayheadAnimation);
  }
}

// -----------------------------------------------------------------------------
// UI Diagnostic Display & Presentation
// -----------------------------------------------------------------------------

function displayDiagnosticResults(predictionProbs, descriptors) {
  const emptyState = document.getElementById('emptyState');
  const activeResults = document.getElementById('activeResults');
  const evalBadge = document.getElementById('evalStatusBadge');
  const diagBanner = document.getElementById('diagnosisBanner');
  const diagTitle = document.getElementById('diagTitle');
  const diagDesc = document.getElementById('diagDescription');
  const confBadge = document.getElementById('confidenceBadge');
  const clinNotes = document.getElementById('clinicalNotes');

  if (emptyState) emptyState.style.display = 'none';
  if (activeResults) activeResults.style.display = 'block';
  if (evalBadge) {
    evalBadge.textContent = 'Completed';
    evalBadge.className = 'badge badge-accent';
  }

  // Find top predicted class
  let bestClass = 'N';
  let bestProb = 0;
  if (predictionProbs) {
    for (const [cls, prob] of Object.entries(predictionProbs)) {
      if (prob > bestProb) {
        bestProb = prob;
        bestClass = cls;
      }
    }
  }

  const pInfo = PATHOLOGY_INFO[bestClass] || PATHOLOGY_INFO['N'];
  const confPercent = (bestProb * 100).toFixed(1);

  const isBorderline = bestProb < 0.55;
  const isNearSilent = descriptors && descriptors.rms < 0.005;

  // Update Hero Banner with Calibrated Uncertainty
  if (diagBanner) {
    if (isBorderline) {
      diagBanner.className = `diagnosis-banner borderline-status`;
    } else {
      diagBanner.className = `diagnosis-banner ${pInfo.statusClass}`;
    }
  }

  if (diagTitle) {
    if (isBorderline) {
      diagTitle.textContent = `${pInfo.title} (Borderline / Inconclusive)`;
    } else {
      diagTitle.textContent = pInfo.title;
    }
  }

  if (diagDesc) {
    if (isNearSilent) {
      diagDesc.textContent = `Acoustic Warning: Very low signal energy detected (RMS < 0.005). Please ensure the recording was captured directly with a digital stethoscope placed firmly on the thorax.`;
    } else if (isBorderline) {
      diagDesc.textContent = `Borderline Confidence: Acoustic profile shows dispersed features across multiple categories (top match: ${pInfo.title} at ${confPercent}%). For external recordings outside the benchmark dataset, verify stethoscope placement and cardiac cycle segmentation.`;
    } else {
      diagDesc.textContent = pInfo.description;
    }
  }

  if (confBadge) {
    if (isBorderline) {
      confBadge.textContent = `${confPercent}% (Borderline)`;
      confBadge.className = 'confidence-badge badge-warning';
    } else {
      confBadge.textContent = `${confPercent}% Confidence`;
      confBadge.className = 'confidence-badge';
    }
  }

  if (clinNotes) clinNotes.textContent = pInfo.notes;

  // Animate Probability Bars
  const classOrder = ['N', 'AS', 'MR', 'MS', 'MVP'];
  classOrder.forEach(c => {
    const prob = (predictionProbs && predictionProbs[c]) || 0;
    const pct = (prob * 100).toFixed(1);
    const bar = document.getElementById(`bar${c}`);
    const label = document.getElementById(`probVal${c}`);
    if (bar) bar.style.width = `${pct}%`;
    if (label) label.textContent = `${pct}%`;
  });

  // Display Bio-Acoustic Descriptors
  if (descriptors) {
    const cEl = document.getElementById('valCentroid');
    const zEl = document.getElementById('valZCR');
    const rEl = document.getElementById('valRolloff');
    const eEl = document.getElementById('valEntropy');

    if (cEl) cEl.textContent = `${Math.round(descriptors.centroid)} Hz`;
    if (zEl) zEl.textContent = descriptors.zcr.toFixed(3);
    if (rEl) rEl.textContent = `${Math.round(descriptors.rolloff85)} Hz`;
    if (eEl) eEl.textContent = descriptors.entropy.toFixed(3);
  }
}

// -----------------------------------------------------------------------------
// Audio Ingestion & Pipeline Orchestration
// -----------------------------------------------------------------------------

function parseWavArrayBuffer(arrayBuffer) {
  const view = new DataView(arrayBuffer);
  if (arrayBuffer.byteLength < 44) return null;
  
  const riff = String.fromCharCode(view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3));
  const wave = String.fromCharCode(view.getUint8(8), view.getUint8(9), view.getUint8(10), view.getUint8(11));
  if (riff !== 'RIFF' || wave !== 'WAVE') return null;

  let pos = 12;
  let format = 1;
  let numChannels = 1;
  let sampleRate = 8000;
  let bitsPerSample = 16;
  let dataOffset = -1;
  let dataSize = 0;

  while (pos + 8 <= arrayBuffer.byteLength) {
    const chunkId = String.fromCharCode(
      view.getUint8(pos), view.getUint8(pos + 1), view.getUint8(pos + 2), view.getUint8(pos + 3)
    );
    const chunkSize = view.getUint32(pos + 4, true);

    if (chunkId === 'fmt ') {
      format = view.getUint16(pos + 8, true);
      numChannels = view.getUint16(pos + 10, true);
      sampleRate = view.getUint32(pos + 12, true);
      bitsPerSample = view.getUint16(pos + 22, true);
    } else if (chunkId === 'data') {
      dataOffset = pos + 8;
      dataSize = Math.min(chunkSize, arrayBuffer.byteLength - dataOffset);
      break;
    }
    pos += 8 + chunkSize;
  }

  if (dataOffset === -1 || dataSize <= 0) return null;

  const bytesPerSample = bitsPerSample / 8;
  const numSamplesTotal = Math.floor(dataSize / bytesPerSample);
  const numFrames = Math.floor(numSamplesTotal / numChannels);
  if (numFrames <= 0) return null;

  const mono = new Float32Array(numFrames);

  if (format === 1 && bitsPerSample === 16) {
    for (let i = 0; i < numFrames; i++) {
      let sum = 0;
      for (let c = 0; c < numChannels; c++) {
        const sampleIdx = (i * numChannels + c) * 2;
        sum += view.getInt16(dataOffset + sampleIdx, true) / 32768.0;
      }
      mono[i] = sum / numChannels;
    }
  } else if (format === 1 && bitsPerSample === 8) {
    for (let i = 0; i < numFrames; i++) {
      let sum = 0;
      for (let c = 0; c < numChannels; c++) {
        const sampleIdx = i * numChannels + c;
        sum += (view.getUint8(dataOffset + sampleIdx) - 128) / 128.0;
      }
      mono[i] = sum / numChannels;
    }
  } else if (format === 1 && bitsPerSample === 24) {
    for (let i = 0; i < numFrames; i++) {
      let sum = 0;
      for (let c = 0; c < numChannels; c++) {
        const byteIdx = dataOffset + (i * numChannels + c) * 3;
        const b0 = view.getUint8(byteIdx);
        const b1 = view.getUint8(byteIdx + 1);
        const b2 = view.getInt8(byteIdx + 2);
        const val = (b2 << 16) | (b1 << 8) | b0;
        sum += val / 8388608.0;
      }
      mono[i] = sum / numChannels;
    }
  } else if (format === 3 && bitsPerSample === 32) {
    for (let i = 0; i < numFrames; i++) {
      let sum = 0;
      for (let c = 0; c < numChannels; c++) {
        const sampleIdx = (i * numChannels + c) * 4;
        sum += view.getFloat32(dataOffset + sampleIdx, true);
      }
      mono[i] = sum / numChannels;
    }
  } else {
    return null;
  }

  // Remove DC offset and peak normalize
  let mean = 0;
  for (let i = 0; i < numFrames; i++) mean += mono[i];
  mean /= numFrames;

  let maxPeak = 0;
  for (let i = 0; i < numFrames; i++) {
    mono[i] -= mean;
    const abs = Math.abs(mono[i]);
    if (abs > maxPeak) maxPeak = abs;
  }
  if (maxPeak > 1e-6) {
    for (let i = 0; i < numFrames; i++) mono[i] /= maxPeak;
  }

  return {
    signal: mono,
    sampleRate: sampleRate,
    duration: numFrames / sampleRate,
    numberOfChannels: numChannels
  };
}

async function processAudioArrayBuffer(arrayBuffer, fileName) {
  const ctx = getAudioContext();
  try {
    stopAudioPlayback();
    pauseOffset = 0;

    let signal8k = null;
    let sampleRate = 8000;
    let durSecNum = 0;
    let numChannels = 1;

    // 1. Fast, lossless pure JS WAV decoding
    const parsedWav = parseWavArrayBuffer(arrayBuffer);
    if (parsedWav) {
      sampleRate = parsedWav.sampleRate;
      durSecNum = parsedWav.duration;
      numChannels = parsedWav.numberOfChannels;
      signal8k = sampleRate === 8000 ? parsedWav.signal : resampleTo8kHz(parsedWav.signal, sampleRate);

      // Create Web Audio buffer for smooth interactive playback
      try {
        currentAudioBuffer = ctx.createBuffer(1, parsedWav.signal.length, sampleRate);
        currentAudioBuffer.copyToChannel(parsedWav.signal, 0);
      } catch (e) {
        console.warn("Could not create Web Audio buffer for playback:", e);
      }
    } else {
      // 2. Decode with Web Audio API (supports MP3, M4A, FLAC, OGG, AAC, etc.)
      const audioBuffer = await new Promise((resolve, reject) => {
        const bufferCopy = arrayBuffer.slice(0);
        ctx.decodeAudioData(bufferCopy, resolve, reject);
      });
      currentAudioBuffer = audioBuffer;
      sampleRate = audioBuffer.sampleRate;
      durSecNum = audioBuffer.duration;
      numChannels = audioBuffer.numberOfChannels;

      const rawSignal = extractAudioChannelData(audioBuffer);
      signal8k = resampleTo8kHz(rawSignal, sampleRate);
    }

    // Update File Meta UI
    const statusPanel = document.getElementById('fileStatusPanel');
    if (statusPanel) statusPanel.style.display = 'block';
    const activeFileName = document.getElementById('activeFileName');
    if (activeFileName) activeFileName.textContent = fileName;
    const durSec = durSecNum.toFixed(2);
    const activeMeta = document.getElementById('activeFileMeta');
    if (activeMeta) {
      activeMeta.textContent = `${sampleRate} Hz • ${durSec} sec • ${numChannels === 1 ? 'Mono' : 'Stereo'}`;
    }
    const totalDur = document.getElementById('totalDuration');
    if (totalDur) {
      totalDur.textContent = `0:${durSecNum < 10 ? '0' : ''}${Math.round(durSecNum)}`;
    }

    // Render Waveform and Spectrogram Canvases
    const waveCanvas = document.getElementById('waveformCanvas');
    const specCanvas = document.getElementById('spectrogramCanvas');
    if (waveCanvas) renderWaveform(signal8k, waveCanvas);
    if (specCanvas) renderSpectrogram(signal8k, specCanvas, 8000);

    // Compute Exact 80 Features & In-Browser GBDT Model Prediction
    const { probs, descriptors } = predictHeartValveDisease(signal8k);

    // Update Diagnostics
    displayDiagnosticResults(probs, descriptors);

  } catch (err) {
    console.error("Error processing audio recording:", err);
    alert(`Failed to decode audio file (${fileName}). Please check that the file is a valid audio recording.\n\nTechnical details: ${err && err.message ? err.message : err}`);
  }
}

// Load audio file from user upload
function handleFileInput(file) {
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (e) => {
    processAudioArrayBuffer(e.target.result, file.name);
  };
  reader.readAsArrayBuffer(file);
}

// Load clinical sample from server/local
async function loadSampleAudio(sampleUrl, sampleClass) {
  try {
    const response = await fetch(sampleUrl);
    if (!response.ok) throw new Error(`HTTP error ${response.status}`);
    const arrayBuffer = await response.arrayBuffer();
    const displayName = `${PATHOLOGY_INFO[sampleClass].title} (Database Sample)`;
    await processAudioArrayBuffer(arrayBuffer, displayName);
  } catch (e) {
    console.error(`Error loading sample audio from ${sampleUrl}:`, e);
    alert(`Could not load clinical sample (${sampleUrl}). Make sure files are served via HTTP/HTTPS.`);
  }
}

// -----------------------------------------------------------------------------
// DOM Event Listeners & Initialization
// -----------------------------------------------------------------------------

document.addEventListener('DOMContentLoaded', () => {
  loadModelWeights();

  // Drag & Drop Handling
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('audioFileInput');
  const btnBrowse = document.getElementById('btnBrowse');

  if (btnBrowse && fileInput) {
    btnBrowse.addEventListener('click', (e) => {
      e.stopPropagation();
      fileInput.click();
    });
  }

  if (dropzone && fileInput) {
    dropzone.addEventListener('click', () => fileInput.click());

    dropzone.addEventListener('dragover', (e) => {
      e.preventDefault();
      dropzone.classList.add('dragover');
    });

    dropzone.addEventListener('dragleave', () => {
      dropzone.classList.remove('dragover');
    });

    dropzone.addEventListener('drop', (e) => {
      e.preventDefault();
      dropzone.classList.remove('dragover');
      if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        handleFileInput(e.dataTransfer.files[0]);
      }
    });

    fileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files.length > 0) {
        handleFileInput(e.target.files[0]);
      }
    });
  }

  // Clear / Remove File
  const btnRemove = document.getElementById('btnRemoveFile');
  if (btnRemove) {
    btnRemove.addEventListener('click', () => {
      stopAudioPlayback();
      currentAudioBuffer = null;
      document.getElementById('fileStatusPanel').style.display = 'none';
      document.getElementById('activeResults').style.display = 'none';
      document.getElementById('emptyState').style.display = 'block';
      document.getElementById('evalStatusBadge').textContent = 'Ready';
      document.getElementById('evalStatusBadge').className = 'badge';
      resetPlayhead();
      // Clear canvases
      const waveCanvas = document.getElementById('waveformCanvas');
      const specCanvas = document.getElementById('spectrogramCanvas');
      waveCanvas.getContext('2d').clearRect(0, 0, waveCanvas.width, waveCanvas.height);
      specCanvas.getContext('2d').clearRect(0, 0, specCanvas.width, specCanvas.height);
    });
  }

  // Play / Pause Controls
  const btnPlay = document.getElementById('btnPlayPause');
  if (btnPlay) {
    btnPlay.addEventListener('click', () => {
      if (!currentAudioBuffer) return;
      if (isPlaying) {
        stopAudioPlayback();
      } else {
        startAudioPlayback();
      }
    });
  }

  // Scrubber Seeking
  const scrubber = document.getElementById('playbackScrubber');
  if (scrubber) {
    scrubber.addEventListener('input', (e) => {
      if (!currentAudioBuffer) return;
      const progress = parseFloat(e.target.value) / 100;
      pauseOffset = progress * currentAudioBuffer.duration;
      if (isPlaying) {
        stopAudioPlayback();
        startAudioPlayback();
      } else {
        const sec = pauseOffset.toFixed(1);
        document.getElementById('currentTime').textContent = `0:${sec < 10 ? '0' : ''}${sec}`;
        document.getElementById('playhead').style.left = `${progress * 100}%`;
      }
    });
  }

  // Oscilloscope / Spectrogram Tabs
  const tabWaveform = document.getElementById('tabWaveform');
  const tabSpectrogram = document.getElementById('tabSpectrogram');
  const waveCanvas = document.getElementById('waveformCanvas');
  const specCanvas = document.getElementById('spectrogramCanvas');

  if (tabWaveform && tabSpectrogram) {
    tabWaveform.addEventListener('click', () => {
      tabWaveform.classList.add('active');
      tabSpectrogram.classList.remove('active');
      waveCanvas.style.display = 'block';
      specCanvas.style.display = 'none';
    });

    tabSpectrogram.addEventListener('click', () => {
      tabSpectrogram.classList.add('active');
      tabWaveform.classList.remove('active');
      waveCanvas.style.display = 'none';
      specCanvas.style.display = 'block';
    });
  }

  // Clinical Samples Library Interaction
  document.querySelectorAll('.sample-card').forEach(card => {
    const playBtn = card.querySelector('.sample-play-btn');
    const analyzeBtn = card.querySelector('.btn-sample-load');
    const filePath = card.dataset.file;
    const cls = card.dataset.class;

    let sampleAudioElem = null;

    if (playBtn) {
      playBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        if (!sampleAudioElem) {
          sampleAudioElem = new Audio(filePath);
          sampleAudioElem.onended = () => {
            playBtn.querySelector('.play-svg').style.display = 'block';
            playBtn.querySelector('.pause-svg').style.display = 'none';
          };
        }

        if (sampleAudioElem.paused) {
          // Pause all other sample audios first
          document.querySelectorAll('audio').forEach(a => a.pause());
          document.querySelectorAll('.sample-play-btn .pause-svg').forEach(p => p.style.display = 'none');
          document.querySelectorAll('.sample-play-btn .play-svg').forEach(p => p.style.display = 'block');

          sampleAudioElem.play();
          playBtn.querySelector('.play-svg').style.display = 'none';
          playBtn.querySelector('.pause-svg').style.display = 'block';
        } else {
          sampleAudioElem.pause();
          playBtn.querySelector('.play-svg').style.display = 'block';
          playBtn.querySelector('.pause-svg').style.display = 'none';
        }
      });
    }

    if (analyzeBtn) {
      analyzeBtn.addEventListener('click', () => {
        loadSampleAudio(filePath, cls);
      });
    }
  });

  // Methodology Modal
  const btnModal = document.getElementById('btnMethodology');
  const modal = document.getElementById('modalMethodology');
  const btnCloseModal = document.getElementById('btnCloseModal');

  if (btnModal && modal) {
    btnModal.addEventListener('click', () => modal.classList.add('open'));
  }
  if (btnCloseModal && modal) {
    btnCloseModal.addEventListener('click', () => modal.classList.remove('open'));
  }
  if (modal) {
    modal.addEventListener('click', (e) => {
      if (e.target === modal) modal.classList.remove('open');
    });
  }
});
