const fs = require('fs');
const path = require('path');
const PCG_MODEL_DATA = require('./model_data.js');

// Parse a simple 16-bit PCM WAV file for node testing
function readWavFile(filePath) {
  const buf = fs.readFileSync(filePath);
  let pos = 12;
  let sampleRate = 8000;
  while (pos < buf.length - 8) {
    const chunkId = buf.toString('ascii', pos, pos + 4);
    const chunkSize = buf.readUInt32LE(pos + 4);
    if (chunkId === 'fmt ') {
      sampleRate = buf.readUInt32LE(pos + 12);
    }
    if (chunkId === 'data') {
      const numSamples = Math.floor(chunkSize / 2);
      const signal = new Float32Array(numSamples);
      for (let i = 0; i < numSamples; i++) {
        signal[i] = buf.readInt16LE(pos + 8 + i * 2) / 32768.0;
      }
      return { signal, sampleRate };
    }
    pos += 8 + chunkSize;
  }
  throw new Error("Could not find data chunk in WAV: " + filePath);
}

// -----------------------------------------------------------------------------
// 1. DWT (db4, level 3 -> cA3, cD3, cD2, cD1 statistics)
// -----------------------------------------------------------------------------

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

  // [cA3, cD3, cD2, cD1]
  const subbands = [l3.cA, l3.cD, l2.cD, l1.cD];
  const feats = [];
  for (let s = 0; s < subbands.length; s++) {
    const stats = computeSubbandStats(subbands[s]);
    for (let k = 0; k < stats.length; k++) feats.push(stats[k]);
  }
  return feats;
}

// -----------------------------------------------------------------------------
// 2. Cooley-Tukey Radix-2 Complex FFT & IFFT
// -----------------------------------------------------------------------------

function fftRadix2(real, imag, inverse = false) {
  const n = real.length;
  // Bit reversal
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

  // Butterflies
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

// -----------------------------------------------------------------------------
// 3. MFCC (19 mean + 19 std = 38)
// -----------------------------------------------------------------------------

function extractMFCC38(signal, fbank, dctMat) {
  const frameLen = 240;
  const hop = 80;
  const nFft = 512;
  const numCepstral = 19;
  const numFilters = 26;

  // Exact numpy.hamming(240): 0.54 - 0.46 * cos(2*pi*n/(M-1))
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

    // Filterbank energies with (1.0 / nFft) * power as in python
    const logMelEnergies = new Float32Array(numFilters);
    const scale = 1.0 / nFft;
    for (let m = 0; m < numFilters; m++) {
      const filter = fbank[m];
      let energy = 0.0;
      for (let k = 0; k < 257; k++) {
        energy += (power[k] * scale) * filter[k];
      }
      // Natural log (np.log) with eps floor
      logMelEnergies[m] = Math.log(Math.max(2.22e-16, energy));
    }

    // Apply DCT-II matrix
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

  // Compute mean and std across frames
  const result = [];
  for (let c = 0; c < numCepstral; c++) {
    let sum = 0, sumSq = 0;
    for (let f = 0; f < numFrames; f++) {
      const v = mfccFrames[f][c];
      sum += v; sumSq += v * v;
    }
    const m = sum / numFrames;
    result.push(m);
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

// -----------------------------------------------------------------------------
// 4. Spectral Shape Descriptors (10)
// -----------------------------------------------------------------------------

function extractSpectral10(signal, fs = 8000) {
  const nperseg = 256;
  const noverlap = 128;
  const hop = nperseg - noverlap; // 128
  const pad = nperseg >> 1; // 128 zero-padding on each side (boundary='zeros')

  const paddedLen = signal.length + 2 * pad;
  const paddedSig = new Float32Array(paddedLen);
  for (let i = 0; i < signal.length; i++) {
    paddedSig[pad + i] = signal[i];
  }

  // scipy Hann window: 0.5 * (1 - cos(2*pi*n/256))
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

// -----------------------------------------------------------------------------
// 5. Temporal & Hilbert Envelope Descriptors (8)
// -----------------------------------------------------------------------------

function extractTemporal8(signal) {
  const n = signal.length;
  if (n === 0) return [0, 0, 0, 0, 0, 0, 0, 0];

  // 1. ZCR
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

  // 2. RMS & Crest Factor
  let sumSq = 0, peakVal = 0;
  for (let i = 0; i < n; i++) {
    const abs = Math.abs(signal[i]);
    if (abs > peakVal) peakVal = abs;
    sumSq += signal[i] * signal[i];
  }
  const rmsVal = Math.sqrt(sumSq / n);
  const crestFactor = peakVal / (rmsVal + 1e-8);

  // 3. Exact Discrete Hilbert Analytic Envelope via FFT
  // Next power of 2 for fast radix-2 FFT
  let nFft = 1;
  while (nFft < n) nFft <<= 1;
  const rFft = new Float32Array(nFft);
  const iFft = new Float32Array(nFft);
  for (let i = 0; i < n; i++) rFft[i] = signal[i];

  fftRadix2(rFft, iFft, false);

  // Apply Hilbert filter in frequency domain
  // H[0] = 1, H[nFft/2] = 1, H[1..nFft/2-1] = 2, H[nFft/2+1..nFft-1] = 0
  const halfN = nFft >> 1;
  for (let k = 1; k < halfN; k++) {
    rFft[k] *= 2.0;
    iFft[k] *= 2.0;
  }
  for (let k = halfN + 1; k < nFft; k++) {
    rFft[k] = 0.0;
    iFft[k] = 0.0;
  }

  // Inverse FFT
  fftRadix2(rFft, iFft, true);

  // Amplitude envelope is magnitude of analytic signal (first n samples)
  const env = new Float32Array(n);
  let envSum = 0, envSumSq = 0;
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
    m4 = (m4 / (n * envStd * envStd * envStd * envStd)) - 3.0; // Fisher's kurtosis
  }

  return [zcrMean, zcrStd, rmsVal, crestFactor, envMean, envStd, m3, m4];
}

// -----------------------------------------------------------------------------
// Combined 80-Feature Extraction
// -----------------------------------------------------------------------------

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

// -----------------------------------------------------------------------------
// GBDT Predictor
// -----------------------------------------------------------------------------

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

function predictPCG(signal, modelData) {
  const feats = extractAll80Features(signal, modelData);
  const numClasses = modelData.classes.length;
  const raw = modelData.init_raw.slice();
  const lr = modelData.learning_rate;

  for (let m = 0; m < modelData.n_estimators; m++) {
    const stageTrees = modelData.trees[m];
    for (let k = 0; k < numClasses; k++) {
      raw[k] += lr * evalTree(stageTrees[k], feats);
    }
  }

  // Softmax
  let maxRaw = -Infinity;
  for (let k = 0; k < numClasses; k++) {
    if (raw[k] > maxRaw) maxRaw = raw[k];
  }
  let expSum = 0;
  const probs = {};
  for (let k = 0; k < numClasses; k++) {
    const e = Math.exp(raw[k] - maxRaw);
    probs[modelData.classes[k]] = e;
    expSum += e;
  }
  for (let k = 0; k < numClasses; k++) {
    probs[modelData.classes[k]] = probs[modelData.classes[k]] / expSum;
  }
  return { probs, feats };
}

// -----------------------------------------------------------------------------
// Run Test on all 5 sample files
// -----------------------------------------------------------------------------

const samples = {
  'N': 'samples/normal_sample.wav',
  'AS': 'samples/aortic_stenosis_sample.wav',
  'MR': 'samples/mitral_regurgitation_sample.wav',
  'MS': 'samples/mitral_stenosis_sample.wav',
  'MVP': 'samples/mitral_valve_prolapse_sample.wav'
};

console.log("=== Node.js In-Browser GBDT Engine Test ===");
for (const [trueCls, filePath] of Object.entries(samples)) {
  const { signal } = readWavFile(filePath);
  const { probs } = predictPCG(signal, PCG_MODEL_DATA);
  let bestCls = 'N', bestProb = 0;
  for (const [c, p] of Object.entries(probs)) {
    if (p > bestProb) { bestProb = p; bestCls = c; }
  }
  const pctStr = Object.entries(probs).map(([c, p]) => `${c}: ${(p*100).toFixed(1)}%`).join(', ');
  const pass = bestCls === trueCls ? "PASS" : "FAIL";
  console.log(`[${pass}] True: ${trueCls.padEnd(4)} -> Pred: ${bestCls.padEnd(4)} (${(bestProb * 100).toFixed(1)}%) | ${pctStr}`);
}

module.exports = { readWavFile, extractAll80Features, predictPCG };
