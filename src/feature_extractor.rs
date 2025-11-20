use crate::types::{GestureWindow, WINDOW_SIZE, NUM_SENSORS, SAMPLING_RATE};
use rustfft::{FftPlanner, num_complex::Complex};

/// Bandas de frecuencia para análisis FFT
const FREQUENCY_BANDS: [(f32, f32); 5] = [
    (0.5, 2.0),
    (2.0, 4.0),
    (4.0, 8.0),
    (8.0, 12.0),
    (12.0, 20.0),
];

pub struct FeatureExtractor {
    planner: FftPlanner<f32>,
}

impl FeatureExtractor {
    pub fn new() -> Self {
        Self {
            planner: FftPlanner::new(),
        }
    }
    
    /// Extrae 250 características de una ventana [16x5x7]
    /// Replica la funcionalidad de feature_extractor.cpp
    pub fn extract(&mut self, window: &GestureWindow) -> Vec<f32> {
        let mut features = Vec::with_capacity(250);
        
        // Procesar cada sensor (5 sensores)
        for sensor_idx in 0..NUM_SENSORS {
            let (ax, ay, az, qw, qx, qy, qz) = self.extract_sensor_channels(window, sensor_idx);
            
            // Features temporales para aceleraciones (7 features x 3 ejes = 21)
            for signal in [&ax, &ay, &az] {
                features.extend(self.time_domain_features(signal));
            }
            
            // Features de cuaterniones (4 features)
            features.extend(self.quaternion_features(&qw, &qx, &qy, &qz));
            
            // Features FFT para aceleraciones (8 features x 3 ejes = 24)
            for signal in [&ax, &ay, &az] {
                features.extend(self.fft_features(signal));
            }
        }
        // Total por sensor: 21 + 4 + 24 = 49
        // Total 5 sensores: 49 * 5 = 245
        
        // Features globales del sensor 0 (5 features)
        let (ax0, ay0, az0, _, _, _, _) = self.extract_sensor_channels(window, 0);
        features.push(ax0.iter().sum());           // sum_ax
        features.push(ay0.iter().sum());           // sum_ay
        features.push(az0.iter().sum());           // sum_az
        features.push(ax0.last().unwrap() - ax0.first().unwrap()); // delta_ax
        features.push(ay0.last().unwrap() - ay0.first().unwrap()); // delta_ay
        
        assert_eq!(features.len(), 250, "Expected 250 features, got {}", features.len());
        features
    }
    
    /// Extrae canales individuales de un sensor
    fn extract_sensor_channels(&self, window: &GestureWindow, sensor_idx: usize) 
        -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) 
    {
        let mut ax = Vec::with_capacity(WINDOW_SIZE);
        let mut ay = Vec::with_capacity(WINDOW_SIZE);
        let mut az = Vec::with_capacity(WINDOW_SIZE);
        let mut qw = Vec::with_capacity(WINDOW_SIZE);
        let mut qx = Vec::with_capacity(WINDOW_SIZE);
        let mut qy = Vec::with_capacity(WINDOW_SIZE);
        let mut qz = Vec::with_capacity(WINDOW_SIZE);
        
        for frame in window {
            ax.push(frame.ax[sensor_idx]);
            ay.push(frame.ay[sensor_idx]);
            az.push(frame.az[sensor_idx]);
            qw.push(frame.qw[sensor_idx]);
            qx.push(frame.qx[sensor_idx]);
            qy.push(frame.qy[sensor_idx]);
            qz.push(frame.qz[sensor_idx]);
        }
        
        (ax, ay, az, qw, qx, qy, qz)
    }
    
    /// 7 características en el dominio del tiempo
    fn time_domain_features(&self, signal: &[f32]) -> Vec<f32> {
        let mean = self.mean(signal);
        let std = self.std(signal);
        
        let mut temp = signal.to_vec();
        let median = self.median(&mut temp);
        let iqr = self.iqr(&mut temp);
        
        let range = self.range(signal);
        let rms = self.rms(signal);
        let mad = self.mean_abs_diff(signal);
        
        vec![mean, std, median, iqr, range, rms, mad]
    }
    
    /// 4 características de cuaterniones (deltas)
    fn quaternion_features(&self, qw: &[f32], qx: &[f32], qy: &[f32], qz: &[f32]) -> Vec<f32> {
        if qw.len() <= 1 {
            return vec![0.0; 4];
        }
        
        // Calcular magnitud del delta quaternion
        let mut deltas = Vec::with_capacity(qw.len() - 1);
        for i in 0..qw.len() - 1 {
            let dw = qw[i + 1] - qw[i];
            let dx = qx[i + 1] - qx[i];
            let dy = qy[i + 1] - qy[i];
            let dz = qz[i + 1] - qz[i];
            deltas.push((dw * dw + dx * dx + dy * dy + dz * dz).sqrt());
        }
        
        let mut temp = deltas.clone();
        vec![
            self.mean(&deltas),
            self.std(&deltas),
            self.max(&deltas),
            self.median(&mut temp),
        ]
    }
    
    /// 8 características FFT (3 base + 5 bandas)
    fn fft_features(&mut self, signal: &[f32]) -> Vec<f32> {
        if signal.is_empty() {
            return vec![0.0; 8];
        }
        
        // Remover media
        let mean = self.mean(signal);
        let mut centered: Vec<Complex<f32>> = signal
            .iter()
            .map(|&x| Complex::new(x - mean, 0.0))
            .collect();
        
        // Calcular FFT
        let fft = self.planner.plan_fft_forward(centered.len());
        fft.process(&mut centered);
        
        // Calcular PSD: |X|² / N
        let n = signal.len() as f32;
        let psd: Vec<f32> = centered
            .iter()
            .take(centered.len() / 2 + 1) // Solo frecuencias positivas
            .map(|c| (c.norm_sqr()) / n)
            .collect();
        
        // Frecuencias
        let freqs: Vec<f32> = (0..psd.len())
            .map(|i| i as f32 / (n / SAMPLING_RATE))
            .collect();
        
        // Frecuencia dominante (ignorando DC)
        let dominant_freq = if psd.len() > 1 {
            let (idx, _) = psd[1..]
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap();
            freqs[idx + 1]
        } else {
            0.0
        };
        
        // Centroide espectral
        let ps_sum: f32 = psd.iter().sum();
        let centroid = if ps_sum > 0.0 {
            freqs.iter().zip(&psd).map(|(f, p)| f * p).sum::<f32>() / ps_sum
        } else {
            0.0
        };
        
        // Entropía espectral
        let entropy = if ps_sum > 0.0 {
            -psd.iter()
                .map(|&p| {
                    let pnorm = p / ps_sum;
                    if pnorm > 1e-12 {
                        pnorm * pnorm.log2()
                    } else {
                        0.0
                    }
                })
                .sum::<f32>()
        } else {
            0.0
        };
        
        let mut features = vec![dominant_freq, centroid, entropy];
        
        // Potencia por bandas
        for &(fmin, fmax) in &FREQUENCY_BANDS {
            features.push(self.bandpower(&psd, &freqs, fmin, fmax));
        }
        
        features
    }
    
    /// Calcula potencia en una banda de frecuencias
    fn bandpower(&self, psd: &[f32], freqs: &[f32], fmin: f32, fmax: f32) -> f32 {
        let mut power = 0.0;
        for i in 1..freqs.len() {
            if freqs[i - 1] >= fmin && freqs[i] < fmax {
                let df = freqs[i] - freqs[i - 1];
                power += 0.5 * (psd[i - 1] + psd[i]) * df;
            }
        }
        power
    }
    
    // ========== Funciones estadísticas ==========
    
    fn mean(&self, data: &[f32]) -> f32 {
        if data.is_empty() { return 0.0; }
        data.iter().sum::<f32>() / data.len() as f32
    }
    
    fn std(&self, data: &[f32]) -> f32 {
        if data.len() <= 1 { return 0.0; }
        let mean = self.mean(data);
        let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / (data.len() - 1) as f32;
        variance.sqrt()
    }
    
    fn median(&self, data: &mut [f32]) -> f32 {
        if data.is_empty() { return 0.0; }
        data.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mid = data.len() / 2;
        if data.len() % 2 == 0 {
            (data[mid - 1] + data[mid]) / 2.0
        } else {
            data[mid]
        }
    }
    
    fn iqr(&self, data: &mut [f32]) -> f32 {
        if data.len() < 4 { return 0.0; }
        data.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let q1_idx = data.len() / 4;
        let q3_idx = 3 * data.len() / 4;
        data[q3_idx] - data[q1_idx]
    }
    
    fn range(&self, data: &[f32]) -> f32 {
        if data.is_empty() { return 0.0; }
        let min = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        max - min
    }
    
    fn rms(&self, data: &[f32]) -> f32 {
        if data.is_empty() { return 0.0; }
        (data.iter().map(|x| x * x).sum::<f32>() / data.len() as f32).sqrt()
    }
    
    fn mean_abs_diff(&self, data: &[f32]) -> f32 {
        if data.len() <= 1 { return 0.0; }
        let sum: f32 = data.windows(2).map(|w| (w[1] - w[0]).abs()).sum();
        sum / (data.len() - 1) as f32
    }
    
    fn max(&self, data: &[f32]) -> f32 {
        data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b))
    }
}

impl Default for FeatureExtractor {
    fn default() -> Self {
        Self::new()
    }
}
