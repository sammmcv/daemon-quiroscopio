use crate::ble::SensorFrame;
use std::collections::VecDeque;

const WINDOW_SIZE: usize = 64;
const SENSORS: usize = 5;
const CHANNELS: usize = 7;
const CAPTURE_SIZE: usize = 200; // Buffer extendido para capturar gesto completo

/// Buffer circular para acumular ventanas de 64 muestras
pub struct GestureBuffer {
    buffer: VecDeque<SensorFrame>,
    window_size: usize,
    max_buffer_size: usize,
}

impl GestureBuffer {
    /// Crea un nuevo buffer con tamaño de ventana especificado
    pub fn new() -> Self {
        Self {
            buffer: VecDeque::with_capacity(CAPTURE_SIZE),
            window_size: WINDOW_SIZE,
            max_buffer_size: CAPTURE_SIZE,
        }
    }

    /// Añade un nuevo frame al buffer
    pub fn push(&mut self, frame: SensorFrame) {
        self.buffer.push_back(frame);
        
        // Mantener el buffer de captura extendido
        if self.buffer.len() > self.max_buffer_size {
            self.buffer.pop_front();
        }
    }

    /// Verifica si hay suficientes datos para formar una ventana completa
    pub fn is_ready(&self) -> bool {
        self.buffer.len() >= self.window_size
    }

    /// Extrae la ventana más reciente de 64 muestras en formato [64, 5, 7]
    /// Si algún sensor está ausente en un frame, usa el último valor conocido
    pub fn get_window(&self) -> Option<[[[f32; CHANNELS]; SENSORS]; WINDOW_SIZE]> {
        if !self.is_ready() {
            return None;
        }

        let mut window = [[[0.0f32; CHANNELS]; SENSORS]; WINDOW_SIZE];
        let mut last_known = [[0.0f32; CHANNELS]; SENSORS];
        let mut has_valid_data = false;
        
        // Tomar los últimos 64 frames
        let start_idx = self.buffer.len() - self.window_size;
        
        for (t, frame) in self.buffer.iter().skip(start_idx).enumerate() {
            for s in 0..SENSORS {
                if let Some(sensor_data) = frame[s] {
                    // Tenemos datos del sensor, actualizar last_known y window
                    last_known[s] = sensor_data;
                    window[t][s] = sensor_data;
                    has_valid_data = true;
                } else {
                    // Sensor ausente, usar último valor conocido
                    window[t][s] = last_known[s];
                }
            }
        }

        // Verificar que al menos tengamos algunos datos válidos
        if !has_valid_data {
            return None;
        }

        Some(window)
    }
    
    /// Calcula la energía de aceleración en una ventana (para detección de gestos)
    /// Retorna la energía promedio de todos los sensores
    pub fn get_energy(&self) -> f32 {
        if self.buffer.len() < 2 {
            return 0.0;
        }
        
        let start_idx = if self.buffer.len() >= self.window_size {
            self.buffer.len() - self.window_size
        } else {
            0
        };
        
        let mut total_energy = 0.0;
        let mut sample_count = 0;
        
        // Calcular energía de aceleración para cada sensor
        for frame in self.buffer.iter().skip(start_idx) {
            for s in 0..SENSORS {
                if let Some(sensor_data) = frame[s] {
                    // Energía: ax^2 + ay^2 + az^2
                    let energy = sensor_data[0].powi(2) + 
                                 sensor_data[1].powi(2) + 
                                 sensor_data[2].powi(2);
                    total_energy += energy;
                    sample_count += 1;
                }
            }
        }
        
        if sample_count > 0 {
            total_energy / sample_count as f32
        } else {
            0.0
        }
    }
    
    /// Detecta si hay un gesto completo en el buffer usando umbrales de energía
    /// Retorna true si detectó inicio y fin de gesto
    pub fn detect_gesture(&self, threshold_on: f32, threshold_off: f32) -> bool {
        if self.buffer.len() < self.window_size {
            return false;
        }
        
        let mut in_gesture = false;
        let mut gesture_started = false;
        let mut gesture_ended = false;
        
        // Aplicar filtro pasa-altas simple (eliminar componente DC)
        let mut last_values: Vec<[f32; 3]> = vec![[0.0; 3]; SENSORS];
        let alpha = 0.99; // Factor de filtro pasa-altas
        
        for frame in self.buffer.iter() {
            let mut frame_energy = 0.0;
            let mut valid_sensors = 0;
            
            for s in 0..SENSORS {
                if let Some(sensor_data) = frame[s] {
                    // Aplicar filtro pasa-altas en aceleraciones
                    let filtered = [
                        alpha * (last_values[s][0] + sensor_data[0] - last_values[s][0]),
                        alpha * (last_values[s][1] + sensor_data[1] - last_values[s][1]),
                        alpha * (last_values[s][2] + sensor_data[2] - last_values[s][2]),
                    ];
                    
                    last_values[s] = [sensor_data[0], sensor_data[1], sensor_data[2]];
                    
                    // Calcular energía filtrada
                    frame_energy += filtered[0].powi(2) + filtered[1].powi(2) + filtered[2].powi(2);
                    valid_sensors += 1;
                }
            }
            
            if valid_sensors > 0 {
                frame_energy /= valid_sensors as f32;
                
                // Detectar inicio de gesto
                if !in_gesture && frame_energy > threshold_on {
                    in_gesture = true;
                    gesture_started = true;
                }
                
                // Detectar fin de gesto
                if in_gesture && frame_energy < threshold_off {
                    gesture_ended = true;
                    break;
                }
            }
        }
        
        gesture_started && gesture_ended
    }
    
    /// Encuentra el pico de energía y extrae una ventana de 64 muestras centrada en él
    /// Retorna la ventana centrada y el índice del pico
    pub fn get_centered_window(&self) -> Option<([[[f32; CHANNELS]; SENSORS]; WINDOW_SIZE], usize)> {
        if self.buffer.len() < self.window_size {
            return None;
        }
        
        // 1. Calcular energía para cada frame con filtro pasa-altas
        let mut energies = Vec::with_capacity(self.buffer.len());
        let mut last_values: Vec<[f32; 3]> = vec![[0.0; 3]; SENSORS];
        let alpha = 0.99;
        
        for frame in self.buffer.iter() {
            let mut frame_energy = 0.0;
            let mut valid_sensors = 0;
            
            for s in 0..SENSORS {
                if let Some(sensor_data) = frame[s] {
                    // Aplicar filtro pasa-altas
                    let filtered = [
                        alpha * (last_values[s][0] + sensor_data[0] - last_values[s][0]),
                        alpha * (last_values[s][1] + sensor_data[1] - last_values[s][1]),
                        alpha * (last_values[s][2] + sensor_data[2] - last_values[s][2]),
                    ];
                    
                    last_values[s] = [sensor_data[0], sensor_data[1], sensor_data[2]];
                    
                    frame_energy += filtered[0].powi(2) + filtered[1].powi(2) + filtered[2].powi(2);
                    valid_sensors += 1;
                }
            }
            
            if valid_sensors > 0 {
                energies.push(frame_energy / valid_sensors as f32);
            } else {
                energies.push(0.0);
            }
        }
        
        // 2. Encontrar el índice del pico máximo
        let peak_idx = energies
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)?;
        
        // 3. Calcular rango de extracción centrado en el pico
        let half_window = self.window_size / 2;
        let start_idx = if peak_idx >= half_window {
            peak_idx - half_window
        } else {
            0
        };
        
        let end_idx = start_idx + self.window_size;
        
        // Verificar que hay suficientes datos
        if end_idx > self.buffer.len() {
            return None;
        }
        
        // 4. Extraer la ventana centrada
        let mut window = [[[0.0f32; CHANNELS]; SENSORS]; WINDOW_SIZE];
        let mut last_known = [[0.0f32; CHANNELS]; SENSORS];
        
        for (w_idx, frame) in self.buffer.iter().skip(start_idx).take(self.window_size).enumerate() {
            for s in 0..SENSORS {
                if let Some(sensor_data) = frame[s] {
                    window[w_idx][s].copy_from_slice(&sensor_data);
                    last_known[s].copy_from_slice(&sensor_data);
                } else {
                    // Usar último valor conocido si el sensor no está presente
                    window[w_idx][s].copy_from_slice(&last_known[s]);
                }
            }
        }
        
        Some((window, peak_idx))
    }

    /// Obtiene el número de frames acumulados
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Limpia el buffer
    pub fn clear(&mut self) {
        self.buffer.clear();
    }
    
    /// Exporta el buffer completo a formato CSV (igual que los archivos de entrenamiento)
    /// Retorna String con formato: sample,sensor,ax,ay,az,w,i,j,k
    pub fn to_csv(&self) -> String {
        let mut csv = String::from("sample,sensor,ax,ay,az,w,i,j,k\n");
        
        for (sample_idx, frame) in self.buffer.iter().enumerate() {
            for sensor_id in 0..SENSORS {
                if let Some(sensor_data) = frame[sensor_id] {
                    csv.push_str(&format!(
                        "{},{},{},{},{},{},{},{},{}\n",
                        sample_idx,
                        sensor_id,
                        sensor_data[0], // ax
                        sensor_data[1], // ay
                        sensor_data[2], // az
                        sensor_data[3], // w (quaternion)
                        sensor_data[4], // i
                        sensor_data[5], // j
                        sensor_data[6]  // k
                    ));
                }
            }
        }
        
        csv
    }
}

impl Default for GestureBuffer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_not_ready_initially() {
        let buffer = GestureBuffer::new();
        assert!(!buffer.is_ready());
        assert!(buffer.get_window().is_none());
    }

    #[test]
    fn test_buffer_ready_after_64_frames() {
        let mut buffer = GestureBuffer::new();
        
        // Añadir 64 frames con datos de prueba
        for _ in 0..64 {
            let mut frame: SensorFrame = [None; 5];
            for s in 0..5 {
                frame[s] = Some([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0]);
            }
            buffer.push(frame);
        }

        assert!(buffer.is_ready());
        assert!(buffer.get_window().is_some());
    }

    #[test]
    fn test_sliding_window() {
        let mut buffer = GestureBuffer::new();
        
        // Añadir 70 frames
        for i in 0..70 {
            let mut frame: SensorFrame = [None; 5];
            frame[0] = Some([i as f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]);
            buffer.push(frame);
        }

        // La ventana debe contener los últimos 64 frames (6..69)
        let window = buffer.get_window().unwrap();
        assert_eq!(window[0][0][0], 6.0); // Primer frame de la ventana
        assert_eq!(window[63][0][0], 69.0); // Último frame de la ventana
    }

    #[test]
    fn test_missing_sensor_interpolation() {
        let mut buffer = GestureBuffer::new();
        
        // Añadir frames con sensor 1 ausente en algunos
        for i in 0..64 {
            let mut frame: SensorFrame = [None; 5];
            frame[0] = Some([i as f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]);
            
            // Sensor 1 solo presente en frames pares
            if i % 2 == 0 {
                frame[1] = Some([100.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]);
            }
            
            buffer.push(frame);
        }

        let window = buffer.get_window().unwrap();
        
        // En frames impares, sensor 1 debe tener el último valor conocido
        assert_eq!(window[1][1][0], 100.0); // Frame 1 usa valor de frame 0
        assert_eq!(window[3][1][0], 100.0); // Frame 3 usa valor de frame 2
    }
}