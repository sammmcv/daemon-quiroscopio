use crate::ble::SensorFrame;
use std::collections::VecDeque;

const WINDOW_SIZE: usize = 64;
const SENSORS: usize = 5;
const CHANNELS: usize = 7;

/// Buffer circular para acumular ventanas de 64 muestras
pub struct GestureBuffer {
    buffer: VecDeque<SensorFrame>,
    window_size: usize,
}

impl GestureBuffer {
    /// Crea un nuevo buffer con tamaño de ventana especificado
    pub fn new() -> Self {
        Self {
            buffer: VecDeque::with_capacity(WINDOW_SIZE * 2),
            window_size: WINDOW_SIZE,
        }
    }

    /// Añade un nuevo frame al buffer
    pub fn push(&mut self, frame: SensorFrame) {
        self.buffer.push_back(frame);
        
        // Mantener solo el doble del tamaño de ventana para permitir sliding
        if self.buffer.len() > self.window_size * 2 {
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
    
    /// Calcula la magnitud del movimiento (aceleración) en la ventana
    /// Retorna la desviación estándar promedio de la aceleración de todos los sensores
    pub fn get_movement_magnitude(&self) -> f32 {
        if self.buffer.len() < 2 {
            return 0.0;
        }
        
        let start_idx = if self.buffer.len() >= self.window_size {
            self.buffer.len() - self.window_size
        } else {
            0
        };
        
        let mut total_variance = 0.0;
        let mut sensor_count = 0;
        
        // Calcular varianza de aceleración para cada sensor
        for s in 0..SENSORS {
            let mut acc_values: Vec<f32> = Vec::new();
            
            for frame in self.buffer.iter().skip(start_idx) {
                if let Some(sensor_data) = frame[s] {
                    // Magnitud de aceleración: sqrt(ax^2 + ay^2 + az^2)
                    let acc_mag = (sensor_data[0].powi(2) + 
                                   sensor_data[1].powi(2) + 
                                   sensor_data[2].powi(2)).sqrt();
                    acc_values.push(acc_mag);
                }
            }
            
            if acc_values.len() > 1 {
                let mean = acc_values.iter().sum::<f32>() / acc_values.len() as f32;
                let variance = acc_values.iter()
                    .map(|&x| (x - mean).powi(2))
                    .sum::<f32>() / acc_values.len() as f32;
                total_variance += variance.sqrt(); // Desviación estándar
                sensor_count += 1;
            }
        }
        
        if sensor_count > 0 {
            total_variance / sensor_count as f32
        } else {
            0.0
        }
    }

    /// Obtiene el número de frames acumulados
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Limpia el buffer
    pub fn clear(&mut self) {
        self.buffer.clear();
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