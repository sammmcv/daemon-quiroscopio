use crate::ble::SensorFrame;
use std::collections::VecDeque;
use std::fs::{self, File};
use std::io::Write;

/// Una muestra compacta de los 5 sensores con 7 canales cada uno
/// Equivalente a SampleFrame en C++
pub type ExtractorFrame = SensorFrame;

/// Parámetros de configuración del extractor
#[derive(Debug, Clone)]
pub struct ExtractorParams {
    /// Tamaño de ventana fija (default: 64)
    pub fixed_len: usize,
    /// Umbral alto para detección (|acc| máx) (default: 10.0)
    pub high_thr: f32,
    /// Ratio para umbral bajo (histéresis) (default: 0.45)
    pub low_thr_ratio: f32,
    /// Mínimo de frames para considerar gesto válido (default: 6)
    pub min_len: usize,
    /// Frames de anti-rebote después de guardar (default: 20)
    pub cooldown_frames: usize,
    /// Directorio de salida para CSVs (default: "gestos_auto")
    pub out_dir: String,
    /// Prefijo para archivos CSV (default: "gesto_")
    pub prefix: String,
}

impl Default for ExtractorParams {
    fn default() -> Self {
        Self {
            fixed_len: 64,
            high_thr: 10.0,
            low_thr_ratio: 0.45,
            min_len: 6,
            cooldown_frames: 20,
            out_dir: "gestos_auto".to_string(),
            prefix: "gesto_".to_string(),
        }
    }
}

/// Estados de la máquina de estados del extractor
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    /// Esperando señal fuerte
    Idle,
    /// Capturando datos del gesto
    Capturing,
    /// Esperando frames adicionales después del gesto
    PostCapture,
    /// Anti-rebote después de guardar
    Cooldown,
}

/// Extractor automático de gestos
/// Equivalente a la clase GestureExtractor de C++
pub struct GestureExtractor {
    params: ExtractorParams,
    low_thr: f32,
    state: State,
    below_cnt: usize,
    post_capture_left: usize,
    cooldown_left: usize,
    file_idx: u64,
    
    /// Buffer de captura (frames desde que se detectó el umbral alto)
    capture: Vec<ExtractorFrame>,
    
    /// Buffer circular de últimos fixed_len*2 frames (128 por defecto)
    /// Para tener contexto antes Y después del pico
    buffer: VecDeque<ExtractorFrame>,
    
    /// Callback que se ejecuta cuando se detecta un gesto
    callback: Option<Box<dyn FnMut(&[ExtractorFrame]) + Send>>,
}

impl GestureExtractor {
    /// Crea un nuevo extractor con los parámetros dados
    pub fn new(params: ExtractorParams) -> Self {
        // Crear directorio de salida si no existe
        let _ = fs::create_dir_all(&params.out_dir);
        
        let low_thr = params.high_thr * params.low_thr_ratio;
        
        Self {
            params,
            low_thr,
            state: State::Idle,
            below_cnt: 0,
            post_capture_left: 0,
            cooldown_left: 0,
            file_idx: 0,
            capture: Vec::new(),
            buffer: VecDeque::with_capacity(128),
            callback: None,
        }
    }
    
    /// Establece el callback que se ejecutará cuando se detecte un gesto
    /// El callback recibe la ventana de 64 frames centrada en el pico
    pub fn set_callback<F>(&mut self, callback: F)
    where
        F: FnMut(&[ExtractorFrame]) + Send + 'static,
    {
        self.callback = Some(Box::new(callback));
    }
    
    /// Alimenta el extractor con un nuevo frame
    /// Llamar por cada tick del sistema
    pub fn feed(&mut self, frame: ExtractorFrame) {
        // Mantener buffer circular de últimos fixed_len*2 frames
        self.buffer.push_back(frame);
        if self.buffer.len() > self.params.fixed_len * 2 {
            self.buffer.pop_front();
        }
        
        let max_acc = Self::frame_max_abs(&frame);
        
        match self.state {
            State::Idle => {
                if max_acc >= self.params.high_thr {
                    self.state = State::Capturing;
                    self.capture.clear();
                    self.capture.push(frame);
                }
            }
            
            State::Capturing => {
                self.capture.push(frame);
                
                // Salir cuando cae por debajo del umbral bajo durante un "hold"
                if max_acc < self.low_thr {
                    self.below_cnt += 1;
                } else {
                    self.below_cnt = 0;
                }
                
                if self.below_cnt >= 2 {
                    // 2 frames seguidos por debajo del umbral bajo
                    self.below_cnt = 0;
                    
                    if self.capture.len() >= self.params.min_len {
                        // Gesto válido, esperar frames posteriores
                        self.state = State::PostCapture;
                        self.post_capture_left = self.params.fixed_len / 2; // 32 frames más
                    } else {
                        // Demasiado corto, descartar
                        self.state = State::Cooldown;
                        self.cooldown_left = self.params.cooldown_frames / 2;
                    }
                }
            }
            
            State::PostCapture => {
                // Seguir llenando el buffer después del gesto
                if self.post_capture_left > 0 {
                    self.post_capture_left -= 1;
                }
                
                if self.post_capture_left == 0 {
                    self.save_current_capture();
                    self.state = State::Cooldown;
                    self.cooldown_left = self.params.cooldown_frames;
                }
            }
            
            State::Cooldown => {
                if self.cooldown_left > 0 {
                    self.cooldown_left -= 1;
                }
                
                if self.cooldown_left == 0 {
                    self.state = State::Idle;
                }
            }
        }
    }
    
    /// Calcula el valor absoluto máximo de aceleración en un frame
    fn frame_max_abs(frame: &ExtractorFrame) -> f32 {
        let mut max = 0.0f32;
        
        for sensor_data in frame.iter().flatten() {
            // sensor_data = [ax, ay, az, qw, qx, qy, qz]
            // Solo consideramos aceleraciones (índices 0, 1, 2)
            max = max.max(sensor_data[0].abs());
            max = max.max(sensor_data[1].abs());
            max = max.max(sensor_data[2].abs());
        }
        
        max
    }
    
    /// Guarda la captura actual: busca el pico, extrae ventana centrada,
    /// llama al callback y escribe CSV
    fn save_current_capture(&mut self) {
        if self.buffer.is_empty() {
            return;
        }
        
        // 1) Encontrar el índice del pico en el buffer circular
        let mut peak_idx = 0;
        let mut best = -1.0f32;
        
        for (i, frame) in self.buffer.iter().enumerate() {
            let v = Self::frame_max_abs(frame);
            if v > best {
                best = v;
                peak_idx = i;
            }
        }
        
        // 2) Construir ventana de L frames centrada en peak_idx
        let l = self.params.fixed_len;
        let half = l / 2;
        
        // Rango deseado: [peak_idx - half, peak_idx + half)
        let start_idx = peak_idx as i32 - half as i32;
        
        let mut out = Vec::with_capacity(l);
        
        for t in 0..l {
            let src_idx = start_idx + t as i32;
            
            // Tomar datos reales de buffer_ si están disponibles
            let frame = if src_idx >= 0 && (src_idx as usize) < self.buffer.len() {
                self.buffer[src_idx as usize]
            } else if src_idx < 0 {
                // Antes del inicio: usar el primer frame del buffer
                self.buffer[0]
            } else {
                // Después del final: usar el último frame del buffer
                *self.buffer.back().unwrap()
            };
            
            out.push(frame);
        }
        
        // 3) Notificar al callback con la ventana extraída
        if let Some(ref mut callback) = self.callback {
            callback(&out);
        }
        
        // 4) Escribir CSV
        let filename = format!(
            "{}/{}_{:05}.csv",
            self.params.out_dir,
            self.params.prefix,
            self.file_idx
        );
        
        self.file_idx += 1;
        
        if let Err(e) = self.write_csv(&filename, &out) {
            eprintln!("Error escribiendo CSV {}: {}", filename, e);
        }
    }
    
    /// Escribe la ventana en formato CSV
    fn write_csv(&self, path: &str, window: &[ExtractorFrame]) -> std::io::Result<()> {
        let mut file = File::create(path)?;
        
        // Escribir encabezado
        writeln!(file, "sample,sensor,ax,ay,az,w,i,j,k")?;
        
        // Escribir datos
        for (t, frame) in window.iter().enumerate() {
            for (s, sensor_opt) in frame.iter().enumerate() {
                if let Some(sensor_data) = sensor_opt {
                    // sensor_data = [ax, ay, az, qw, qx, qy, qz]
                    writeln!(
                        file,
                        "{},{},{},{},{},{},{},{},{}",
                        t,
                        s,
                        sensor_data[0], // ax
                        sensor_data[1], // ay
                        sensor_data[2], // az
                        sensor_data[3], // qw
                        sensor_data[4], // qx
                        sensor_data[5], // qy
                        sensor_data[6], // qz
                    )?;
                }
            }
        }
        
        Ok(())
    }
    
    /// Retorna el estado actual del extractor (para debugging)
    pub fn state(&self) -> &str {
        match self.state {
            State::Idle => "IDLE",
            State::Capturing => "CAPTURING",
            State::PostCapture => "POST_CAPTURE",
            State::Cooldown => "COOLDOWN",
        }
    }
    
    /// Retorna el tamaño actual del buffer circular
    pub fn buffer_len(&self) -> usize {
        self.buffer.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_frame(acc_magnitude: f32) -> ExtractorFrame {
        let sensor_data = [acc_magnitude, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        [
            Some(sensor_data),
            Some(sensor_data),
            Some(sensor_data),
            Some(sensor_data),
            Some(sensor_data),
        ]
    }

    #[test]
    fn test_idle_to_capturing() {
        let mut extractor = GestureExtractor::new(ExtractorParams::default());
        assert_eq!(extractor.state(), "IDLE");
        
        // Alimentar con frame de alta aceleración
        extractor.feed(create_test_frame(15.0));
        assert_eq!(extractor.state(), "CAPTURING");
    }

    #[test]
    fn test_buffer_circular() {
        let mut extractor = GestureExtractor::new(ExtractorParams::default());
        
        // Alimentar más de 128 frames
        for i in 0..200 {
            extractor.feed(create_test_frame(i as f32 / 100.0));
        }
        
        // El buffer debe mantener máximo 128 frames
        assert!(extractor.buffer_len() <= 128);
    }

    #[test]
    fn test_callback_execution() {
        use std::sync::{Arc, Mutex};
        
        let mut extractor = GestureExtractor::new(ExtractorParams {
            fixed_len: 64,
            high_thr: 10.0,
            low_thr_ratio: 0.45,
            min_len: 6,
            cooldown_frames: 20,
            out_dir: "/tmp/gestos_test".to_string(),
            prefix: "test_".to_string(),
        });
        
        let callback_called = Arc::new(Mutex::new(false));
        let callback_called_clone = Arc::clone(&callback_called);
        
        extractor.set_callback(move |_frames| {
            *callback_called_clone.lock().unwrap() = true;
        });
        
        // Simular un gesto: 100 frames de bajo → 20 de alto → 100 de bajo
        for _ in 0..100 {
            extractor.feed(create_test_frame(1.0));
        }
        
        for _ in 0..20 {
            extractor.feed(create_test_frame(15.0));
        }
        
        for _ in 0..100 {
            extractor.feed(create_test_frame(1.0));
        }
        
        // El callback debe haberse ejecutado
        assert!(*callback_called.lock().unwrap());
    }
}
