use crate::types::{SampleFrame, WINDOW_SIZE};
use std::collections::VecDeque;
use std::fs::{self, File};
use std::io::Write;

/// Una muestra compacta de los 5 sensores con 7 canales cada uno
/// Equivalente a SampleFrame en C++
pub type ExtractorFrame = SampleFrame;

/// Parámetros de configuración del extractor
#[derive(Debug, Clone)]
pub struct ExtractorParams {
    /// Tamaño de ventana fija (default: 16)
    pub fixed_len: usize,
    /// Umbral alto para detección (|acc| máx) (default: 10.0)
    pub high_thr: f32,
    /// Ratio para umbral bajo (histéresis) (default: 0.45)
    pub low_thr_ratio: f32,
    /// Mínimo de frames para considerar gesto válido (default: 6)
    pub min_len: usize,
    /// Frames de anti-rebote después de guardar (default: 20)
    pub cooldown_frames: usize,
    /// Cantidad de frames entre ventanas de votación (default: 3)
    pub offset_shift: i32,
    /// Directorio de salida para CSVs (default: "gestos_auto")
    pub out_dir: String,
    /// Prefijo para archivos CSV (default: "gesto_")
    pub prefix: String,
}

impl Default for ExtractorParams {
    fn default() -> Self {
        Self {
            fixed_len: WINDOW_SIZE,
            high_thr: 10.0,
            low_thr_ratio: 0.45,
            min_len: 6,
            cooldown_frames: 20,
            offset_shift: 3,
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

    /// Buffer circular de últimos fixed_len*2 frames (32 por defecto, como en C++)
    /// Para tener contexto antes y después del pico
    buffer: VecDeque<ExtractorFrame>,

    /// Callback que se ejecuta cuando se detecta un gesto: entrega 5 ventanas
    /// con offsets [-6, -3, 0, +3, +6] para votación, igual al extractor en C++
    callback: Option<Box<dyn FnMut(&[Vec<ExtractorFrame>; 5]) + Send>>,
}

impl GestureExtractor {
    /// Crea un nuevo extractor con los parámetros dados
    pub fn new(params: ExtractorParams) -> Self {
        // Crear directorio de salida si no existe
        let _ = fs::create_dir_all(&params.out_dir);

        let low_thr = params.high_thr * params.low_thr_ratio;
        let buffer_capacity = params.fixed_len * 2;

        Self {
            params,
            low_thr,
            state: State::Idle,
            below_cnt: 0,
            post_capture_left: 0,
            cooldown_left: 0,
            file_idx: 0,
            capture: Vec::new(),
            buffer: VecDeque::with_capacity(buffer_capacity),
            callback: None,
        }
    }

    /// Establece el callback que se ejecutará cuando se detecte un gesto
    /// El callback recibe 5 ventanas para votación con offsets [-6,-3,0,+3,+6]
    /// multiplicados por `offset_shift` para emular la lógica del extractor C++
    pub fn set_callback<F>(&mut self, callback: F)
    where
        F: FnMut(&[Vec<ExtractorFrame>; 5]) + Send + 'static,
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
                        self.post_capture_left = self.params.fixed_len / 4;
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
        frame.max_abs_accel()
    }

    /// Guarda la captura actual: busca el pico, extrae 5 ventanas alrededor
    /// del pico, llama al callback y escribe CSV de la ventana centrada
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

        // 2) Construir 5 ventanas de L frames con offsets [-6,-3,0,+3,+6], igual que el extractor en C++
        let l = self.params.fixed_len;
        let half = l / 2;
        let offsets = [-6, -3, 0, 3, 6];
        let shift = self.params.offset_shift;
        let mut windows: [Vec<ExtractorFrame>; 5] = [
            Vec::with_capacity(l),
            Vec::with_capacity(l),
            Vec::with_capacity(l),
            Vec::with_capacity(l),
            Vec::with_capacity(l),
        ];

        for (widx, off) in offsets.iter().enumerate() {
            let start_idx = peak_idx as i32 - half as i32 + off * shift;
            for t in 0..l {
                let src_idx = start_idx + t as i32;
                let frame = if src_idx >= 0 && (src_idx as usize) < self.buffer.len() {
                    self.buffer[src_idx as usize]
                } else if src_idx < 0 {
                    self.buffer[0]
                } else {
                    *self.buffer.back().unwrap()
                };
                windows[widx].push(frame);
            }
        }

        // 3) Notificar al callback con las 5 ventanas
        if let Some(ref mut callback) = self.callback {
            callback(&windows);
        }

        // 4) Escribir CSV de la ventana centrada (widx=2)
        let filename = format!(
            "{}/{}{:05}.csv",
            self.params.out_dir, self.params.prefix, self.file_idx
        );

        self.file_idx += 1;

        if let Err(e) = self.write_csv(&filename, &windows[2]) {
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
            for s in 0..5 {
                writeln!(
                    file,
                    "{},{},{},{},{},{},{},{},{}",
                    t,
                    s,
                    frame.ax[s], // ax
                    frame.ay[s], // ay
                    frame.az[s], // az
                    frame.qw[s], // qw
                    frame.qx[s], // qx (i)
                    frame.qy[s], // qy (j)
                    frame.qz[s], // qz (k)
                )?;
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
        let mut frame = ExtractorFrame::default();
        for sensor_idx in 0..5 {
            frame.ax[sensor_idx] = acc_magnitude;
            frame.qw[sensor_idx] = 1.0;
        }
        frame
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
        let max_frames = ExtractorParams::default().fixed_len * 2;

        // Alimentar más de max_frames frames
        for i in 0..200 {
            extractor.feed(create_test_frame(i as f32 / 100.0));
        }

        // El buffer debe mantener máximo fixed_len*2 frames (32 por defecto)
        assert!(extractor.buffer_len() <= max_frames);
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
            offset_shift: 3,
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
