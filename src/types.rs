/// Datos de un sensor IMU: [ax, ay, az, qw, qx, qy, qz]
pub type SensorData = [f32; 7];

/// Frame completo con datos de los 5 sensores
pub type SensorFrame = [Option<SensorData>; 5];

/// Una muestra compacta de los 5 sensores con 7 canales cada uno
#[derive(Debug, Clone, Copy, Default)]
pub struct SampleFrame {
    /// Aceleración en x para 5 sensores: [sensor0, sensor1, sensor2, sensor3, sensor4]
    pub ax: [f32; 5],
    pub ay: [f32; 5],
    pub az: [f32; 5],
    
    /// Cuaterniones (qw, qx, qy, qz) para 5 sensores
    pub qw: [f32; 5],
    pub qx: [f32; 5],
    pub qy: [f32; 5],
    pub qz: [f32; 5],
}

impl SampleFrame {
    /// Crea un SampleFrame desde un SensorFrame de BLE
    pub fn from_sensor_frame(frame: &SensorFrame) -> Self {
        let mut sample = Self::default();
        
        for (i, sensor_opt) in frame.iter().enumerate() {
            if let Some(sensor_data) = sensor_opt {
                // sensor_data = [ax, ay, az, qw, qx, qy, qz]
                sample.ax[i] = sensor_data[0];
                sample.ay[i] = sensor_data[1];
                sample.az[i] = sensor_data[2];
                sample.qw[i] = sensor_data[3];
                sample.qx[i] = sensor_data[4];
                sample.qy[i] = sensor_data[5];
                sample.qz[i] = sensor_data[6];
            }
        }
        
        sample
    }
    
    /// Calcula el valor máximo absoluto de aceleración en todos los ejes y sensores
    pub fn max_abs_accel(&self) -> f32 {
        let mut max = 0.0f32;
        for i in 0..5 {
            max = max.max(self.ax[i].abs());
            max = max.max(self.ay[i].abs());
            max = max.max(self.az[i].abs());
        }
        max
    }
    
    /// Convierte a formato plano [timestep * 5 * 7] para procesamiento
    /// Layout: [t][s][c] donde t=timestep, s=sensor, c=channel
    pub fn to_flat_array(&self, output: &mut [f32], time_idx: usize) {
        for s in 0..5 {
            let base_idx = time_idx * 5 * 7 + s * 7;
            output[base_idx + 0] = self.ax[s];
            output[base_idx + 1] = self.ay[s];
            output[base_idx + 2] = self.az[s];
            output[base_idx + 3] = self.qw[s];
            output[base_idx + 4] = self.qx[s];
            output[base_idx + 5] = self.qy[s];
            output[base_idx + 6] = self.qz[s];
        }
    }
}

/// Tipo de ventana: [16 timesteps x SampleFrame]
pub type GestureWindow = Vec<SampleFrame>;

/// 5 ventanas con diferentes offsets para sistema de votación
pub type VotingWindows = [GestureWindow; 5];

/// Constantes del sistema
pub const WINDOW_SIZE: usize = 16;
pub const NUM_SENSORS: usize = 5;
pub const NUM_CHANNELS: usize = 7; // ax, ay, az, qw, qx, qy, qz
pub const TOTAL_WINDOW_FEATURES: usize = WINDOW_SIZE * NUM_SENSORS * NUM_CHANNELS; // 560
pub const TOTAL_EXTRACTED_FEATURES: usize = 250; // después de feature extraction
pub const SAMPLING_RATE: f32 = 64.0; // Hz
