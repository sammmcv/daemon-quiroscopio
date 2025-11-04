/*
Gesture Recognition en Tiempo Real con BLE

Sistema de reconocimiento de gestos que:
1. Recibe datos IMU desde dispositivos BLE (5 sensores)
2. Acumula ventanas de 64 muestras en buffer circular
3. Realiza predicciones en tiempo real usando clasificador Python
4. Implementa votación para estabilizar predicciones

Para compilar y ejecutar:
cargo build --release
./target/release/onnx-predictor <MAC_ADDRESS>

Ejemplo:
./target/release/onnx-predictor 28:CD:C1:08:37:69
*/

mod ble;
mod gesture_buffer;
mod gesture_extractor;

use anyhow::Result;
use crossbeam_channel::{bounded, select};
use numpy::PyArray3;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::VecDeque;
use std::env;
use std::time::Duration;

use ble::{SensorFrame, start_ble_receiver};
use gesture_extractor::{GestureExtractor, ExtractorParams};

const WINDOW_SIZE: usize = 64;
const SENSORS: usize = 5;
const CHANNELS: usize = 7;
const CONFIDENCE_THRESHOLD: f32 = 0.70; // Umbral de confianza mínima para SVM (igual que C++)

/// Predice un gesto desde una ventana usando el pipeline Python
fn predict_window(
    py: Python<'_>,
    clf: &PyAny,
    window: &[[[f32; CHANNELS]; SENSORS]; WINDOW_SIZE],
) -> Result<(String, f32)> {
    // Convertir ventana Rust a numpy array Python
    let np_window = PyArray3::from_array(py, &numpy::ndarray::Array3::from_shape_fn(
        (WINDOW_SIZE, SENSORS, CHANNELS),
        |(t, s, c)| window[t][s][c],
    ));
    
    // Llamar al método predict del clasificador
    let result = clf.call_method1("predict", (np_window,))?;
    
    // Extraer label y confidence
    let label: String = result.get_item(0)?.extract()?;
    let conf: f32 = result.get_item(1)?.extract()?;
    
    Ok((label, conf))
}



fn main() -> Result<()> {
    println!("🎯 Gesture Recognition System\n");
    
    // Obtener MAC address desde argumentos (opcional)
    let args: Vec<String> = env::args().collect();
    
    if args.len() < 2 {
        // Modo carpetas: procesar CSVs en gesto-*
        println!("🔧 Modo: Procesamiento de carpetas gesto-*\n");
        return process_gesture_folders();
    }
    
    let target_mac = &args[1];
    println!("🔧 Modo: BLE Real-Time");
    println!("🎯 Objetivo BLE: {}\n", target_mac);
    
    // Canal para recibir frames BLE
    let (tx, rx) = bounded::<SensorFrame>(100);
    
    // Lanzar hilo BLE en segundo plano
    let target_mac_clone = target_mac.to_string();
    std::thread::spawn(move || {
        if let Err(e) = start_ble_receiver(&target_mac_clone, tx) {
            eprintln!("❌ Error en BLE: {}", e);
        }
    });
    
    // Dar tiempo para la conexión BLE
    std::thread::sleep(Duration::from_secs(3));
    
    // Inicializar clasificador Python
    Python::with_gil(|py| -> Result<()> {
        println!("🔧 Inicializando clasificador Python...");
        
        let sys = py.import("sys")?;
        let sys_path: &pyo3::types::PyList = sys.getattr("path")?.downcast().unwrap();
        sys_path.insert(0, "python")?;
        
        let gi = py.import("gesture_infer")?;
        let cls = gi.getattr("GestureClassifier")?;
        
        let kwargs = PyDict::new(py);
        kwargs.set_item("artifacts_dir", "python")?;
        kwargs.set_item("try_calibrated", true)?;
        
        let clf = cls.call((), Some(kwargs))?;
        println!("✅ Clasificador cargado\n");
        
        // ===== Inicializar GestureExtractor automático =====
        let mut extractor = GestureExtractor::new(ExtractorParams {
            fixed_len: 64,
            high_thr: 10.0,
            low_thr_ratio: 0.45,
            min_len: 6,
            cooldown_frames: 20,
            out_dir: "gestos_auto_rust".to_string(),
            prefix: "gesto_".to_string(),
        });
        
        // Cola thread-safe para gestos detectados (equivalente a pending_gestures en C++)
        use std::sync::{Arc, Mutex};
        let pending_gestures: Arc<Mutex<VecDeque<Vec<SensorFrame>>>> = Arc::new(Mutex::new(VecDeque::new()));
        let pending_gestures_clone = Arc::clone(&pending_gestures);
        
        // Configurar callback del extractor para encolar gestos detectados
        extractor.set_callback(move |frames: &[SensorFrame]| {
            let mut queue = pending_gestures_clone.lock().unwrap();
            queue.push_back(frames.to_vec());
        });
        
        println!("✅ GestureExtractor automático inicializado");
        println!("   - Umbral alto: 10.0 m/s²");
        println!("   - Umbral bajo: 4.5 m/s² (histéresis)");
        println!("   - Ventana: 64 frames centrados en el pico");
        println!("   - Cooldown: 20 frames\n");
        
        // Control de frames
        let mut frames_received = 0u32;
        
        println!("🎬 Iniciando reconocimiento en tiempo real...\n");
        println!("┌──────────────────────────────────────────────────────────┐");
        println!("│  Frames │ Gesto detectado     │ Confianza │ Motor    │");
        println!("├──────────────────────────────────────────────────────────┤");
        
        loop {
            select! {
                recv(rx) -> msg => {
                    match msg {
                        Ok(frame) => {
                            frames_received += 1;
                            
                            // ===== Alimentar el GestureExtractor automático =====
                            extractor.feed(frame);
                            
                            // ===== Procesar gestos detectados automáticamente =====
                            let mut gestures_to_process = Vec::new();
                            {
                                let mut queue = pending_gestures.lock().unwrap();
                                while let Some(gesture_frames) = queue.pop_front() {
                                    gestures_to_process.push(gesture_frames);
                                }
                            }
                            
                            for gesture_frames in gestures_to_process {
                                // Convertir Vec<SensorFrame> a formato ventana [64, 5, 7]
                                let mut window = [[[0.0f32; CHANNELS]; SENSORS]; WINDOW_SIZE];
                                
                                for (t, frame) in gesture_frames.iter().enumerate().take(WINDOW_SIZE) {
                                    for (s, sensor_opt) in frame.iter().enumerate() {
                                        if let Some(sensor_data) = sensor_opt {
                                            window[t][s] = *sensor_data;
                                        }
                                    }
                                }
                                
                                // Clasificar el gesto detectado automáticamente con SVM
                                match predict_window(py, clf, &window) {
                                    Ok((label, conf)) => {
                                        // Imprimir todos los gestos detectados
                                        if label != "desconocido" && conf >= CONFIDENCE_THRESHOLD {
                                            println!("│ {:>7} │ 🎯 {:<18} │ {:>8.1}% │ [SVM]    │",
                                                frames_received,
                                                label,
                                                conf * 100.0
                                            );
                                        }
                                    }
                                    Err(e) => {
                                        eprintln!("❌ Error clasificando gesto automático: {}", e);
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            eprintln!("❌ Error recibiendo frame: {}", e);
                            break;
                        }
                    }
                }
            }
        }
        
        println!("└──────────────────────────────────────────────────────────┘");
        println!("\n✅ Sesión finalizada");
        println!("   Frames totales: {}", frames_received);
        
        Ok(())
    })
}

/// Procesa carpetas gesto-* y clasifica CSVs
fn process_gesture_folders() -> Result<()> {
    use std::fs;
    use std::path::PathBuf;
    
    // Buscar carpetas que empiezan con "gesto-" en la carpeta "gestos/"
    let current_dir = env::current_dir()?;
    let gestos_dir = current_dir.join("gestos");
    
    if !gestos_dir.exists() {
        eprintln!("❌ No se encontró la carpeta 'gestos/' en el directorio actual");
        eprintln!("   Directorio actual: {}", current_dir.display());
        return Ok(());
    }
    
    let mut gesture_folders: Vec<PathBuf> = fs::read_dir(&gestos_dir)?
        .filter_map(|entry| entry.ok())
        .filter(|entry| {
            let path = entry.path();
            path.is_dir() && path.file_name()
                .and_then(|n| n.to_str())
                .map(|s| s.starts_with("gesto-"))
                .unwrap_or(false)
        })
        .map(|entry| entry.path())
        .collect();
    
    gesture_folders.sort();
    
    if gesture_folders.is_empty() {
        eprintln!("❌ No se encontraron carpetas gesto-* en gestos/");
        eprintln!("   Buscando en: {}", gestos_dir.display());
        return Ok(());
    }
    
    println!("📂 Carpetas encontradas: {}\n", gesture_folders.len());
    
    // Inicializar clasificador Python
    Python::with_gil(|py| -> Result<()> {
        let sys = py.import("sys")?;
        let sys_path: &pyo3::types::PyList = sys.getattr("path")?.downcast().unwrap();
        sys_path.insert(0, "python")?;
        
        let gi = py.import("gesture_infer")?;
        let cls = gi.getattr("GestureClassifier")?;
        
        let kwargs = PyDict::new(py);
        kwargs.set_item("artifacts_dir", "python")?;
        kwargs.set_item("try_calibrated", true)?;
        
        let clf = cls.call((), Some(kwargs))?;
        println!("✅ Clasificador cargado\n");
        
        let mut total_correct = 0;
        let mut total_files = 0;
        
        for folder_path in &gesture_folders {
            let folder_name = folder_path.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown");
            
            // El nombre esperado es el nombre completo de la carpeta
            let expected_label = folder_name;
            
            println!("📁 Procesando: {}", folder_name);
            
            // Buscar archivos CSV
            let mut csv_files: Vec<PathBuf> = fs::read_dir(folder_path)?
                .filter_map(|entry| entry.ok())
                .filter(|entry| {
                    let path = entry.path();
                    path.extension()
                        .and_then(|ext| ext.to_str())
                        .map(|ext| ext == "csv")
                        .unwrap_or(false)
                })
                .map(|entry| entry.path())
                .collect();
            
            csv_files.sort();
            
            if csv_files.is_empty() {
                println!("  ⚠️  No se encontraron archivos CSV\n");
                continue;
            }
            
            // Procesar primeros 30 archivos
            csv_files.truncate(30);
            println!("  📄 Archivos: {}", csv_files.len());
            
            let mut correct = 0;
            let mut confidences = Vec::new();
            
            for csv_path in &csv_files {
                let file_name = csv_path.file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("unknown.csv");
                
                match load_window_from_csv(csv_path) {
                    Ok(window) => {
                        match predict_window(py, clf, &window) {
                            Ok((label, conf)) => {
                                confidences.push(conf);
                                
                                let is_correct = label == expected_label;
                                if is_correct {
                                    correct += 1;
                                    total_correct += 1;
                                }
                                
                                let status = if is_correct && conf >= CONFIDENCE_THRESHOLD {
                                    "✅"
                                } else if is_correct {
                                    "⚠️ "
                                } else {
                                    "❌"
                                };
                                
                                println!("  {} {} → {} ({:.1}%)", status, file_name, label, conf * 100.0);
                            }
                            Err(e) => {
                                println!("  ❌ {} → Error: {}", file_name, e);
                            }
                        }
                    }
                    Err(e) => {
                        println!("  ❌ {} → Error cargando: {}", file_name, e);
                    }
                }
                
                total_files += 1;
            }
            
            let accuracy = (correct as f32 / csv_files.len() as f32) * 100.0;
            let avg_conf = if !confidences.is_empty() {
                confidences.iter().sum::<f32>() / confidences.len() as f32
            } else {
                0.0
            };
            
            println!("  📊 Precisión: {}/{} ({:.1}%)", correct, csv_files.len(), accuracy);
            println!("  📈 Confianza promedio: {:.1}%\n", avg_conf * 100.0);
        }
        
        let total_accuracy = (total_correct as f32 / total_files as f32) * 100.0;
        println!("✅ Procesamiento completado");
        println!("   Total: {}/{} ({:.1}%)", total_correct, total_files, total_accuracy);
        
        Ok(())
    })
}

/// Carga una ventana desde un archivo CSV
fn load_window_from_csv(path: &std::path::Path) -> Result<[[[f32; CHANNELS]; SENSORS]; WINDOW_SIZE]> {
    use csv::ReaderBuilder;
    use std::collections::HashMap;
    
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_path(path)?;
    
    // Estructura temporal: [sample][sensor] -> [valores]
    let mut data: HashMap<(usize, usize), [f32; CHANNELS]> = HashMap::new();
    
    for result in reader.records() {
        let record = result?;
        
        if record.len() < 9 {
            continue;
        }
        
        let sample: usize = record[0].parse()?;
        let sensor: usize = record[1].parse()?;
        
        if sample >= WINDOW_SIZE || sensor >= SENSORS {
            continue;
        }
        
        let ax: f32 = record[2].parse()?;
        let ay: f32 = record[3].parse()?;
        let az: f32 = record[4].parse()?;
        let w: f32 = record[5].parse()?;
        let i: f32 = record[6].parse()?;
        let j: f32 = record[7].parse()?;
        let k: f32 = record[8].parse()?;
        
        data.insert((sample, sensor), [ax, ay, az, w, i, j, k]);
    }
    
    // Construir ventana [64, 5, 7]
    let mut window = [[[0.0f32; CHANNELS]; SENSORS]; WINDOW_SIZE];
    
    for t in 0..WINDOW_SIZE {
        for s in 0..SENSORS {
            if let Some(sensor_data) = data.get(&(t, s)) {
                window[t][s] = *sensor_data;
            }
        }
    }
    
    Ok(window)
}