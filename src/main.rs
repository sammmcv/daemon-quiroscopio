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

use anyhow::Result;
use crossbeam_channel::{bounded, select};
use numpy::PyArray3;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::VecDeque;
use std::env;
use std::time::{Duration, Instant};

use ble::{SensorFrame, start_ble_receiver, get_stats};
use gesture_buffer::GestureBuffer;

const WINDOW_SIZE: usize = 64;
const SENSORS: usize = 5;
const CHANNELS: usize = 7;
const VOTE_SIZE: usize = 3;  // Reducido para gestos más rápidos
const CONFIDENCE_THRESHOLD: f32 = 0.85;  // Umbral aumentado para reducir falsos positivos
const MOVEMENT_THRESHOLD: f32 = 1.0;  // Umbral más alto para detectar solo gestos reales
const COOLDOWN_MS: u64 = 1000;  // Tiempo de espera entre gestos (ms)

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

/// Realiza votación sobre las últimas N predicciones
fn vote_prediction(history: &VecDeque<(String, f32)>) -> Option<(String, f32)> {
    if history.is_empty() {
        return None;
    }
    
    // Contar votos por clase
    let mut votes: std::collections::HashMap<String, (usize, Vec<f32>)> = std::collections::HashMap::new();
    
    for (label, conf) in history {
        let entry = votes.entry(label.clone()).or_insert((0, Vec::new()));
        entry.0 += 1;
        entry.1.push(*conf);
    }
    
    // Encontrar la clase con más votos
    let mut max_votes = 0;
    let mut winner = None;
    let mut winner_confs = Vec::new();
    
    for (label, (count, confs)) in votes {
        if count > max_votes {
            max_votes = count;
            winner = Some(label);
            winner_confs = confs;
        }
    }
    
    // Calcular confianza promedio del ganador
    if let Some(label) = winner {
        let avg_conf = winner_confs.iter().sum::<f32>() / winner_confs.len() as f32;
        Some((label, avg_conf))
    } else {
        None
    }
}

fn main() -> Result<()> {
    println!("🎯 Gesture Recognition System - BLE Real-Time\n");
    
    // Obtener MAC address desde argumentos
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Uso: {} <MAC_ADDRESS>", args[0]);
        eprintln!("Ejemplo: {} E8:9F:6D:2B:8D:9A", args[0]);
        std::process::exit(1);
    }
    
    let target_mac = &args[1];
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
        
        // Buffer circular y control de predicciones
        let mut buffer = GestureBuffer::new();
        let mut prediction_history: VecDeque<(String, f32)> = VecDeque::with_capacity(VOTE_SIZE);
        let mut last_prediction_time = Instant::now();
        let mut last_gesture_time = Instant::now();
        let mut frames_received = 0u32;
        let mut predictions_made = 0u32;
        let mut in_cooldown = false;
        
        println!("🎬 Iniciando reconocimiento en tiempo real...\n");
        println!("┌──────────────────────────────────────────────────────────────────┐");
        println!("│  Frames │ Predicción          │ Conf.  │ Votación     │ Mov.   │");
        println!("├──────────────────────────────────────────────────────────────────┤");
        
        loop {
            select! {
                recv(rx) -> msg => {
                    match msg {
                        Ok(frame) => {
                            frames_received += 1;
                            buffer.push(frame);
                            
                            // Verificar si terminó el cooldown
                            if in_cooldown && last_gesture_time.elapsed() > Duration::from_millis(COOLDOWN_MS) {
                                in_cooldown = false;
                                prediction_history.clear(); // Resetear historial para nuevo gesto
                            }
                            
                            // Solo predecir cuando:
                            // 1. No estemos en cooldown (esperando entre gestos)
                            // 2. Tengamos suficientes datos (64 frames)
                            // 3. Haya pasado al menos 300ms desde la última predicción
                            // 4. Haya movimiento significativo detectado
                            if !in_cooldown && 
                               buffer.is_ready() && 
                               last_prediction_time.elapsed() > Duration::from_millis(300) {
                                let movement = buffer.get_movement_magnitude();
                                
                                // Solo predecir si hay movimiento significativo
                                if movement >= MOVEMENT_THRESHOLD {
                                    if let Some(window) = buffer.get_window() {
                                        match predict_window(py, clf, &window) {
                                            Ok((label, conf)) => {
                                                predictions_made += 1;
                                                
                                                // Agregar a historia de votación
                                                prediction_history.push_back((label.clone(), conf));
                                                if prediction_history.len() > VOTE_SIZE {
                                                    prediction_history.pop_front();
                                                }
                                                
                                                // Realizar votación
                                                if let Some((voted_label, voted_conf)) = vote_prediction(&prediction_history) {
                                                    // Solo mostrar si cumple el umbral de confianza
                                                    if voted_conf >= CONFIDENCE_THRESHOLD {
                                                        let vote_info = format!("{}/{}", 
                                                            prediction_history.iter().filter(|(l, _)| l == &voted_label).count(),
                                                            prediction_history.len()
                                                        );
                                                        
                                                        println!("│ {:>7} │ ✅ {:<15} │ {:>5.1}% │ {:>12} │ [mov:{:.2}]",
                                                            frames_received,
                                                            voted_label,
                                                            voted_conf * 100.0,
                                                            vote_info,
                                                            movement
                                                        );
                                                        
                                                        // Activar cooldown después de detectar un gesto
                                                        in_cooldown = true;
                                                        last_gesture_time = Instant::now();
                                                    }
                                                }
                                                
                                                last_prediction_time = Instant::now();
                                            }
                                            Err(e) => {
                                                eprintln!("❌ Error en predicción: {}", e);
                                            }
                                        }
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
            
            // ... (Print de estadísticas deshabilitado)
            // if frames_received % 500 == 0 && frames_received > 0 {
            //     let stats = get_stats();
            //     println!("├──────────────────────────────────────────────────────────────────┤");
            //     println!("│ 📊 STATS: Frames={} Predicciones={} Pérdida={:.1}%              │",
            //         stats.superframes,
            //         predictions_made,
            //         if stats.superframes > 0 {
            //             (stats.lost_frames as f32 / stats.superframes as f32) * 100.0
            //         } else {
            //             0.0
            //         }
            //     );
            //     println!("├──────────────────────────────────────────────────────────────────┤");
            // }
        }
        
        println!("└──────────────────────────────────────────────────────────────────┘");
        println!("\n✅ Sesión finalizada");
        println!("   Frames totales: {}", frames_received);
        println!("   Predicciones: {}", predictions_made);
        
        Ok(())
    })
}