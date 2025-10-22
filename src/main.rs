/*
Gesture Recognition usando Pipeline Python desde Rust

Para compilar y ejecutar:
cargo build --release
./target/release/onnx-predictor

O simplemente:

cargo run --release
o
./target/release/onnx-predictor

debugg por carpeta: ./target/release/onnx-predictor 2>&1 | grep -E "📁|📊|📈|✅ Procesamiento"

*/

use anyhow::Result;
use csv::ReaderBuilder;
use numpy::PyArray3;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::fs;
use std::path::PathBuf;

const WINDOW_T: usize = 64;
const SENSORS: usize = 5;
const CHANNELS: usize = 7;
const UMBRAL: f32 = 0.90;

/// Lee un CSV y retorna una ventana [64, 5, 7]
fn read_csv_window(csv_path: &str) -> Result<[[[f32; CHANNELS]; SENSORS]; WINDOW_T]> {
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_path(csv_path)?;
    
    let mut window = [[[0.0f32; CHANNELS]; SENSORS]; WINDOW_T];
    
    for result in reader.records() {
        let record = result?;
        let sample: usize = record.get(0).unwrap().parse()?;
        let sensor: usize = record.get(1).unwrap().parse()?;
        
        if sample < WINDOW_T && sensor < SENSORS {
            window[sample][sensor][0] = record.get(2).unwrap().parse()?; // ax
            window[sample][sensor][1] = record.get(3).unwrap().parse()?; // ay
            window[sample][sensor][2] = record.get(4).unwrap().parse()?; // az
            window[sample][sensor][3] = record.get(5).unwrap().parse()?; // w
            window[sample][sensor][4] = record.get(6).unwrap().parse()?; // i
            window[sample][sensor][5] = record.get(7).unwrap().parse()?; // j
            window[sample][sensor][6] = record.get(8).unwrap().parse()?; // k
        }
    }
    
    Ok(window)
}

/// Predice un gesto desde una ventana usando el pipeline Python
fn predict_window(
    py: Python<'_>,
    clf: &PyAny,
    window: &[[[f32; CHANNELS]; SENSORS]; WINDOW_T],
) -> Result<(String, f32)> {
    // Convertir ventana Rust a numpy array Python
    let np_window = PyArray3::from_array(py, &numpy::ndarray::Array3::from_shape_fn(
        (WINDOW_T, SENSORS, CHANNELS),
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
    
    Python::with_gil(|py| -> Result<()> {
        // 1. Inicializar el clasificador Python
        let sys = py.import("sys").map_err(|e| anyhow::anyhow!("Error importing sys: {}", e))?;
        let sys_path: &pyo3::types::PyList = sys.getattr("path")
            .map_err(|e| anyhow::anyhow!("Error getting sys.path: {}", e))?
            .downcast()
            .map_err(|e| anyhow::anyhow!("Error downcasting: {}", e))?;
        sys_path.insert(0, "python")
            .map_err(|e| anyhow::anyhow!("Error inserting path: {}", e))?;
        
        let gi = py.import("gesture_infer")
            .map_err(|e| anyhow::anyhow!("Error importing gesture_infer: {}", e))?;
        let cls = gi.getattr("GestureClassifier")
            .map_err(|e| anyhow::anyhow!("Error getting GestureClassifier: {}", e))?;
        
        let kwargs = PyDict::new(py);
        kwargs.set_item("artifacts_dir", "python")
            .map_err(|e| anyhow::anyhow!("Error setting artifacts_dir: {}", e))?;
        kwargs.set_item("try_calibrated", true)
            .map_err(|e| anyhow::anyhow!("Error setting try_calibrated: {}", e))?;
        
        let clf = cls.call((), Some(kwargs))
            .map_err(|e| anyhow::anyhow!("Error calling GestureClassifier: {}", e))?;
        
        println!("✅ Clasificador cargado desde python/\n");
        
        // 2. Buscar todas las carpetas gesto-*
        let gesto_dirs: Vec<PathBuf> = fs::read_dir(".")?
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
        
        if gesto_dirs.is_empty() {
            println!("⚠️  No se encontraron carpetas gesto-*");
            return Ok(());
        }
        
        println!("� Carpetas encontradas: {}\n", gesto_dirs.len());
        
        // 3. Procesar cada carpeta
        for gesto_dir in gesto_dirs {
            let dir_name = gesto_dir.file_name().unwrap().to_str().unwrap();
            println!("📁 Procesando: {}", dir_name);
            
            // Leer primeros 30 CSVs
            let mut csv_files: Vec<PathBuf> = fs::read_dir(&gesto_dir)?
                .filter_map(|entry| entry.ok())
                .filter(|entry| {
                    entry.path().extension()
                        .and_then(|ext| ext.to_str())
                        .map(|ext| ext == "csv")
                        .unwrap_or(false)
                })
                .map(|entry| entry.path())
                .collect();
            
            csv_files.sort();
            csv_files.truncate(30);
            
            if csv_files.is_empty() {
                println!("  ⚠️  Sin archivos CSV\n");
                continue;
            }
            
            println!("  📄 Archivos: {}", csv_files.len());
            
            // Procesar cada CSV
            let mut correct = 0;
            let mut total = 0;
            let mut confidences = Vec::new();
            
            for csv_path in &csv_files {
                let filename = csv_path.file_name().unwrap().to_str().unwrap();
                
                match read_csv_window(csv_path.to_str().unwrap()) {
                    Ok(window) => {
                        match predict_window(py, clf, &window) {
                            Ok((label, conf)) => {
                                total += 1;
                                confidences.push(conf);
                                
                                // Verificar si la predicción es correcta
                                if label == dir_name {
                                    correct += 1;
                                    if conf >= UMBRAL {
                                        println!("  ✅ {} → {} ({:.1}%)", filename, label, conf * 100.0);
                                    } else {
                                        println!("  ⚠️  {} → {} ({:.1}%) [bajo umbral]", filename, label, conf * 100.0);
                                    }
                                } else {
                                    println!("  ❌ {} → {} (esperado: {}, {:.1}%)", filename, label, dir_name, conf * 100.0);
                                }
                            }
                            Err(e) => println!("  ❌ Error prediciendo {}: {}", filename, e),
                        }
                    }
                    Err(e) => println!("  ❌ Error leyendo {}: {}", filename, e),
                }
            }
            
            // Resumen
            let accuracy = if total > 0 { (correct as f32 / total as f32) * 100.0 } else { 0.0 };
            let avg_conf = if !confidences.is_empty() {
                confidences.iter().sum::<f32>() / confidences.len() as f32
            } else {
                0.0
            };
            
            println!("  📊 Precisión: {}/{} ({:.1}%)", correct, total, accuracy);
            println!("  📈 Confianza promedio: {:.1}%\n", avg_conf * 100.0);
        }
        
        println!("✅ Procesamiento completado");
        Ok(())
    })
}
