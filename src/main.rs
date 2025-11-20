/*
Gesture Recognition en Tiempo Real con BLE - Rust Puro + ONNX

Sistema de reconocimiento de gestos que:
1. Recibe datos IMU desde dispositivos BLE (5 sensores)
2. Extrae gestos automáticamente usando GestureExtractor
3. Realiza predicciones usando ONNX Runtime (sin Python)
4. Implementa votación para estabilizar predicciones

Para compilar y ejecutar:
export LD_LIBRARY_PATH=onnxruntime-linux-x64-1.22.0/lib:$LD_LIBRARY_PATH
cargo build --release
./target/release/quiroscopio <MAC_ADDRESS>

Ejemplo:
./target/release/quiroscopio 28:CD:C1:08:37:69

Para debug con teclado:
sg input -c './target/debug/quiroscopio'
*/

mod ble;
mod types;
mod feature_extractor;
mod gesture_classifier;
mod gesture_extractor;
mod hid;

use anyhow::Result;
use crossbeam_channel::{bounded, select, unbounded};
use std::collections::VecDeque;
use std::env;
use std::time::Duration;

use ble::{SensorFrame, start_ble_receiver};
use types::SampleFrame;
use gesture_extractor::{GestureExtractor, ExtractorParams};
use gesture_classifier::GestureClassifier;
use hid::{HidOutput, GestureAction};

const CONFIDENCE_THRESHOLD: f32 = 0.70; // Umbral de confianza mínima (igual que C++)

/// Convierte el quaternion del sensor 0 en movimiento de cursor (dx, dy).
fn orientation_to_cursor_movement(frame: &SampleFrame) -> (i32, i32) {
    let w = frame.qw[0];
    let x = frame.qx[0];
    let y = frame.qy[0];
    let z = frame.qz[0];

    // Calcular pitch y roll
    let pitch = (2.0 * (w * y + x * z)).atan2(1.0 - 2.0 * (y * y + z * z));
    let roll = (2.0 * (w * x + y * z)).atan2(1.0 - 2.0 * (x * x + y * y));

    let sensitivity = 0.5;
    let dx = (roll.to_degrees() * sensitivity) as i32;
    let dy = (pitch.to_degrees() * sensitivity) as i32;

    let max_delta = 15;
    let dx = dx.clamp(-max_delta, max_delta);
    let dy = dy.clamp(-max_delta, max_delta);

    (dx, dy)
}

/// Determina dirección del slide al estilo C++
fn slide_direction_cpp_style(window: &[SampleFrame]) -> Option<(bool, String)> {
    const SENSOR: usize = 1;
    
    let az_series: Vec<f32> = window.iter().map(|f| f.az[SENSOR]).collect();
    
    if az_series.len() <= 10 {
        return None;
    }

    let mut max_val = -1e9f32;
    let mut min_val = 1e9f32;
    let mut idx_max: i32 = -1;
    let mut idx_min: i32 = -1;

    for t in 5..(az_series.len() - 5) {
        let v = az_series[t];
        if v > max_val {
            max_val = v;
            idx_max = t as i32;
        }
        if v < min_val {
            min_val = v;
            idx_min = t as i32;
        }
    }

    if idx_max < 0 || idx_min < 0 {
        return None;
    }

    let magnitud_max = max_val.abs();
    let magnitud_min = min_val.abs();

    let mut direccion = "DESCONOCIDA".to_string();
    let mut patron_info = String::new();

    if magnitud_max > 2.0 && magnitud_min > 2.0 {
        if idx_max < idx_min {
            direccion = "IZQUIERDA".to_string();
            patron_info = "campana↑→↓".to_string();
        } else {
            direccion = "DERECHA".to_string();
            patron_info = "campana↓→↑".to_string();
        }
    } else if magnitud_max > magnitud_min * 1.5 {
        direccion = "IZQUIERDA".to_string();
        patron_info = "solo↑".to_string();
    } else if magnitud_min > magnitud_max * 1.5 {
        direccion = "DERECHA".to_string();
        patron_info = "solo↓".to_string();
    }

    if direccion == "DESCONOCIDA" {
        None
    } else {
        Some((direccion == "DERECHA", patron_info))
    }
}

/// Aplica corrección de dirección a labels de slide
fn apply_slide_correction(label: &str, window: &[SampleFrame]) -> String {
    if !label.contains("slide") && !label.contains("Slide") && !label.contains("SLIDE") {
        return label.to_string();
    }

    if let Some((is_right, patron_info)) = slide_direction_cpp_style(window) {
        let mut display_label = label.to_string();
        let tiene_derecha = display_label.contains("derecha");
        let tiene_izquierda = display_label.contains("izquierda");

        let direccion_detectada = if is_right { "DERECHA" } else { "IZQUIERDA" };

        if tiene_derecha || tiene_izquierda {
            let es_derecha_label = tiene_derecha;
            let es_derecha_detectada = is_right;

            if es_derecha_label != es_derecha_detectada {
                if es_derecha_label {
                    if let Some(pos) = display_label.find("derecha") {
                        display_label.replace_range(pos..pos + "derecha".len(), "izquierda");
                    }
                } else {
                    if let Some(pos) = display_label.find("izquierda") {
                        display_label.replace_range(pos..pos + "izquierda".len(), "derecha");
                    }
                }
                display_label.push_str(&format!(" [CORREGIDO por patrón: {}]", patron_info));
            } else {
                display_label.push_str(&format!(" [✓ {}]", patron_info));
            }
        } else {
            display_label.push_str(&format!("-{} [{}]", direccion_detectada, patron_info));
        }

        display_label
    } else {
        label.to_string()
    }
}

fn normalize_quaternion(qw: f32, qx: f32, qy: f32, qz: f32) -> Option<[f32; 4]> {
    let norm = (qw * qw + qx * qx + qy * qy + qz * qz).sqrt();
    if norm <= 1e-6 {
        None
    } else {
        Some([qw / norm, qx / norm, qy / norm, qz / norm])
    }
}

fn mean_rotation_sensor(window: &[SampleFrame], sensor_idx: usize) -> Option<f32> {
    if window.len() < 2 {
        return None;
    }

    let mut total = 0.0f32;
    let mut count = 0;

    for pair in window.windows(2) {
        let prev = &pair[0];
        let curr = &pair[1];

        if let (Some(q0), Some(q1)) = (
            normalize_quaternion(prev.qw[sensor_idx], prev.qx[sensor_idx], prev.qy[sensor_idx], prev.qz[sensor_idx]),
            normalize_quaternion(curr.qw[sensor_idx], curr.qx[sensor_idx], curr.qy[sensor_idx], curr.qz[sensor_idx]),
        ) {
            let mut dot = q0[0] * q1[0] + q0[1] * q1[1] + q0[2] * q1[2] + q0[3] * q1[3];
            dot = dot.abs();
            dot = dot.clamp(-1.0, 1.0);
            total += 2.0 * dot.acos();
            count += 1;
        }
    }

    if count > 0 {
        Some(total / count as f32)
    } else {
        None
    }
}

fn quaternion_yaw_roll(qw: f32, qx: f32, qy: f32, qz: f32) -> Option<(f64, f64)> {
    let mut qw = qw as f64;
    let mut qx = qx as f64;
    let mut qy = qy as f64;
    let mut qz = qz as f64;

    let norm = (qw * qw + qx * qx + qy * qy + qz * qz).sqrt();
    if norm < 1e-9 {
        return None;
    }

    qw /= norm;
    qx /= norm;
    qy /= norm;
    qz /= norm;

    let siny = 2.0 * (qw * qz + qx * qy);
    let cosy = 1.0 - 2.0 * (qy * qy + qz * qz);
    let yaw = siny.atan2(cosy).to_degrees();

    let sinr = 2.0 * (qw * qx + qy * qz);
    let cosr = 1.0 - 2.0 * (qx * qx + qy * qy);
    let roll = sinr.atan2(cosr).to_degrees();

    Some((yaw, roll))
}

fn apply_zoom_grab_drop_correction(label: &str, window: &[SampleFrame]) -> String {
    let label_lower = label.to_lowercase();
    let relevant = label_lower.contains("zoom-in")
        || label_lower.contains("zoom_in")
        || label_lower.contains("zoom")
        || label_lower.contains("grab")
        || label_lower.contains("drop");

    if !relevant {
        return label.to_string();
    }

    const SENSOR_IDX: usize = 3; // Meñique
    const ROT_THRESHOLD: f32 = 0.0348;
    const YAW_THRESHOLD: f32 = 4.57;
    const ROLL_THRESHOLD: f32 = 12.37;

    let mean_rot = match mean_rotation_sensor(window, SENSOR_IDX) {
        Some(val) => val,
        None => return label.to_string(),
    };

    let (target_label, info) = if mean_rot < ROT_THRESHOLD {
        (
            "gesto-zoom-in".to_string(),
            format!("rot_s3={:.4} [ZOOM:meñique estable]", mean_rot),
        )
    } else {
        let first = match window.first() {
            Some(f) => f,
            None => return label.to_string(),
        };
        let last = match window.last() {
            Some(f) => f,
            None => return label.to_string(),
        };

        let (yaw_i, roll_i) = match quaternion_yaw_roll(first.qw[SENSOR_IDX], first.qx[SENSOR_IDX], first.qy[SENSOR_IDX], first.qz[SENSOR_IDX]) {
            Some(vals) => vals,
            None => return label.to_string(),
        };
        let (yaw_f, roll_f) = match quaternion_yaw_roll(last.qw[SENSOR_IDX], last.qx[SENSOR_IDX], last.qy[SENSOR_IDX], last.qz[SENSOR_IDX]) {
            Some(vals) => vals,
            None => return label.to_string(),
        };

        let yaw_delta = (yaw_f - yaw_i) as f32;
        let roll_delta = (roll_f - roll_i) as f32;

        if yaw_delta >= YAW_THRESHOLD {
            (
                "gesto-grab".to_string(),
                format!("s3_yaw={:.2}°", yaw_delta),
            )
        } else if yaw_delta <= -YAW_THRESHOLD {
            (
                "gesto-drop".to_string(),
                format!("s3_yaw={:.2}°", yaw_delta),
            )
        } else if roll_delta >= ROLL_THRESHOLD {
            (
                "gesto-drop".to_string(),
                format!("s3_roll={:.2}°", roll_delta),
            )
        } else if roll_delta <= -ROLL_THRESHOLD {
            (
                "gesto-grab".to_string(),
                format!("s3_roll={:.2}°", roll_delta),
            )
        } else {
            return label.to_string();
        }
    };

    let target_lower = target_label.to_lowercase();
    if label_lower.starts_with(&target_lower) {
        format!("{} [✓ {}]", label, info)
    } else {
        format!("{} [CORREGIDO: {}]", target_label, info)
    }
}

/// Conversión string → enum GestureAction
fn map_label_to_action(label: &str) -> Option<GestureAction> {
    use GestureAction::*;

    if label.starts_with("gesto-slide-derecha") {
        Some(SlideDer)
    } else if label.starts_with("gesto-slide-izquierda") {
        Some(SlideIzq)
    } else if label.starts_with("gesto-zoom-in") {
        Some(ZoomIn)
    } else if label.starts_with("gesto-zoom-out") {
        Some(ZoomOut)
    } else if label.starts_with("gesto-grab") {
        Some(Grab)
    } else if label.starts_with("gesto-drop") {
        Some(Drop)
    } else {
        None
    }
}


fn main() -> Result<()> {
    println!("🎯 Gesture Recognition System - Rust + ONNX\n");
    
    // Obtener MAC address desde argumentos (opcional)
    let args: Vec<String> = env::args().collect();
    
    if args.len() < 2 {
        println!("🔧 Modo: DEBUG - Teclado Interactivo\n");
        return debug_mode();
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
    
    std::thread::sleep(Duration::from_secs(3));
    
    // Inicializar clasificador ONNX
    println!("🔧 Inicializando clasificador ONNX...");
    let mut classifier = GestureClassifier::new(
        "best_pipeline__time+fft__svm_rbf.onnx",
        "classes.json"
    )?;
    println!("✅ Clasificador cargado\n");
    
    // Canal y hilo HID
    let (tx_gesture, rx_gesture) = unbounded::<GestureAction>();
    let (tx_cursor, rx_cursor) = unbounded::<(i32, i32)>();
    
    use std::sync::{Arc, Mutex};
    let cursor_mode_active = Arc::new(Mutex::new(false));
    let cursor_mode_clone = Arc::clone(&cursor_mode_active);
    let grab_was_active = Arc::new(Mutex::new(false));
    let grab_was_active_clone = Arc::clone(&grab_was_active);
    
    std::thread::spawn(move || {
        let mut hid = match HidOutput::new() {
            Ok(h) => {
                println!("✅ HID inicializado (/dev/uinput)");
                h
            }
            Err(e) => {
                eprintln!("❌ No se pudo inicializar HID: {}", e);
                return;
            }
        };
        
        loop {
            select! {
                recv(rx_gesture) -> action_result => {
                    if let Ok(action) = action_result {
                        match action {
                            GestureAction::Grab => {
                                *grab_was_active_clone.lock().unwrap() = true;
                            }
                            GestureAction::Drop => {
                                let grab_active = *grab_was_active_clone.lock().unwrap();
                                if !grab_active {
                                    let mut cursor_mode = cursor_mode_clone.lock().unwrap();
                                    *cursor_mode = !*cursor_mode;
                                    let new_state = *cursor_mode;
                                    drop(cursor_mode);
                                    
                                    if new_state {
                                        println!("🖱️  Modo control de cursor ACTIVADO");
                                    } else {
                                        println!("🖱️  Modo control de cursor DESACTIVADO");
                                    }
                                    continue;
                                } else {
                                    *grab_was_active_clone.lock().unwrap() = false;
                                }
                            }
                            _ => {}
                        }
                        
                        if let Err(e) = hid.send(action) {
                            eprintln!("❌ Error enviando gesto HID {:?}: {}", action, e);
                        }
                    }
                }
                recv(rx_cursor) -> cursor_result => {
                    if let Ok((dx, dy)) = cursor_result {
                        if let Err(e) = hid.move_cursor(dx, dy) {
                            eprintln!("❌ Error moviendo cursor: {}", e);
                        }
                    }
                }
            }
        }
    });

    // Inicializar GestureExtractor automático
    let mut extractor = GestureExtractor::new(ExtractorParams {
        out_dir: "gestos_auto_rust".to_string(),
        prefix: "gesto__".to_string(),
        ..Default::default()
    });
    
    let pending_gestures: Arc<Mutex<VecDeque<[Vec<SampleFrame>; 5]>>> = 
        Arc::new(Mutex::new(VecDeque::new()));
    let pending_gestures_clone = Arc::clone(&pending_gestures);
    
    extractor.set_callback(move |windows5: &[Vec<SampleFrame>; 5]| {
        let mut queue = pending_gestures_clone.lock().unwrap();
        queue.push_back(windows5.clone());
    });
    
    println!("✅ GestureExtractor automático inicializado\n");
    println!("🎬 Iniciando reconocimiento en tiempo real...\n");
    println!("┌──────────────────────────────────────────────────────────┐");
    println!("│  Frames │ Gesto detectado     │ Confianza │ Motor    │");
    println!("├──────────────────────────────────────────────────────────┤");
    
    let mut frames_received = 0u32;
    
    loop {
        select! {
            recv(rx) -> msg => {
                match msg {
                    Ok(frame) => {
                        frames_received += 1;
                        
                        let sample_frame = SampleFrame::from_sensor_frame(&frame);
                        
                        // Control de cursor si está activo
                        if *cursor_mode_active.lock().unwrap() {
                            let (dx, dy) = orientation_to_cursor_movement(&sample_frame);
                            if dx.abs() > 1 || dy.abs() > 1 {
                                let _ = tx_cursor.send((dx, dy));
                            }
                        }
                        
                        // Alimentar el GestureExtractor
                        extractor.feed(sample_frame);
                        
                        // Procesar gestos detectados
                        let mut gestures_to_process: Vec<[Vec<SampleFrame>; 5]> = Vec::new();
                        {
                            let mut queue = pending_gestures.lock().unwrap();
                            while let Some(gesture_frames) = queue.pop_front() {
                                gestures_to_process.push(gesture_frames);
                            }
                        }
                        
                        for windows5 in gestures_to_process {
                            // Clasificar con votación usando el clasificador ONNX
                            match classifier.predict_with_voting(&windows5) {
                                Ok((label, conf)) => {
                                    let slide_corrected = apply_slide_correction(&label, &windows5[2]);
                                    let final_label = apply_zoom_grab_drop_correction(&slide_corrected, &windows5[2]);

                                    if final_label != "desconocido" && conf >= CONFIDENCE_THRESHOLD {
                                        println!("│ {:>7} │ 🎯 {:<18} │ {:>8.1}% │ [VOTE]   │",
                                            frames_received,
                                            final_label,
                                            conf * 100.0
                                        );
                                        
                                        if let Some(action) = map_label_to_action(&final_label) {
                                            let _ = tx_gesture.send(action);
                                        }
                                    }
                                }
                                Err(e) => {
                                    eprintln!("❌ Error clasificando: {}", e);
                                }
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("❌ Error recibiendo frame: {}", e);
                        return Ok(());
                    }
                }
            }
        }
    }
}

/// Modo DEBUG: lee teclas y procesa CSVs correspondientes
fn debug_mode() -> Result<()> {
    use std::fs;
    use std::path::PathBuf;
    use evdev::{Device, InputEventKind, Key};
    
    println!("🔍 Buscando teclado...");
    
    let mut keyboard_device: Option<Device> = None;
    
    for entry in fs::read_dir("/dev/input")? {
        if let Ok(entry) = entry {
            let path = entry.path();
            if let Some(name) = path.file_name() {
                if name.to_string_lossy().starts_with("event") {
                    if let Ok(device) = Device::open(&path) {
                        if let Some(dev_name) = device.name() {
                            if dev_name.to_lowercase().contains("keyboard") 
                                || dev_name.to_lowercase().contains("at translated") {
                                println!("✅ Teclado encontrado: {} ({})", dev_name, path.display());
                                keyboard_device = Some(device);
                                break;
                            }
                        }
                    }
                }
            }
        }
    }
    
    let mut device = keyboard_device.ok_or_else(|| {
        anyhow::anyhow!("No se encontró ningún dispositivo de teclado en /dev/input")
    })?;
    
    println!("✅ Captura de teclado global activada\n");
    
    // Inicializar clasificador ONNX
    let mut classifier = GestureClassifier::new(
        "best_pipeline__time+fft__svm_rbf.onnx",
        "classes.json"
    )?;
    println!("✅ Clasificador ONNX cargado\n");
    
    let (tx_gesture, rx_gesture) = unbounded::<GestureAction>();
    
    std::thread::spawn(move || {
        let mut hid = match HidOutput::new() {
            Ok(h) => {
                println!("✅ HID inicializado (/dev/uinput)");
                h
            }
            Err(e) => {
                eprintln!("❌ No se pudo inicializar HID: {}", e);
                return;
            }
        };
        
        while let Ok(action) = rx_gesture.recv() {
            println!("🎮 Enviando acción HID: {:?}", action);
            if let Err(e) = hid.send(action) {
                eprintln!("❌ Error enviando gesto HID {:?}: {}", action, e);
            }
        }
    });
    
    println!("✅ Sistema listo\n");
    println!("Presiona teclas para simular gestos:");
    println!("  d → gesto-drop");
    println!("  g → gesto-grab");
    println!("  r → gesto-slide-derecha");
    println!("  l → gesto-slide-izquierda");
    println!("  i → gesto-zoom-in");
    println!("  o → gesto-zoom-out");
    println!("  q → salir\n");
    
    let key_to_folder: std::collections::HashMap<Key, (&str, &str)> = [
        (Key::KEY_D, ("gestos/gesto-drop", "d")),
        (Key::KEY_G, ("gestos/gesto-grab", "g")),
        (Key::KEY_R, ("gestos/gesto-slide-derecha", "r")),
        (Key::KEY_L, ("gestos/gesto-slide-izquierda", "l")),
        (Key::KEY_I, ("gestos/gesto-zoom-in", "i")),
        (Key::KEY_O, ("gestos/gesto-zoom-out", "o")),
    ].iter().cloned().collect();
    
    println!("🎧 Escuchando teclas globales...\n");
    
    loop {
        for ev in device.fetch_events()? {
            if let InputEventKind::Key(key) = ev.kind() {
                if ev.value() == 1 {
                    if key == Key::KEY_Q {
                        println!("\n👋 Saliendo...");
                        return Ok(());
                    }
                    
                    if let Some((folder_name, key_char)) = key_to_folder.get(&key) {
                        println!("\n🔑 Tecla presionada: '{}'", key_char);
                        println!("📂 Buscando CSV en: {}/", folder_name);
                        
                        let folder_path = PathBuf::from(folder_name);
                        
                        if !folder_path.exists() {
                            eprintln!("❌ Carpeta no existe: {}", folder_name);
                            continue;
                        }
                        
                        let csv_files: Vec<PathBuf> = fs::read_dir(&folder_path)?
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
                        
                        if csv_files.is_empty() {
                            eprintln!("❌ No hay archivos CSV en {}", folder_name);
                            continue;
                        }
                        
                        use rand::Rng;
                        let random_idx = rand::thread_rng().gen_range(0..csv_files.len());
                        let csv_path = &csv_files[random_idx];
                        let file_name = csv_path.file_name()
                            .and_then(|n| n.to_str())
                            .unwrap_or("unknown.csv");
                        
                        println!("📄 Archivo: {}", file_name);
                        
                        match load_window_from_csv(csv_path) {
                            Ok(window) => {
                                match classifier.predict_single(&window) {
                                    Ok((label, conf)) => {
                                        let final_label = apply_slide_correction(&label, &window);
                                        println!("🎯 Predicción: {} ({:.1}%)", final_label, conf * 100.0);
                                        
                                        if let Some(action) = map_label_to_action(&final_label) {
                                            if conf >= CONFIDENCE_THRESHOLD {
                                                let _ = tx_gesture.send(action);
                                            } else {
                                                println!("⚠️  Confianza baja, no se envía HID");
                                            }
                                        } else {
                                            println!("⚠️  Gesto no mapeado a acción HID: {}", final_label);
                                        }
                                    }
                                    Err(e) => {
                                        eprintln!("❌ Error en predicción: {}", e);
                                    }
                                }
                            }
                            Err(e) => {
                                eprintln!("❌ Error cargando CSV: {}", e);
                            }
                        }
                    }
                }
            }
        }
        
        std::thread::sleep(Duration::from_millis(10));
    }
}

/// Carga una ventana desde un archivo CSV
fn load_window_from_csv(path: &std::path::Path) -> Result<Vec<SampleFrame>> {
    use csv::ReaderBuilder;
    use std::collections::HashMap;
    
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_path(path)?;
    
    // Estructura temporal: [sample][sensor] -> [valores]
    let mut data: HashMap<(usize, usize), [f32; 7]> = HashMap::new();
    let mut max_sample = 0;
    
    for result in reader.records() {
        let record = result?;
        
        if record.len() < 9 {
            continue;
        }
        
        let sample: usize = record[0].parse()?;
        let sensor: usize = record[1].parse()?;
        
        if sensor >= 5 {
            continue;
        }
        
        max_sample = max_sample.max(sample);
        
        let ax: f32 = record[2].parse()?;
        let ay: f32 = record[3].parse()?;
        let az: f32 = record[4].parse()?;
        let w: f32 = record[5].parse()?;
        let i: f32 = record[6].parse()?;
        let j: f32 = record[7].parse()?;
        let k: f32 = record[8].parse()?;
        
        data.insert((sample, sensor), [ax, ay, az, w, i, j, k]);
    }
    
    // Construir ventana de SampleFrames
    let window_size = max_sample + 1;
    let mut window = Vec::with_capacity(window_size);
    
    for t in 0..window_size {
        let mut frame = SampleFrame::default();
        
        for s in 0..5 {
            if let Some(sensor_data) = data.get(&(t, s)) {
                frame.ax[s] = sensor_data[0];
                frame.ay[s] = sensor_data[1];
                frame.az[s] = sensor_data[2];
                frame.qw[s] = sensor_data[3];
                frame.qx[s] = sensor_data[4];
                frame.qy[s] = sensor_data[5];
                frame.qz[s] = sensor_data[6];
            }
        }
        
        window.push(frame);
    }
    
    Ok(window)
}

