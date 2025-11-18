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

para debug con teclado:
sg input -c './target/debug/onnx-predictor'

*/

mod ble;
mod gesture_buffer;
mod gesture_extractor;
mod hid;

use anyhow::Result;
use crossbeam_channel::{bounded, select, unbounded};
use numpy::PyArray3;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::VecDeque;
use std::env;
use std::time::Duration;

use ble::{SensorFrame, start_ble_receiver};
use gesture_extractor::{GestureExtractor, ExtractorParams};
use hid::{HidOutput, GestureAction};

const WINDOW_SIZE: usize = 64;
const SENSORS: usize = 5;
const CHANNELS: usize = 7;
const CONFIDENCE_THRESHOLD: f32 = 0.70; // Umbral de confianza mínima para SVM (igual que C++)

/// Convierte el quaternion del sensor 0 en movimiento de cursor (dx, dy).
/// Usa pitch (inclinación adelante/atrás) y roll (inclinación izq/der) para controlar el cursor.
fn orientation_to_cursor_movement(sensor_data: &[f32; CHANNELS]) -> (i32, i32) {
    // sensor_data = [ax, ay, az, w, i, j, k]
    let w = sensor_data[3];
    let x = sensor_data[4]; // i
    let y = sensor_data[5]; // j
    let z = sensor_data[6]; // k

    // Calcular pitch (rotación en eje Y, hacia adelante/atrás)
    // pitch = atan2(2(wy + xz), 1 - 2(y² + z²))
    let pitch = (2.0 * (w * y + x * z)).atan2(1.0 - 2.0 * (y * y + z * z));
    
    // Calcular roll (rotación en eje X, hacia los lados)
    // roll = atan2(2(wx + yz), 1 - 2(x² + y²))
    let roll = (2.0 * (w * x + y * z)).atan2(1.0 - 2.0 * (x * x + y * y));

    // Mapear ángulos a movimiento de cursor
    // pitch > 0 → mirar abajo → cursor hacia abajo (dy positivo)
    // roll > 0 → inclinar derecha → cursor hacia derecha (dx positivo)
    
    // Sensibilidad: ~20 grados = 10 píxeles de movimiento
    let sensitivity = 0.5; // píxeles por grado (ajustable)
    
    let dx = (roll.to_degrees() * sensitivity) as i32;
    let dy = (pitch.to_degrees() * sensitivity) as i32;

    // Limitar movimiento máximo por frame
    let max_delta = 15;
    let dx = dx.clamp(-max_delta, max_delta);
    let dy = dy.clamp(-max_delta, max_delta);

    (dx, dy)
}

/// Determina dirección del slide al estilo C++ usando az del sensor 1 en la
/// ventana central. Devuelve Some(true)=derecha, Some(false)=izquierda,
/// None=indefinido.
fn slide_direction_cpp_style(
    window: &[[[f32; CHANNELS]; SENSORS]; WINDOW_SIZE],
) -> Option<(bool, String)> {
    const SENSOR: usize = 1; // igual que C++
    const AZ: usize = 2; // componente az

    // 1) Extraer serie az
    let mut az_series = [0.0f32; WINDOW_SIZE];
    for t in 0..WINDOW_SIZE {
        az_series[t] = window[t][SENSOR][AZ];
    }

    // 2) Buscar max/min ignorando extremos (5..T-5)
    let mut max_val = -1e9f32;
    let mut min_val = 1e9f32;
    let mut idx_max: i32 = -1;
    let mut idx_min: i32 = -1;

    if WINDOW_SIZE <= 10 {
        return None;
    }

    for t in 5..(WINDOW_SIZE - 5) {
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

    // 3) Decidir dirección
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
        // bool = true si DERECHA, false si IZQUIERDA
        Some((direccion == "DERECHA", patron_info))
    }
}

/// Aplica corrección de dirección a labels de slide siguiendo la lógica de
/// main_ble_vsr.cpp. Devuelve un nuevo String (posiblemente igual al input).
fn apply_slide_correction(
    label: &str,
    window: &[[[f32; CHANNELS]; SENSORS]; WINDOW_SIZE],
) -> String {
    // Solo nos interesa para gestos de tipo slide
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
                // corregir texto
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
            // No tenía dirección explícita: la añadimos
            display_label.push_str(&format!("-{} [{}]", direccion_detectada, patron_info));
        }

        display_label
    } else {
        label.to_string()
    }
}

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

/// Predice un gesto y devuelve TODAS las confianzas por clase
fn predict_all_scores(
    py: Python<'_>,
    clf: &PyAny,
    window: &[[[f32; CHANNELS]; SENSORS]; WINDOW_SIZE],
) -> Result<(String, f32, std::collections::HashMap<String, f32>)> {
    use std::collections::HashMap;
    
    // Convertir ventana Rust a numpy array Python
    let np_window = PyArray3::from_array(py, &numpy::ndarray::Array3::from_shape_fn(
        (WINDOW_SIZE, SENSORS, CHANNELS),
        |(t, s, c)| window[t][s][c],
    ));
    
    // Llamar al método predict_all_scores del clasificador
    let result = clf.call_method1("predict_all_scores", (np_window,))?;
    
    // Extraer label, confidence, y all_scores dict
    let label: String = result.get_item(0)?.extract()?;
    let conf: f32 = result.get_item(1)?.extract()?;
    let scores_dict = result.get_item(2)?;
    
    let mut all_scores = HashMap::new();
    
    // Convertir dict Python a HashMap Rust
    if let Ok(dict) = scores_dict.downcast::<PyDict>() {
        for (key, value) in dict.iter() {
            if let (Ok(gesture_name), Ok(score)) = (key.extract::<String>(), value.extract::<f32>()) {
                all_scores.insert(gesture_name, score);
            }
        }
    }
    
    Ok((label, conf, all_scores))
}

/// Conversión string → enum GestureAction
/// Nota: usamos prefijos para ignorar sufijos como "[CORREGIDO por patrón ...]".
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
    println!("🎯 Gesture Recognition System\n");
    
    // Obtener MAC address desde argumentos (opcional)
    let args: Vec<String> = env::args().collect();
    
    if args.len() < 2 {
        // Modo DEBUG: ejecutar con teclado interactivo
        println!("🔧 Modo: DEBUG - Teclado Interactivo\n");
        println!("Presiona teclas para simular gestos:");
        println!("  d → gesto-drop");
        println!("  g → gesto-grab");
        println!("  r → gesto-slide-derecha");
        println!("  l → gesto-slide-izquierda");
        println!("  i → gesto-zoom-in");
        println!("  o → gesto-zoom-out");
        println!("  q → salir\n");
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
        
        // ===== Canal y hilo HID para enviar acciones =====
        let (tx_gesture, rx_gesture) = unbounded::<GestureAction>();
        let (tx_cursor, rx_cursor) = unbounded::<(i32, i32)>(); // Canal para movimiento de cursor (dx, dy)
        
        // Estado compartido para control de cursor
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
                    eprintln!("   Ejecuta con: sudo modprobe uinput");
                    return;
                }
            };
            
            // Loop de procesamiento de comandos HID
            loop {
                select! {
                    recv(rx_gesture) -> action_result => {
                        if let Ok(action) = action_result {
                            // Lógica de activación/desactivación de modo cursor
                            match action {
                                GestureAction::Grab => {
                                    *grab_was_active_clone.lock().unwrap() = true;
                                }
                                GestureAction::Drop => {
                                    let grab_active = *grab_was_active_clone.lock().unwrap();
                                    if !grab_active {
                                        // Drop sin Grab previo -> toggle cursor mode
                                        let mut cursor_mode = cursor_mode_clone.lock().unwrap();
                                        *cursor_mode = !*cursor_mode;
                                        let new_state = *cursor_mode;
                                        drop(cursor_mode); // liberar lock
                                        
                                        if new_state {
                                            println!("🖱️  Modo control de cursor ACTIVADO (Drop sin Grab previo)");
                                        } else {
                                            println!("🖱️  Modo control de cursor DESACTIVADO (segundo Drop)");
                                        }
                                        continue; // No enviar evento Drop al HID
                                    } else {
                                        // Drop con Grab previo -> enviar Drop normal
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
                            // Enviar movimiento de cursor
                            if let Err(e) = hid.move_cursor(dx, dy) {
                                eprintln!("❌ Error moviendo cursor: {}", e);
                            }
                        }
                    }
                }
            }
        });

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
        
        // Cola thread-safe para gestos detectados (5 ventanas por gesto)
        let pending_gestures: Arc<Mutex<VecDeque<[Vec<SensorFrame>; 5]>>> = Arc::new(Mutex::new(VecDeque::new()));
        let pending_gestures_clone = Arc::clone(&pending_gestures);
        
        // Configurar callback del extractor para encolar las 5 ventanas del gesto detectado
        extractor.set_callback(move |windows5: &[Vec<SensorFrame>; 5]| {
            let mut queue = pending_gestures_clone.lock().unwrap();
            queue.push_back(windows5.clone());
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
                            
                            // ===== Control de cursor si está activo =====
                            if *cursor_mode_active.lock().unwrap() {
                                // Usar sensor 0 para controlar cursor
                                if let Some(sensor0_data) = &frame[0] {
                                    let (dx, dy) = orientation_to_cursor_movement(sensor0_data);
                                    // Solo enviar si hay movimiento significativo
                                    if dx.abs() > 1 || dy.abs() > 1 {
                                        let _ = tx_cursor.send((dx, dy));
                                    }
                                }
                            }
                            
                            // ===== Alimentar el GestureExtractor automático =====
                            extractor.feed(frame);
                            
                            // ===== Procesar gestos detectados automáticamente =====
                            let mut gestures_to_process: Vec<[Vec<SensorFrame>; 5]> = Vec::new();
                            {
                                let mut queue = pending_gestures.lock().unwrap();
                                while let Some(gesture_frames) = queue.pop_front() {
                                    gestures_to_process.push(gesture_frames);
                                }
                            }
                            
                            for windows5 in gestures_to_process {
                                // Clasificar las 5 ventanas y votar
                                use std::collections::HashMap;
                                let mut votes: HashMap<String, (u32, f32)> = HashMap::new();
                                let mut centered_window_buf = [[[0.0f32; CHANNELS]; SENSORS]; WINDOW_SIZE];

                                for (idx_w, frames_vec) in windows5.iter().enumerate() {
                                    let mut window = [[[0.0f32; CHANNELS]; SENSORS]; WINDOW_SIZE];
                                    for (t, frame) in frames_vec.iter().enumerate().take(WINDOW_SIZE) {
                                        for (s, sensor_opt) in frame.iter().enumerate() {
                                            if let Some(sensor_data) = sensor_opt {
                                                window[t][s] = *sensor_data;
                                            }
                                        }
                                    }
                                    if idx_w == 2 { centered_window_buf = window; }
                                    match predict_window(py, clf, &window) {
                                        Ok((label, conf)) => {
                                            let e = votes.entry(label).or_insert((0, 0.0));
                                            e.0 += 1; // voto
                                            if conf > e.1 { e.1 = conf; }
                                        }
                                        Err(e) => eprintln!("❌ Error clasificando ventana {}: {}", idx_w, e),
                                    }
                                }

                                if votes.is_empty() { continue; }

                                // Elegir label con más votos; desempatar por mayor confianza
                                let (mut best_label, mut best_cnt, mut best_conf) = (String::new(), 0u32, 0.0f32);
                                for (label, (cnt, conf)) in votes.into_iter() {
                                    if cnt > best_cnt || (cnt == best_cnt && conf > best_conf) {
                                        best_label = label;
                                        best_cnt = cnt;
                                        best_conf = conf;
                                    }
                                }

                                // Corrección de dirección para gestos de slide al estilo C++
                                let final_label = apply_slide_correction(&best_label, &centered_window_buf);

                                if final_label != "desconocido" && best_conf >= CONFIDENCE_THRESHOLD {
                                    println!("│ {:>7} │ 🎯 {:<18} │ {:>8.1}% │ [VOTE]   │",
                                        frames_received,
                                        final_label,
                                        best_conf * 100.0
                                    );
                                    if let Some(action) = map_label_to_action(&final_label) {
                                        let _ = tx_gesture.send(action);
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

/// Modo DEBUG: lee teclas y procesa CSVs correspondientes con HID
fn debug_mode() -> Result<()> {
    use std::fs;
    use std::path::PathBuf;
    use evdev::{Device, InputEventKind, Key};
    
    println!("🔍 Buscando teclado...");
    
    // Buscar dispositivo de teclado
    let mut keyboard_device: Option<Device> = None;
    
    for entry in fs::read_dir("/dev/input")? {
        if let Ok(entry) = entry {
            let path = entry.path();
            if let Some(name) = path.file_name() {
                if name.to_string_lossy().starts_with("event") {
                    if let Ok(device) = Device::open(&path) {
                        if let Some(dev_name) = device.name() {
                            // Buscar dispositivos que sean teclados
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
    
    println!("✅ Captura de teclado global activada (sin necesidad de foco en terminal)\n");
    
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
        println!("✅ Clasificador Python cargado");
        
        // Canal para GestureAction
        let (tx_gesture, rx_gesture) = unbounded::<GestureAction>();
        
        // Hilo HID dedicado
        std::thread::spawn(move || {
            let mut hid = match HidOutput::new() {
                Ok(h) => {
                    println!("✅ HID inicializado (/dev/uinput)");
                    h
                }
                Err(e) => {
                    eprintln!("❌ No se pudo inicializar HID: {}", e);
                    eprintln!("   Ejecuta con: sudo modprobe uinput");
                    eprintln!("   O ejecuta el binario con sudo");
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
        
        // Mapeo tecla → carpeta
        let key_to_folder: std::collections::HashMap<Key, (&str, &str)> = [
            (Key::KEY_D, ("gesto-drop", "d")),
            (Key::KEY_G, ("gesto-grab", "g")),
            (Key::KEY_R, ("gesto-slide-derecha", "r")),
            (Key::KEY_L, ("gesto-slide-izquierda", "l")),
            (Key::KEY_I, ("gesto-zoom-in", "i")),
            (Key::KEY_O, ("gesto-zoom-out", "o")),
        ].iter().cloned().collect();
        
        println!("🎧 Escuchando teclas globales en segundo plano...");
        println!("   Presiona Ctrl+C para salir\n");
        
        // Loop de eventos
        loop {
            for ev in device.fetch_events()? {
                if let InputEventKind::Key(key) = ev.kind() {
                    // Solo procesar cuando se presiona la tecla (value == 1)
                    if ev.value() == 1 {
                        // Verificar si es Q para salir
                        if key == Key::KEY_Q {
                            println!("\n👋 Saliendo...");
                            return Ok(());
                        }
                        
                        if let Some((folder_name, key_char)) = key_to_folder.get(&key) {
                            println!("\n🔑 Tecla presionada: '{}'", key_char);
                            println!("📂 Buscando CSV en: {}/", folder_name);
                            
                            // Buscar un CSV aleatorio en la carpeta
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
                            
                            // Tomar un CSV aleatorio
                            use rand::Rng;
                            let random_idx = rand::thread_rng().gen_range(0..csv_files.len());
                            let csv_path = &csv_files[random_idx];
                            let file_name = csv_path.file_name()
                                .and_then(|n| n.to_str())
                                .unwrap_or("unknown.csv");
                            
                            println!("📄 Archivo: {}", file_name);
                            
                            // Cargar ventana desde CSV
                            match load_window_from_csv(csv_path) {
                                Ok(window) => {
                                    // Predecir gesto
                                    match predict_window(py, clf, &window) {
                                        Ok((label, conf)) => {
                                            // Corrección de dirección para slide al estilo C++
                                            let final_label = apply_slide_correction(&label, &window);
                                            println!("🎯 Predicción: {} ({:.1}%)", final_label, conf * 100.0);
                                            
                                            // Convertir a GestureAction y enviar a HID
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
            
            // Pequeña pausa para no consumir CPU
            std::thread::sleep(Duration::from_millis(10));
        }
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
                        match predict_all_scores(py, clf, &window) {
                            Ok((label, conf, all_scores)) => {
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
                                
                                // Mostrar predicción principal
                                println!("  {} {} → {} ({:.1}%)", status, file_name, label, conf * 100.0);
                                
                                // Mostrar top 3 confianzas (excluyendo la predicción principal)
                                if !all_scores.is_empty() {
                                    let mut scores_vec: Vec<(String, f32)> = all_scores
                                        .into_iter()
                                        .filter(|(name, _)| name != &label)
                                        .collect();
                                    scores_vec.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                                    
                                    let top_n = scores_vec.iter().take(3);
                                    print!("     └─ Otros: ");
                                    for (i, (name, score)) in top_n.enumerate() {
                                        if i > 0 { print!(", "); }
                                        print!("{}: {:.1}%", name, score * 100.0);
                                    }
                                    println!();
                                }
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