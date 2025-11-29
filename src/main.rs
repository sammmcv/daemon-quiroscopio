/*
Gesture Recognition en Tiempo Real con BLE - Rust Puro + ONNX

Sistema de reconocimiento de gestos que:
1. Recibe datos IMU desde dispositivos BLE (5 sensores)
2. Extrae gestos automáticamente usando GestureExtractor
3. Realiza predicciones usando ONNX Runtime 
4. Replica el post-procesado del proyecto C++ 

Antes de todo, asegurarse de tener onnxruntime instalado.
wget https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-linux-x64-1.22.0.tgz
tar -xzf onnxruntime-linux-x64-1.22.0.tgz

Para compilar y ejecutar:
set -x LD_LIBRARY_PATH (pwd)/onnxruntime-linux-x64-1.22.0/lib $LD_LIBRARY_PATH
     ./target/release/quiroscopio 28:CD:C1:08:37:69

Para debug con teclado:
sg input -c './target/debug/quiroscopio'
*/

use anyhow::Result;
use crossbeam_channel::{bounded, select, unbounded};
use std::collections::VecDeque;
use std::env;
use std::time::{Duration, Instant};

use quiroscopio::ble::{start_ble_receiver, SensorFrame};
use quiroscopio::csv_loader::load_window_from_csv;
use quiroscopio::gesture_classifier::GestureClassifier;
use quiroscopio::gesture_extractor::{ExtractorParams, GestureExtractor};
use quiroscopio::hid::{GestureAction, HidOutput};
use quiroscopio::mouse_filter::{GyroMouseConfig, GyroMouseFilter, Quaternion};
use quiroscopio::types::{SampleFrame, VotingWindows, SAMPLING_RATE};

const CONFIDENCE_THRESHOLD: f32 = 0.70; // Umbral de confianza mínima (igual que C++)
const CURSOR_MOTION_WARMUP_SECS: f32 = 0.5; // Tiempo de movimiento continuo antes de mover cursor

struct PostProcessResult {
    canonical_label: String,
    display_label: String,
}

struct CorrectionOutcome {
    canonical_label: String,
    display_label: String,
}

fn apply_cpp_post_processing(label: &str, window: &[SampleFrame]) -> PostProcessResult {
    let mut canonical_label = label.to_string();
    let mut display_label = label.to_string();
    if window.len() < 2 {
        return PostProcessResult {
            canonical_label,
            display_label,
        };
    }

    if let Some(slide_fix) = slide_correction(label, window) {
        canonical_label = slide_fix.canonical_label;
        display_label = slide_fix.display_label;
    }

    if let Some(zoom_fix) = zoom_grab_drop_correction(label, window) {
        canonical_label = zoom_fix.canonical_label;
        display_label = zoom_fix.display_label;
    }

    PostProcessResult {
        canonical_label,
        display_label,
    }
}

fn slide_correction(label: &str, window: &[SampleFrame]) -> Option<CorrectionOutcome> {
    let label_lower = label.to_lowercase();
    if !label_lower.contains("slide") {
        return None;
    }

    let last_idx = window.len() - 1;
    let delta_qx = window[last_idx].qx[0] - window[0].qx[0];

    let s1_yaw_initial = quaternion_to_yaw_deg(
        window[0].qw[1],
        window[0].qx[1],
        window[0].qy[1],
        window[0].qz[1],
    );
    let s1_yaw_final = quaternion_to_yaw_deg(
        window[last_idx].qw[1],
        window[last_idx].qx[1],
        window[last_idx].qy[1],
        window[last_idx].qz[1],
    );
    let s1_yaw_delta = s1_yaw_final - s1_yaw_initial;

    let s0_roll_initial = quaternion_to_roll_deg(
        window[0].qw[0],
        window[0].qx[0],
        window[0].qy[0],
        window[0].qz[0],
    );
    let s0_roll_final = quaternion_to_roll_deg(
        window[last_idx].qw[0],
        window[last_idx].qx[0],
        window[last_idx].qy[0],
        window[last_idx].qz[0],
    );
    let s0_roll_delta = s0_roll_final - s0_roll_initial;

    let (is_right, patron_info) = if s1_yaw_delta >= 10.38 {
        (true, format!("s1_yaw={}°", short_float(s1_yaw_delta, 5)))
    } else if s1_yaw_delta <= -10.38 {
        (false, format!("s1_yaw={}°", short_float(s1_yaw_delta, 5)))
    } else if s0_roll_delta >= 0.62 {
        (false, format!("s0_roll={}°", short_float(s0_roll_delta, 5)))
    } else if s0_roll_delta <= -0.62 {
        (true, format!("s0_roll={}°", short_float(s0_roll_delta, 5)))
    } else if delta_qx >= -0.2228 {
        (false, format!("Δqx={}", short_float(delta_qx, 6)))
    } else {
        (true, format!("Δqx={}", short_float(delta_qx, 6)))
    };

    let canonical_label = if is_right {
        "gesto-slide-derecha".to_string()
    } else {
        "gesto-slide-izquierda".to_string()
    };

    let direction_str = if is_right { "DERECHA" } else { "IZQUIERDA" };
    let has_direction = label_lower.contains("derecha") || label_lower.contains("izquierda");

    let display_label = if has_direction {
        let label_is_right = label_lower.contains("derecha");
        if label_is_right == is_right {
            format!("{} [✓ {}]", label, patron_info)
        } else {
            format!("{} [CORREGIDO: {}]", canonical_label, patron_info)
        }
    } else {
        format!("{}-{} [{}]", label, direction_str, patron_info)
    };

    Some(CorrectionOutcome {
        canonical_label,
        display_label,
    })
}

fn zoom_grab_drop_correction(label: &str, window: &[SampleFrame]) -> Option<CorrectionOutcome> {
    let should_run = label.contains("zoom-in")
        || label.contains("zoom_in")
        || label.contains("Zoom")
        || label.contains("grab")
        || label.contains("Grab")
        || label.contains("drop")
        || label.contains("Drop");
    if !should_run {
        return None;
    }

    let mean_rot_s3 = mean_rotation_for_sensor(window, 3);
    let (result_class, result_info) = if mean_rot_s3 < 0.0348 {
        (
            "zoom-in".to_string(),
            format!(
                "rot_s3={} [ZOOM:meñique estable]",
                short_float(mean_rot_s3, 6)
            ),
        )
    } else {
        let last_idx = window.len() - 1;
        let yaw_initial = quaternion_to_yaw_deg(
            window[0].qw[3],
            window[0].qx[3],
            window[0].qy[3],
            window[0].qz[3],
        );
        let yaw_final = quaternion_to_yaw_deg(
            window[last_idx].qw[3],
            window[last_idx].qx[3],
            window[last_idx].qy[3],
            window[last_idx].qz[3],
        );
        let yaw_delta = yaw_final - yaw_initial;

        let roll_initial = quaternion_to_roll_deg(
            window[0].qw[3],
            window[0].qx[3],
            window[0].qy[3],
            window[0].qz[3],
        );
        let roll_final = quaternion_to_roll_deg(
            window[last_idx].qw[3],
            window[last_idx].qx[3],
            window[last_idx].qy[3],
            window[last_idx].qz[3],
        );
        let roll_delta = roll_final - roll_initial;

        if yaw_delta >= 4.57 {
            (
                "grab".to_string(),
                format!("s3_yaw={}°", short_float(yaw_delta, 5)),
            )
        } else if yaw_delta <= -4.57 {
            (
                "drop".to_string(),
                format!("s3_yaw={}°", short_float(yaw_delta, 5)),
            )
        } else if roll_delta >= 12.37 {
            (
                "drop".to_string(),
                format!("s3_roll={}°", short_float(roll_delta, 5)),
            )
        } else if roll_delta <= -12.37 {
            (
                "grab".to_string(),
                format!("s3_roll={}°", short_float(roll_delta, 5)),
            )
        } else {
            (
                String::new(),
                format!(
                    "s3_yaw={}°,s3_roll={}° [ambiguo]",
                    short_float(yaw_delta, 5),
                    short_float(roll_delta, 5)
                ),
            )
        }
    };

    if result_class.is_empty() {
        return None;
    }

    let canonical_target = format!("gesto-{}", result_class);
    let label_lower = label.to_lowercase();
    let matches_prediction = label_lower.contains(&result_class);

    let (canonical_label, display_label) = if matches_prediction {
        (label.to_string(), format!("{} [✓ {}]", label, result_info))
    } else {
        (
            canonical_target.clone(),
            format!("{} [CORREGIDO: {}]", canonical_target, result_info),
        )
    };

    Some(CorrectionOutcome {
        canonical_label,
        display_label,
    })
}

fn quaternion_to_yaw_deg(qw: f32, qx: f32, qy: f32, qz: f32) -> f32 {
    let (qw, qx, qy, qz) = normalize_quaternion(qw, qx, qy, qz);
    let siny = 2.0 * (qw * qz + qx * qy);
    let cosy = 1.0 - 2.0 * (qy * qy + qz * qz);
    siny.atan2(cosy).to_degrees()
}

fn quaternion_to_roll_deg(qw: f32, qx: f32, qy: f32, qz: f32) -> f32 {
    let (qw, qx, qy, qz) = normalize_quaternion(qw, qx, qy, qz);
    let sinr = 2.0 * (qw * qx + qy * qz);
    let cosr = 1.0 - 2.0 * (qx * qx + qy * qy);
    sinr.atan2(cosr).to_degrees()
}

fn normalize_quaternion(qw: f32, qx: f32, qy: f32, qz: f32) -> (f32, f32, f32, f32) {
    let norm = (qw * qw + qx * qx + qy * qy + qz * qz).sqrt().max(1e-9);
    (qw / norm, qx / norm, qy / norm, qz / norm)
}

fn mean_rotation_for_sensor(window: &[SampleFrame], sensor_idx: usize) -> f32 {
    if window.len() <= 1 {
        return 0.0;
    }

    let mut total_rot = 0.0;
    let mut count = 0;
    for t in 1..window.len() {
        let (qw0, qx0, qy0, qz0) = (
            window[t - 1].qw[sensor_idx],
            window[t - 1].qx[sensor_idx],
            window[t - 1].qy[sensor_idx],
            window[t - 1].qz[sensor_idx],
        );
        let (qw1, qx1, qy1, qz1) = (
            window[t].qw[sensor_idx],
            window[t].qx[sensor_idx],
            window[t].qy[sensor_idx],
            window[t].qz[sensor_idx],
        );

        let (qw0, qx0, qy0, qz0) = normalize_quaternion(qw0, qx0, qy0, qz0);
        let (qw1, qx1, qy1, qz1) = normalize_quaternion(qw1, qx1, qy1, qz1);

        let mut dot = qw0 * qw1 + qx0 * qx1 + qy0 * qy1 + qz0 * qz1;
        dot = dot.abs().clamp(-1.0, 1.0);
        let angle = 2.0 * dot.acos();
        total_rot += angle;
        count += 1;
    }

    if count > 0 {
        total_rot / count as f32
    } else {
        0.0
    }
}

fn short_float(value: f32, max_chars: usize) -> String {
    let mut s = format!("{}", value);
    if s.len() > max_chars {
        s.truncate(max_chars);
    }
    s
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
    let mut classifier =
        GestureClassifier::new("best_pipeline__time+fft__svm_rbf.onnx", "classes.json")?;
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
                        let cursor_active = *cursor_mode_clone.lock().unwrap();

                        let should_drop_action = cursor_active && !matches!(
                            action,
                            GestureAction::Grab | GestureAction::Drop | GestureAction::ZoomIn
                        );

                        if should_drop_action {
                            continue;
                        }

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

                        if let Err(e) = hid.send_with_mode(action, cursor_active) {
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

    let pending_gestures: Arc<Mutex<VecDeque<VotingWindows>>> =
        Arc::new(Mutex::new(VecDeque::new()));
    let pending_gestures_clone = Arc::clone(&pending_gestures);

    extractor.set_callback(move |windows5: &[Vec<SampleFrame>; 5]| {
        let mut queue = pending_gestures_clone.lock().unwrap();
        queue.push_back(windows5.clone());
    });

    println!("✅ GestureExtractor automático inicializado\n");
    println!("🎬 Iniciando reconocimiento en tiempo real...\n");

    let mut _frames_received = 0u32;
    let mut gyro_filter = GyroMouseFilter::new(GyroMouseConfig {
        gain_x: 18.0,
        gain_y: 12.0,
        max_speed: 40.0,
        alpha: 0.35,  // Más suavizado
        axis_sign_y: 1.0,  // Movimiento vertical normal
        horizontal_axis: quiroscopio::mouse_filter::MotionAxis::Rx,  // Z controla movimiento horizontal
        vertical_axis: quiroscopio::mouse_filter::MotionAxis::Rz,  // X controla movimiento vertical
        ..GyroMouseConfig::default()
    });
    let mut cursor_mode_prev = false;
    let mut last_quaternion: Option<Quaternion> = None;
    let mut last_timestamp: Option<Instant> = None;
    let mut cursor_motion_accum = 0.0f32;

    loop {
        select! {
            recv(rx) -> msg => {
                match msg {
                    Ok(frame) => {
                        _frames_received += 1;

                        let sample_frame = SampleFrame::from_sensor_frame(&frame);

                        let cursor_active = *cursor_mode_active.lock().unwrap();
                        let hand_quat = Quaternion::from_sample(&sample_frame, 0);
                        let now = Instant::now();

                        let dt = last_timestamp
                            .map(|prev| {
                                let delta = now.duration_since(prev).as_secs_f32();
                                delta.clamp(1.0 / 500.0, 0.2)
                            })
                            .unwrap_or(1.0 / SAMPLING_RATE);
                        last_timestamp = Some(now);

                        let prev_quat = last_quaternion;
                        last_quaternion = Some(hand_quat);

                        if cursor_active && !cursor_mode_prev {
                            gyro_filter.reset();
                            cursor_motion_accum = 0.0;
                        }

                        if cursor_active {
                            if let Some(prev) = prev_quat {
                                let (wx, wy, wz) = prev.angular_velocity_to(hand_quat, dt);
                                let (dx, dy) = gyro_filter.update(wx, wy, wz, dt);
                                if dx != 0 || dy != 0 {
                                    cursor_motion_accum = (cursor_motion_accum + dt)
                                        .min(CURSOR_MOTION_WARMUP_SECS);
                                    if cursor_motion_accum >= CURSOR_MOTION_WARMUP_SECS {
                                        let _ = tx_cursor.send((dx, dy));
                                    }
                                } else {
                                    cursor_motion_accum = 0.0;
                                }
                            } else {
                                gyro_filter.reset();
                                cursor_motion_accum = 0.0;
                            }
                        } else {
                            cursor_motion_accum = 0.0;
                        }

                        cursor_mode_prev = cursor_active;

                        // Alimentar el GestureExtractor
                        extractor.feed(sample_frame);

                        // Procesar gestos detectados
                        let mut gestures_to_process: Vec<VotingWindows> = Vec::new();
                        {
                            let mut queue = pending_gestures.lock().unwrap();
                            while let Some(gesture_frames) = queue.pop_front() {
                                gestures_to_process.push(gesture_frames);
                            }
                        }

                        for windows5 in gestures_to_process {
                            let center_window = &windows5[2];

                            match classifier.predict_single(center_window) {
                                Ok((label, conf)) => {
                                    let post = apply_cpp_post_processing(&label, center_window);
                                    let PostProcessResult {
                                        canonical_label,
                                        display_label,
                                    } = post;

                                    if canonical_label != "desconocido" && conf >= CONFIDENCE_THRESHOLD {
                                        println!(
                                            "[GESTO AUTO][svm] {} (conf: {:.2}%)",
                                            display_label,
                                            conf * 100.0
                                        );

                                        if let Some(action) = map_label_to_action(&canonical_label) {
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
    use evdev::{Device, InputEventKind, Key};
    use std::fs;
    use std::path::PathBuf;

    println!("🔍 Buscando teclado...");

    let mut keyboard_device: Option<Device> = None;

    for entry in fs::read_dir("/dev/input")? {
        if let Ok(entry) = entry {
            let path = entry.path();
            if let Some(name) = path.file_name() {
                if name.to_string_lossy().starts_with("event") {
                    if let Ok(device) = Device::open(&path) {
                        if let Some(dev_name) = device.name() {
                            let dev_name_lc = dev_name.to_lowercase();
                            if dev_name_lc.contains("keyboard")
                                || dev_name_lc.contains("at translated")
                            {
                                println!(
                                    "✅ Teclado encontrado: {} ({})",
                                    dev_name,
                                    path.display()
                                );
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

    let mut classifier =
        GestureClassifier::new("best_pipeline__time+fft__svm_rbf.onnx", "classes.json")?;
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
    ]
    .iter()
    .cloned()
    .collect();

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
                            .map(|entry| entry.path())
                            .filter(|path| {
                                path.extension()
                                    .and_then(|ext| ext.to_str())
                                    .map(|ext| ext.eq_ignore_ascii_case("csv"))
                                    .unwrap_or(false)
                            })
                            .collect();

                        if csv_files.is_empty() {
                            eprintln!("❌ No hay archivos CSV en {}", folder_name);
                            continue;
                        }

                        use rand::Rng;
                        let random_idx = rand::thread_rng().gen_range(0..csv_files.len());
                        let csv_path = &csv_files[random_idx];
                        let file_name = csv_path
                            .file_name()
                            .and_then(|n| n.to_str())
                            .unwrap_or("unknown.csv");

                        println!("📄 Archivo: {}", file_name);

                        match load_window_from_csv(csv_path) {
                            Ok(window) => {
                                match classifier.predict_single(&window) {
                                    Ok((label, conf)) => {
                                        let post = apply_cpp_post_processing(&label, &window);
                                        let PostProcessResult {
                                            canonical_label,
                                            display_label,
                                        } = post;

                                        println!(
                                            "🎯 Predicción: {} ({:.1}%)",
                                            display_label,
                                            conf * 100.0
                                        );

                                        if canonical_label == "desconocido" {
                                            println!("⚠️  Gesto desconocido, no se mapea a HID");
                                            continue;
                                        }

                                        if conf < CONFIDENCE_THRESHOLD {
                                            println!("⚠️  Confianza baja, no se envía HID");
                                            continue;
                                        }

                                        if let Some(action) = map_label_to_action(&canonical_label)
                                        {
                                            let _ = tx_gesture.send(action);
                                        } else {
                                            println!(
                                                "⚠️  Gesto no mapeado a acción HID: {}",
                                                display_label
                                            );
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
