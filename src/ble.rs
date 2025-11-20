use dbus::blocking::Connection;
use dbus::arg::{Variant, RefArg};
use std::time::Duration;
use std::sync::atomic::{AtomicU16, AtomicU32, Ordering};
use crossbeam_channel::Sender;

/// Datos de un sensor IMU: [ax, ay, az, qw, qx, qy, qz]
pub type SensorData = [f32; 7];

/// Frame completo con datos de los 5 sensores
pub type SensorFrame = [Option<SensorData>; 5];

const EXPECTED_SENSORS: usize = 5;

/// Estadísticas de recepción BLE
#[derive(Debug, Clone, Default)]
pub struct BleStats {
    pub superframes: u32,
    pub lost_frames: u32,
}

// Contadores globales para estadísticas
static SUPERFRAMES: AtomicU32 = AtomicU32::new(0);
static LOST_FRAMES: AtomicU32 = AtomicU32::new(0);
static LAST_CNT10: AtomicU16 = AtomicU16::new(0);
static HAVE_LAST_CNT: AtomicU32 = AtomicU32::new(0);

/// Conecta al dispositivo BLE y comienza a recibir datos
/// Envía cada frame decodificado por el canal proporcionado
pub fn start_ble_receiver(target_mac: &str, tx: Sender<SensorFrame>) -> Result<(), Box<dyn std::error::Error>> {
    let conn = Connection::new_system()?;
    println!("🔌 Conectado a D-Bus del sistema");
    
    // Encender el adaptador Bluetooth
    let adapter_proxy = conn.with_proxy("org.bluez", "/org/bluez/hci0", Duration::from_secs(5));
    
    let _: () = adapter_proxy.method_call(
        "org.freedesktop.DBus.Properties", 
        "Set", 
        ("org.bluez.Adapter1", "Powered", Variant(true))
    )?;
    println!("✅ Adaptador Bluetooth encendido");
    
    // Detener cualquier descubrimiento previo
    let stop_result: Result<(), _> = adapter_proxy.method_call("org.bluez.Adapter1", "StopDiscovery", ());
    if let Err(e) = stop_result {
        if !format!("{}", e).contains("No discovery started") {
            println!("⚠️  Error al detener el descubrimiento: {}", e);
        }
    }
    
    // Construir la ruta del dispositivo basada en la MAC
    let device_path_str = format!("/org/bluez/hci0/dev_{}", target_mac.replace(':', "_"));
    println!("🔍 Buscando dispositivo en: {}", device_path_str);
    
    // Intentar conectar al dispositivo específico
    let device_proxy = conn.with_proxy("org.bluez", &device_path_str, Duration::from_secs(10));
    
    match device_proxy.method_call::<(), _, _, _>("org.bluez.Device1", "Connect", ()) {
        Ok(_) => {
            println!("✅ Conectado exitosamente al dispositivo {}", target_mac);
            std::thread::sleep(Duration::from_secs(2));
        }
        Err(e) => {
            println!("❌ No se pudo conectar al dispositivo {}: {}", target_mac, e);
            println!("⏳ Reintentando en 3 segundos...");
            std::thread::sleep(Duration::from_secs(3));
            
            device_proxy.method_call::<(), _, _, _>("org.bluez.Device1", "Connect", ())?;
            println!("✅ Conectado en segundo intento");
            std::thread::sleep(Duration::from_secs(2));
        }
    }
    
    // Configurar notificaciones
    let char_path = format!("{}/service0001/char0002", device_path_str);
    let char_proxy = conn.with_proxy("org.bluez", &char_path, Duration::from_secs(5));
    
    char_proxy.method_call::<(), _, _, _>("org.bluez.GattCharacteristic1", "StartNotify", ())?;
    println!("📡 Notificaciones BLE iniciadas en {}", char_path);

    // Lanzar hilo de estadísticas
    std::thread::spawn(move || {
        loop {
            std::thread::sleep(Duration::from_secs(5));
            let _sf = SUPERFRAMES.load(Ordering::Relaxed);
            let _lost = LOST_FRAMES.load(Ordering::Relaxed);
                // (Print de estadísticas deshabilitado)
                // if _sf > 0 {
                //     let loss_rate = (_lost as f32 / _sf as f32) * 100.0;
                //     println!("[📊 STATS] Frames recibidos={} perdidos={} ({:.2}%)", _sf, _lost, loss_rate);
                // }
        }
    });
    
    // Preparar el handler de notificaciones
    use dbus::blocking::stdintf::org_freedesktop_dbus::PropertiesPropertiesChanged as PC;
    use dbus::message::SignalArgs;
    
    let char_path_clone = char_path.clone();
    let tx_clone = tx.clone();
    
    let mr = PC::match_rule(None, None);
    conn.add_match(mr, move |pc: PC, _, msg| {
        if msg.path().map(|p| p.to_string()) != Some(char_path_clone.clone()) {
            return true;
        }
        
        if let Some(value_var) = pc.changed_properties.get("Value") {
            if let Some(value) = value_var.0.as_iter().and_then(|iter| {
                let v: Vec<u8> = iter.filter_map(|item| item.as_u64().map(|b| b as u8)).collect();
                Some(v)
            }) {
                if let Some(frame) = decode_superframe(&value) {
                    SUPERFRAMES.fetch_add(1, Ordering::Relaxed);
                    let _ = tx_clone.send(frame);
                }
            }
        }
        true
    })?;

    println!("🎯 Recibiendo datos BLE en tiempo real...\n");
    
    loop {
        conn.process(Duration::from_secs(1))?;
    }
}

/// Decodifica una supertrama BLE y retorna un SensorFrame
fn decode_superframe(value: &[u8]) -> Option<SensorFrame> {
    if value.len() < 2 {
        return None;
    }
    
    let b0 = value[0];
    let b1 = value[1];
    let presence = (b0 >> 3) & 0x1F;
    let cnt10 = (((b0 & 0x06) as u16) << 7) | b1 as u16;
    
    // Calcular sensores presentes
    let mut n_sensors = 0u8;
    for i in 0..5 {
        if presence & (1 << i) != 0 {
            n_sensors += 1;
        }
    }
    
    let expected_len = 2 + (14 * n_sensors as usize);
    if value.len() != expected_len || presence > 0x1F {
        return None;
    }
    
    // Detectar frames perdidos
    if HAVE_LAST_CNT.load(Ordering::Relaxed) != 0 {
        let last = LAST_CNT10.load(Ordering::Relaxed);
        let expected = (last + 1) & 0x03FF;
        
        if cnt10 != expected {
            let diff = (cnt10.wrapping_sub(expected)) & 0x03FF;
            if diff > 0 && diff < 20 {
                LOST_FRAMES.fetch_add(diff as u32, Ordering::Relaxed);
            }
        }
    }
    
    LAST_CNT10.store(cnt10, Ordering::Relaxed);
    HAVE_LAST_CNT.store(1, Ordering::Relaxed);
    
    // Decodificar sensores con mapeo de presencia -> índice lógico
    // Mapa según referencia C++:
    // bit 0 -> idx 0 (UART c0)
    // bit 1 -> idx 1 (0x28 c0)
    // bit 2 -> idx 3 (0x28 c1)
    // bit 3 -> idx 2 (0x29 c0)
    // bit 4 -> idx 4 (0x29 c1)
    let map_idx = [0usize, 1usize, 3usize, 2usize, 4usize];

    let mut frame: SensorFrame = [None; 5];
    let mut offset = 2;
    
    for i in 0..5 {
        if presence & (1 << i) == 0 {
            continue;
        }
        
        if offset + 14 > value.len() {
            break;
        }
        
        let ax = i16::from_le_bytes([value[offset], value[offset + 1]]) as f32 / 100.0;
        let ay = i16::from_le_bytes([value[offset + 2], value[offset + 3]]) as f32 / 100.0;
        let az = i16::from_le_bytes([value[offset + 4], value[offset + 5]]) as f32 / 100.0;
        let qx = i16::from_le_bytes([value[offset + 6], value[offset + 7]]) as f32 / 16384.0;
        let qy = i16::from_le_bytes([value[offset + 8], value[offset + 9]]) as f32 / 16384.0;
        let qz = i16::from_le_bytes([value[offset + 10], value[offset + 11]]) as f32 / 16384.0;
        let qw = i16::from_le_bytes([value[offset + 12], value[offset + 13]]) as f32 / 16384.0;
        
        // Formato: [ax, ay, az, qw, qx, qy, qz]
        let dst = map_idx[i];
        frame[dst] = Some([ax, ay, az, qw, qx, qy, qz]);
        
        offset += 14;
    }
    
    Some(frame)
}

/// Obtiene las estadísticas actuales de BLE
pub fn get_stats() -> BleStats {
    BleStats {
        superframes: SUPERFRAMES.load(Ordering::Relaxed),
        lost_frames: LOST_FRAMES.load(Ordering::Relaxed),
    }
}