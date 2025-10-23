# Reconocimiento de gestos BLE en tiempo real con Rust + Python

Este proyecto permite reconocer gestos de mano en tiempo real usando sensores IMU conectados por Bluetooth Low Energy (BLE), procesando los datos en Rust y ejecutando la inferencia de gestos con un pipeline de Machine Learning en Python.

## Estructura del proyecto

```
├── src/
│   ├── main.rs                # Binario principal: recibe frames BLE y ejecuta inferencia
│   ├── ble.rs                 # Módulo BLE: conexión y decodificación de frames
│   └── gesture_buffer.rs      # Buffer circular para ventanas IMU
├── python/
│   ├── gesture_infer.py       # Pipeline de clasificación Python
│   ├── best_pipeline__*.joblib  # Modelo SVM entrenado
│   ├── robust_scaler_acc.joblib # Scaler para acelerómetros
│   └── labels.json            # Lista de clases de gestos
├── Cargo.toml                 # Configuración Rust
└── README.md                  # Este archivo
```

## Flujo de ejecución

1. **Conexión BLE**: Rust se conecta a los sensores IMU vía Bluetooth y recibe frames en tiempo real.
2. **Buffer de ventanas**: Los datos se acumulan en un buffer circular de 64 muestras por sensor.
3. **Inferencia Python**: Cuando hay suficiente movimiento, Rust llama al pipeline Python embebido (PyO3) para predecir el gesto.
4. **Votación y salida**: Se estabilizan las predicciones con votación y se muestra el resultado en consola.

## Instalación y dependencias

### Requisitos
- **Rust 2021+** — [Instalar](https://rustup.rs/)
- **Python 3.8+** — Con pip instalado

### Dependencias Rust
En `Cargo.toml`:
- `pyo3` — Embebe el intérprete Python
- `numpy` — Puente Rust ↔ NumPy arrays
- `anyhow`, `serde_json`, `dbus`, `crossbeam-channel`

Instala con:
```bash
cargo build --release
```

### Dependencias Python
Instala los paquetes requeridos:
```bash
pip install numpy scipy scikit-learn joblib pandas
```

## Ejecución

### Reconocimiento en tiempo real (BLE)
```bash
./target/release/onnx-predictor <MAC_ADDRESS>
# Ejemplo:
./target/release/onnx-predictor 28:CD:C1:08:37:69
```

### Inferencia por CSV (Python)
```bash
python3 python/gesture_infer.py --artifacts python --csv gesto-drop/000001_plot.csv
```

## Ejemplo de salida

```
🎯 Gesture Recognition System - BLE Real-Time

✅ Clasificador cargado
┌──────────────────────────────────────────────────────────────────┐
│  Frames │ Predicción          │ Conf.  │ Votación     │ Mov.   │
├──────────────────────────────────────────────────────────────────┤
│    1234 │ ✅ gesto-grab       │  99.2% │ 3/3          │ [mov:1.23]
└──────────────────────────────────────────────────────────────────┘
```

## Personalización

- Cambia el umbral de confianza en `src/main.rs`:
  ```rust
  const CONFIDENCE_THRESHOLD: f32 = 0.85;
  ```
- Cambia el umbral de movimiento para detectar gestos reales:
  ```rust
  const MOVEMENT_THRESHOLD: f32 = 1.0;
  ```

## Formato de datos IMU

- Cada frame: `[Option<[f32; 7]>; 5]` (5 sensores, 7 canales: ax, ay, az, w, i, j, k)
- Ventana para inferencia: `[64, 5, 7]`

## Troubleshooting

- **Error de módulo Python**: Verifica que `python/gesture_infer.py` exista y que los artefactos estén en la carpeta correcta.
- **Error de conexión BLE**: Asegúrate de que el adaptador Bluetooth esté encendido y el dispositivo disponible.
- **Performance lento**: Usa `--release` y asegúrate de que los paquetes Python estén instalados.

## Notas técnicas

- El pipeline Python extrae 420 features por ventana (5 sensores × 7 canales × 12 estadísticas).
- El sistema embebe Python en Rust usando PyO3, sin subprocesos externos.
- El buffer circular permite ventanas deslizantes y manejo de sensores ausentes.