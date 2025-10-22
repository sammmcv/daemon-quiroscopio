# Reconocimiento de gestos con Rust y Python

Este proyecto integra procesamiento de gestos IMU usando Rust y un pipeline de Machine Learning en Python.

## Estructura del Workspace

```
rust/
├── src/
│   └── main.rs                    # Binario principal: procesa CSVs y ejecuta predicciones
├── python/
│   ├── gesture_infer.py           # Pipeline de clasificación Python
│   ├── best_pipeline__time-summary__svm_linear.joblib  # Modelo SVM entrenado
│   ├── robust_scaler_acc.joblib   # Scaler para acelerómetros
│   └── labels.json                # Lista de clases de gestos
├── gesto-drop/                    # Carpeta con CSVs del gesto "drop"
│   ├── 000001_plot.csv
│   ├── 000002_plot.csv
│   └── ...
├── gesto-grab/                    # Carpeta con CSVs del gesto "grab"
├── gesto-slide-derecha/           # Carpeta con CSVs del gesto "slide-derecha"
├── gesto-slide-izquierda/         # Carpeta con CSVs del gesto "slide-izquierda"
├── gesto-zoom-in/                 # Carpeta con CSVs del gesto "zoom-in"
├── gesto-zoom-out/                # Carpeta con CSVs del gesto "zoom-out"
├── gesto-test/                    # Carpeta con CSVs de prueba
├── Cargo.toml                     # Configuración del proyecto Rust
└── README.md                      # Este archivo
```

## Flujo de Ejecución

```mermaid
graph LR
    A[Carpetas gesto-*] --> B[main.rs]
    B --> C[Lee CSV 64x5x7]
    C --> D[Convierte a Array]
    D --> E[PyO3: Embebe Python]
    E --> F[gesture_infer.py]
    F --> G[Normaliza cuaterniones]
    G --> H[Aplica RobustScaler]
    H --> I[Extrae 420 features]
    I --> J[Pipeline SVM]
    J --> K[Predicción + Confianza]
    K --> L[Muestra Resultados]
```

### Paso a paso:

1. **Rust detecta carpetas** — El binario busca todas las carpetas que empiezan con `gesto-*`
2. **Lee CSVs** — Por cada carpeta, lee los primeros 30 archivos CSV ordenados alfabéticamente
3. **Parsea ventana IMU** — Cada CSV se convierte en una ventana `[64 samples, 5 sensors, 7 channels]`
   - Channels: `[ax, ay, az, w, i, j, k]` (acelerómetro + cuaternión)
4. **Embebe Python con PyO3** — Rust inicializa el intérprete Python in-process (sin subprocesos)
5. **Carga el clasificador** — Importa `GestureClassifier` desde `python/gesture_infer.py`
6. **Pipeline de ML Python**:
   - Normaliza cuaterniones (w ≥ 0)
   - Aplica `RobustScaler` a acelerómetros
   - Extrae 420 features de estadísticas (5 sensores × 7 canales × 12 stats)
     - Stats: mean, std, min, max, p25, p50, p75, energy, rms, beta, skew, kurt
   - Ejecuta el pipeline SVM lineal
7. **Retorna predicción** — Label del gesto + confianza (0-1)
8. **Valida y muestra** — Rust compara con el nombre de la carpeta y calcula métricas

## Instalación y Dependencias

### Prerequisitos
- **Rust 2021+** — [Instalar](https://rustup.rs/)
- **Python 3.8+** — Con pip instalado

### Dependencias Rust
El proyecto usa las siguientes crates:
- `pyo3` — Embebe el intérprete Python
- `numpy` — Puente Rust ↔ NumPy arrays
- `csv` — Lectura de archivos CSV
- `anyhow` — Manejo de errores
- `serde_json` — Serialización JSON

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

### Método 1: Con cargo
```bash
cargo run --release
```

### Método 2: Binario directo
```bash
./target/release/onnx-predictor
```

### Método 3: Debug resumido
```bash
./target/release/onnx-predictor 2>&1 | grep -E "📁|📊|📈|✅ Procesamiento"
```

## Ejemplo de Salida

```
🎯 Gesture Recognition System

✅ Clasificador cargado desde python/

📂 Carpetas encontradas: 7

📁 Procesando: gesto-drop
  📄 Archivos: 30
  ✅ 000001_plot.csv → gesto-drop (99.5%)
  ✅ 000002_plot.csv → gesto-drop (99.5%)
  ...
  📊 Precisión: 30/30 (100.0%)
  📈 Confianza promedio: 99.5%

📁 Procesando: gesto-grab
  📄 Archivos: 30
  ✅ 000001_plot.csv → gesto-grab (99.5%)
  ...
  📊 Precisión: 30/30 (100.0%)
  📈 Confianza promedio: 99.5%

✅ Procesamiento completado
```

### Símbolos de estado:
- ✅ — Predicción correcta y confianza ≥ 90%
- ⚠️  — Predicción correcta pero confianza < 90%
- ❌ — Predicción incorrecta

## Configuración y Personalización

### Cambiar umbral de confianza
Edita `src/main.rs`:
```rust
const UMBRAL: f32 = 0.90; // Cambia a 0.95 para más estrictez
```

### Procesar más/menos archivos
Por defecto procesa 30 CSVs por carpeta. Para cambiar:
```rust
csv_files.truncate(30); // Cambia el número
```

### Agregar nuevas clases
1. Crea una nueva carpeta `gesto-nuevo-gesto/`
2. Agrega archivos CSV con el formato esperado
3. Actualiza `python/labels.json` (si el modelo fue reentrenado)
4. Ejecuta el binario

### Formato de CSV esperado
Cada CSV debe tener:
- **Columnas**: `sample, sensor, ax, ay, az, w, i, j, k`
- **Filas**: 320 (64 samples × 5 sensors)
- **Valores**: Flotantes con acelerómetros y cuaterniones normalizados

Ejemplo:
```csv
sample,sensor,ax,ay,az,w,i,j,k
0,0,0.12,-0.34,9.81,0.98,0.01,0.02,0.03
0,1,0.15,-0.32,9.79,0.97,0.02,0.01,0.04
...
```

## Testing

### Validar un solo CSV
Modifica temporalmente `src/main.rs` para cargar un archivo específico, o usa el script Python:
```bash
python3 python/gesture_infer.py --artifacts python --csv gesto-drop/000001_plot.csv
```

### Comparar Rust vs Python
Ambos deben dar resultados idénticos:
```bash
# Rust
./target/release/onnx-predictor 2>&1 | grep "000001_plot"

# Python
python3 python/gesture_infer.py --artifacts python --csv gesto-drop/000001_plot.csv
```

## Arquitectura Técnica

### Por qué Rust + Python (PyO3)?
- **Rust**: Performance para I/O de archivos y orquestación
- **Python embebido**: Reutiliza el pipeline ML sin reescribir
- **PyO3**: Zero-copy entre Rust y NumPy arrays
- **Sin subprocesos**: Más rápido que llamar scripts externos

### Formato de ventana IMU
```
Shape: [64, 5, 7]
       │   │  └─ Channels: [ax, ay, az, w, i, j, k]
       │   └──── Sensors: 5 IMUs en la mano
       └────────── Samples: 64 timesteps @ ~100Hz
```

### Pipeline de features
El modelo **NO usa valores raw**. Extrae 420 estadísticas:
```
5 sensors × 7 channels × 12 statistics = 420 features
```

Estadísticas por canal:
1. `mean` — Media
2. `std` — Desviación estándar
3. `min` / `max` — Rango
4. `p25` / `p50` / `p75` — Percentiles
5. `energy` — Suma de cuadrados
6. `rms` — Root mean square
7. `beta` — Pendiente (regresión lineal)
8. `skew` — Asimetría
9. `kurt` — Curtosis

## Notas Importantes

- **Python embebido**: El sistema requiere que Python esté instalado en el sistema con todos los paquetes
- **Warnings de versión**: Los warnings de scikit-learn (1.6.1 vs 1.7.2) no afectan los resultados
- **Sin ONNX Runtime**: La versión actual usa el pipeline Python directo, no el modelo `.onnx` exportado
- **Memoria**: PyO3 mantiene Python en memoria, más eficiente que subprocesos
- **Carpeta `gesto-test`**: Se procesa pero suele tener precisión 0% (contiene muestras mezcladas)

## Troubleshooting

### Error: "ModuleNotFoundError: No module named 'gesture_infer'"
- Verifica que `python/gesture_infer.py` exista
- Ejecuta desde la raíz del proyecto, no desde subcarpetas

### Error: "FileNotFoundError: python/..."
- Verifica que los archivos `.joblib` y `labels.json` estén en `python/`

### Predicciones incorrectas
- Verifica que los CSV tengan el formato correcto
- Confirma que el modelo fue entrenado con el mismo preprocesado
- Revisa que los archivos estén en la carpeta correcta

### Performance lento
- Usa `--release` siempre: `cargo build --release`
- El primer archivo es más lento (carga del modelo)
