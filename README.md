# Sistema de Reconocimiento de Gestos BLE

Sistema de reconocimiento de gestos que utiliza sensores IMU conectados por Bluetooth Low Energy (BLE). Desarrollado en Rust para el procesamiento de datos, con integración de modelos de Machine Learning en Python para la clasificación y traducción de gestos.

## Estructura del proyecto

```
├── src/
│   ├── lib.rs                 # Punto común para los módulos públicos
│   ├── main.rs                # Binario principal (BLE en tiempo real)
│   ├── bin/
│   │   └── replay_csv.rs      # Reproducción offline de gestos desde CSV
│   ├── ble.rs                 # Comunicación BLE y parsing de frames
│   ├── csv_loader.rs          # Utilidades para reconstruir ventanas desde CSV
│   ├── feature_extractor.rs   # Extracción de 250 features (tiempo + FFT)
│   ├── gesture_classifier.rs  # Wrapper de ONNX Runtime + sistema de votación
│   ├── gesture_extractor.rs   # Detector de gestos y generación de 5 ventanas
│   ├── hid.rs                 # Emulación HID vía /dev/uinput
│   └── types.rs               # Tipos compartidos (SampleFrame, ventanas, etc.)
├── cpp/                       # Referencia C++ original
├── gestos_auto_rust/          # Ventanas CSV capturadas para debug/offline
├── best_pipeline__time+fft__svm_rbf.onnx
├── classes.json
├── Cargo.toml
└── README.md
```

## Flujo de ejecución

1. **Captura BLE**: Conexión a sensores IMU y recepción de datos
2. **Procesamiento**: Acumulación de datos en ventanas de tiempo y extracción de características
3. **Clasificación**: Inferencia de gestos mediante modelos de ML (integración PyO3)
4. **Salida**: Generación de eventos de entrada del sistema o visualización de resultados

## Requisitos del sistema

- **Rust**: Versión estable reciente ([Instalar](https://rustup.rs/))
- **Python**: Versión 3.8 o superior
- **Bluetooth**: Adaptador BLE compatible

## Instalación

### 1. Clonar el repositorio
```bash
git clone <repository-url>
cd rust
```

### 2. Instalar dependencias de Python
```bash
pip install numpy scipy scikit-learn joblib pandas
```

### 3. Compilar el proyecto
```bash
# Modo desarrollo
cargo build

# Modo producción (optimizado)
cargo build --release
```

## Uso

### Reconocimiento en tiempo continuo
```bash
# Exporta ONNX Runtime en el LD_LIBRARY_PATH (ajusta la ruta según tu entorno)
set -x LD_LIBRARY_PATH onnxruntime-linux-x64-1.22.0/lib $LD_LIBRARY_PATH

# Ejecuta el binario principal con la MAC del gateway BLE
cargo run --release -- 28:CD:C1:08:37:69
```

### Modo replay desde CSV
```bash
# Clasifica una ventana guardada en disco y muestra el top-5
cargo run --release --bin replay_csv -- gestos_auto_rust/gesto__00000.csv

# Opcional: imprime el tensor plano (16×5×7) y los 250 features
cargo run --release --bin replay_csv -- --dump-flat --dump-features gestos_auto_rust/gesto___00042.csv
```

El replay ejecuta exactamente el `FeatureExtractor` de Rust y el modelo SVM en ONNX.
Además replica las 5 ventanas de votación (rellenadas desde la ventana centrada)
para que puedas contrastar las probabilidades con el binario C++.

## Configuración

Los parámetros principales del sistema pueden ajustarse según las necesidades:

- **Umbral de confianza**: Nivel mínimo de certeza para aceptar predicciones
- **Tamaño de ventana**: Cantidad de muestras para análisis temporal
- **Umbral de movimiento**: Sensibilidad de detección de gestos
- **Sensores activos**: Configuración de dispositivos IMU

Consulta los archivos de código fuente para modificar estos parámetros.

## Características principales

- **Procesamiento continuo**: Captura y análisis continuo de datos IMU
- **Integración Rust-Python**: Combinación de rendimiento de Rust con ecosistema ML de Python
- **Múltiples sensores**: Soporte para configuraciones multi-sensor
- **Estabilización de predicciones**: Sistema de votación para reducir falsos positivos
- **Interfaz HID**: Generación de eventos de entrada del sistema
- **Formato flexible**: Capacidad de procesar datos desde dispositivos BLE o archivos CSV

## Gestos soportados

El sistema puede reconocer diferentes tipos de gestos, incluyendo:
- Movimientos de agarrar y soltar
- Deslizamientos direccionales
- Gestos de zoom

Los gestos específicos y sus configuraciones se encuentran en el directorio `gestos/`.

## Arquitectura técnica

### Comunicación BLE
- Protocolo de bajo nivel para comunicación con sensores IMU
- Manejo de múltiples dispositivos simultáneos
- Procesamiento asíncrono de frames

### Pipeline de clasificación
- Extracción de características temporales y espectrales
- Modelos de Machine Learning optimizados
- Procesamiento vectorizado de datos

### Interfaz de salida
- Generación de eventos del sistema
- Compatibilidad con protocolos HID
- Logging y visualización de predicciones

## Resolución de problemas

### Problemas comunes

**Error de módulo Python**
- Verifica que los archivos del pipeline estén en el directorio correcto
- Asegúrate de que todas las dependencias de Python estén instaladas

**Problemas de conexión BLE**
- Confirma que el adaptador Bluetooth esté activo
- Verifica que el dispositivo esté encendido y dentro del rango
- Comprueba los permisos de acceso a Bluetooth en tu sistema

**Rendimiento lento**
- Utiliza la versión `--release` para mejor performance
- Verifica que no haya otros procesos consumiendo recursos
- Considera ajustar los parámetros de ventana y umbrales

**Errores de compilación**
- Actualiza Rust a la versión más reciente
- Verifica que todas las dependencias estén disponibles
- Revisa la compatibilidad de las versiones de las librerías

## Desarrollo

### Estructura de módulos

El proyecto está organizado en módulos independientes que pueden ser modificados según necesidades:

- **BLE**: Gestión de comunicación con sensores
- **Buffer**: Manejo de datos temporales
- **Extractor**: Procesamiento de características
- **HID**: Interfaz con el sistema operativo
- **Main**: Orquestación del flujo general

### Añadir nuevos gestos

1. Captura datos del gesto en formato CSV
2. Organiza los archivos en el directorio `gestos/`
3. Actualiza el modelo de clasificación según sea necesario
4. Ajusta los parámetros de reconocimiento si es requerido

## Licencia

Consulta el archivo LICENSE.txt para más información.

## Notas adicionales

- El sistema utiliza PyO3 para integración directa Rust-Python sin overhead de IPC
- Los datos IMU incluyen acelerómetro y cuaterniones de orientación
- El procesamiento puede funcionar con sensores faltantes mediante manejo de opcionales
- La arquitectura permite agregar nuevos tipos de salida o procesamiento fácilmente
