/*
Descargar el paquete
curl -L -o onnxruntime-linux-x64-1.22.0.tgz https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-linux-x64-1.22.0.tgz

Extraer el contenido
tar -xzf onnxruntime-linux-x64-1.22.0.tgz

para compilar:

cargo clean
cargo build --release
export ORT_DYLIB_PATH=/home/samcv/pico/rust/onnxruntime-linux-x64-1.22.0/lib/libonnxruntime.so
export LD_LIBRARY_PATH=/home/samcv/pico/rust/onnxruntime-linux-x64-1.22.0/lib:$LD_LIBRARY_PATH
./target/release/onnx-predictor

   o

cargo clean
cargo build --release
./run.sh


*/


use std::thread;
use std::time::Duration;
use ndarray::Array2;
use ort::{session::Session, value::Value};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Descargar y configurar ONNX Runtime automáticamente
    ort::init()
        .with_name("onnxruntime")
        .commit()?;

    // Cargar el modelo ONNX
    let mut session = Session::builder()?
        .commit_from_file("time-summary_model.onnx")?;

    loop {
        // Datos de entrada para la predicción - el modelo espera 420 valores
        let input_data = vec![1.0_f32; 420]; // Reemplaza con tus datos reales (420 valores)

        // Crear el array de entrada y convertirlo a Value
        let array = Array2::from_shape_vec((1, 420), input_data.clone())?;
        let input_value = Value::from_array(array)?;
        
        // Realizar la predicción
        let outputs = session.run(ort::inputs![input_value])?;
        
        // Obtener el resultado - el modelo devuelve Strings
        let prediction = outputs[0].try_extract_strings()?;
        
        println!("Predicción: {}", prediction.1[0]);

        // Simulamos que el daemon se ejecuta cada cierto tiempo
        thread::sleep(Duration::from_secs(5)); // Espera 5 segundos
    }
}
