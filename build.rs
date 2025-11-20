fn main() {
    // Configurar la ruta de búsqueda para ONNX Runtime
    println!("cargo:rustc-link-search=native=onnxruntime-linux-x64-1.22.0/lib");
    println!("cargo:rustc-link-lib=dylib=onnxruntime");
    
    // Recompilar si cambia el directorio de ONNX Runtime
    println!("cargo:rerun-if-changed=onnxruntime-linux-x64-1.22.0/");
}
