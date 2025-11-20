use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{anyhow, bail, ensure, Context, Result};
use csv::ReaderBuilder;

use crate::types::{GestureWindow, SampleFrame, NUM_SENSORS, TOTAL_WINDOW_FEATURES, WINDOW_SIZE};

/// Carga una secuencia de SampleFrame desde un CSV en el formato
/// sample,sensor,ax,ay,az,w,i,j,k ordenado por sample y sensor.
pub fn load_frames_from_csv(path: impl AsRef<Path>) -> Result<Vec<SampleFrame>> {
    let path = path.as_ref();
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_path(path)
        .with_context(|| format!("No se pudo abrir el CSV {:?}", path))?;

    let mut samples: BTreeMap<usize, SampleFrame> = BTreeMap::new();

    for (row_idx, result) in reader.records().enumerate() {
        let record =
            result.with_context(|| format!("Fila {} inválida en {:?}", row_idx + 1, path))?;
        if record.len() < 9 {
            bail!("La fila {} no tiene 9 columnas", row_idx + 1);
        }

        let sample: usize = record[0]
            .parse()
            .with_context(|| format!("sample inválido en fila {}", row_idx + 1))?;
        let sensor: usize = record[1]
            .parse()
            .with_context(|| format!("sensor inválido en fila {}", row_idx + 1))?;

        if sensor >= NUM_SENSORS {
            bail!("Sensor {} fuera de rango (fila {})", sensor, row_idx + 1);
        }

        let ax: f32 = record[2].parse()?;
        let ay: f32 = record[3].parse()?;
        let az: f32 = record[4].parse()?;
        let qw: f32 = record[5].parse()?;
        let qx: f32 = record[6].parse()?;
        let qy: f32 = record[7].parse()?;
        let qz: f32 = record[8].parse()?;

        let frame = samples.entry(sample).or_insert_with(SampleFrame::default);
        frame.ax[sensor] = ax;
        frame.ay[sensor] = ay;
        frame.az[sensor] = az;
        frame.qw[sensor] = qw;
        frame.qx[sensor] = qx;
        frame.qy[sensor] = qy;
        frame.qz[sensor] = qz;
    }

    if samples.is_empty() {
        return Err(anyhow!("El CSV {:?} no contiene datos", path));
    }

    let (&min_sample, _) = samples.iter().next().unwrap();
    ensure!(
        min_sample == 0,
        "El CSV debe iniciar en sample=0 (encontrado sample={})",
        min_sample
    );
    let max_sample = *samples.keys().max().unwrap();

    let mut frames = Vec::with_capacity(max_sample + 1);
    let mut last_frame = SampleFrame::default();
    for sample_idx in 0..=max_sample {
        if let Some(frame) = samples.get(&sample_idx) {
            last_frame = *frame;
            frames.push(*frame);
        } else {
            // Rellenar huecos repitiendo la última muestra válida
            frames.push(last_frame);
        }
    }

    Ok(frames)
}

/// Reconstruye exactamente una ventana de 16 frames desde un CSV.
pub fn load_window_from_csv(path: impl AsRef<Path>) -> Result<GestureWindow> {
    let mut frames = load_frames_from_csv(path)?;
    if frames.len() < WINDOW_SIZE {
        let pad = *frames.last().unwrap_or(&SampleFrame::default());
        frames.resize(WINDOW_SIZE, pad);
    } else if frames.len() > WINDOW_SIZE {
        frames.truncate(WINDOW_SIZE);
    }
    Ok(frames)
}

/// Devuelve la ventana aplanada en formato [t * 5 * 7 + sensor * 7 + canal]
pub fn flatten_window(window: &GestureWindow) -> Vec<f32> {
    let mut flat = vec![0.0; TOTAL_WINDOW_FEATURES];
    for (t, frame) in window.iter().enumerate() {
        frame.to_flat_array(&mut flat, t);
    }
    flat
}
