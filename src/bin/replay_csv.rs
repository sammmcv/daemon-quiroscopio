use std::env;
use std::path::PathBuf;

use anyhow::{anyhow, bail, Result};
use quiroscopio::csv_loader::{flatten_window, load_window_from_csv};
use quiroscopio::feature_extractor::FeatureExtractor;
use quiroscopio::gesture_classifier::GestureClassifier;
use quiroscopio::types::{VotingWindows, WINDOW_SIZE};

struct ReplayOptions {
    dump_flat: bool,
    dump_features: bool,
}

fn parse_args() -> Result<(PathBuf, ReplayOptions)> {
    let mut dump_flat = false;
    let mut dump_features = false;
    let mut csv_path: Option<PathBuf> = None;

    for arg in env::args().skip(1) {
        match arg.as_str() {
            "--dump-flat" => dump_flat = true,
            "--dump-features" => dump_features = true,
            _ => {
                if csv_path.is_some() {
                    bail!("Uso: replay_csv [--dump-flat] [--dump-features] <archivo.csv>");
                }
                csv_path = Some(PathBuf::from(arg));
            }
        }
    }

    let csv_path = csv_path.ok_or_else(|| anyhow!("Debes especificar un archivo CSV"))?;
    Ok((
        csv_path,
        ReplayOptions {
            dump_flat,
            dump_features,
        },
    ))
}

fn main() -> Result<()> {
    let (csv_path, opts) = parse_args()?;
    println!("🎞️  Reproduciendo gesto desde {:?}", csv_path);

    let window = load_window_from_csv(&csv_path)?;
    if window.len() != WINDOW_SIZE {
        println!("ℹ️  Ventana ajustada a {} frames", WINDOW_SIZE);
    }

    let mut classifier =
        GestureClassifier::new("best_pipeline__time+fft__svm_rbf.onnx", "classes.json")?;

    let mut scores: Vec<(String, f32)> = classifier.predict_scores(&window)?.into_iter().collect();
    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let (best_label, best_conf) = scores
        .first()
        .cloned()
        .ok_or_else(|| anyhow!("El modelo no devolvió probabilidades"))?;

    let windows: VotingWindows = std::array::from_fn(|_| window.clone());
    let (voted_label, voted_conf) = classifier.predict_with_voting(&windows)?;

    println!(
        "\n🥇 Ventana centrada: {} ({:.1}%)",
        best_label,
        best_conf * 100.0
    );
    println!(
        "🗳️  Votación (ventanas replicadas): {} ({:.1}%)",
        voted_label,
        voted_conf * 100.0
    );

    println!("\nTop-5 probabilidades:");
    for (idx, (label, score)) in scores.iter().take(5).enumerate() {
        println!("  {:>2}. {:<25} {:>6.2}%", idx + 1, label, score * 100.0);
    }

    if opts.dump_features {
        let mut extractor = FeatureExtractor::new();
        let features = extractor.extract(&window);
        println!("\n📊 250 features (orden exacto):");
        for (idx, value) in features.iter().enumerate() {
            println!("  {:03}: {:>12.6}", idx, value);
        }
    }

    if opts.dump_flat {
        let flat = flatten_window(&window);
        println!("\n🧱 Tensor plano ({} valores):", flat.len());
        for (idx, value) in flat.iter().enumerate() {
            println!("  {:03}: {:>12.6}", idx, value);
        }
    }

    Ok(())
}
