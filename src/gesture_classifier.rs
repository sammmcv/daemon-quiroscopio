use crate::feature_extractor::FeatureExtractor;
use crate::types::{GestureWindow, VotingWindows, TOTAL_EXTRACTED_FEATURES};
use ort::session::Session;
use ort::tensor::TensorElementType;
use ort::value::ValueType;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ClassifierError {
    #[error("ONNX Runtime error: {0}")]
    OnnxError(#[from] ort::Error),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),

    #[error("Invalid feature size: expected {expected}, got {actual}")]
    InvalidFeatureSize { expected: usize, actual: usize },

    #[error("No output tensor found")]
    NoOutputTensor,

    #[error("Missing ONNX {kind}")]
    MissingIo { kind: &'static str },
}

#[derive(Debug, Deserialize)]
struct ClassesJson {
    index_to_class: HashMap<String, String>,
}

pub struct GestureClassifier {
    session: Session,
    labels: Vec<String>,
    feature_extractor: FeatureExtractor,
    input_name: String,
    prob_output_name: String,
}

impl GestureClassifier {
    pub fn new(model_path: &str, classes_path: &str) -> Result<Self, ClassifierError> {
        // Cargar clases
        let labels = Self::load_classes(classes_path)?;

        // Cargar modelo ONNX
        let session = Session::builder()?.commit_from_file(model_path)?;

        let input_name = session
            .inputs
            .get(0)
            .map(|input| input.name.clone())
            .ok_or(ClassifierError::MissingIo { kind: "input" })?;

        let prob_output_name = session
            .outputs
            .iter()
            .find(|output| {
                matches!(
                    output.output_type,
                    ValueType::Tensor {
                        ty: TensorElementType::Float32,
                        ..
                    }
                )
            })
            .or_else(|| session.outputs.get(0))
            .map(|output| output.name.clone())
            .ok_or(ClassifierError::MissingIo { kind: "output" })?;

        println!("[ONNX] Modelo cargado: {}", model_path);
        println!("[ONNX] Clases: {:?}", labels);
        println!("[ONNX] Input: {}", input_name);
        println!("[ONNX] Output: {}", prob_output_name);

        Ok(Self {
            session,
            labels,
            feature_extractor: FeatureExtractor::new(),
            input_name,
            prob_output_name,
        })
    }

    fn load_classes(path: &str) -> Result<Vec<String>, ClassifierError> {
        let content = fs::read_to_string(path)?;
        let data: ClassesJson = serde_json::from_str(&content)?;

        // Convertir HashMap a Vec ordenado por índice
        let mut pairs: Vec<(usize, String)> = data
            .index_to_class
            .into_iter()
            .filter_map(|(k, v)| k.parse::<usize>().ok().map(|idx| (idx, v)))
            .collect();

        pairs.sort_by_key(|(idx, _)| *idx);
        Ok(pairs.into_iter().map(|(_, name)| name).collect())
    }

    /// Predice el gesto de una ventana individual
    pub fn predict_single(
        &mut self,
        window: &GestureWindow,
    ) -> Result<(String, f32), ClassifierError> {
        let scores = self.predict_scores(window)?;

        // Encontrar máxima probabilidad
        let (label, &score) = scores
            .iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        Ok((label.clone(), score))
    }

    /// Predice probabilidades para todas las clases
    pub fn predict_scores(
        &mut self,
        window: &GestureWindow,
    ) -> Result<HashMap<String, f32>, ClassifierError> {
        // Extraer características [250]
        let features = self.feature_extractor.extract(window);

        if features.len() != TOTAL_EXTRACTED_FEATURES {
            return Err(ClassifierError::InvalidFeatureSize {
                expected: TOTAL_EXTRACTED_FEATURES,
                actual: features.len(),
            });
        }

        // Preparar tensor de entrada [1, 250]
        // ort 2.x requiere OwnedTensorArrayData: (shape, data) donde shape es &[usize], Vec<usize>, etc.
        let input_data = features; // Ya es Vec<f32>
        let shape_vec = vec![1_usize, TOTAL_EXTRACTED_FEATURES];

        // Crear un Value de ONNX Runtime usando la tupla (Vec<usize>, Vec<f32>)
        let input_value = ort::value::Value::from_array((shape_vec, input_data))?;

        // Ejecutar inferencia
        let outputs = self.session.run(ort::inputs![
            self.input_name.as_str() => &input_value,
        ])?;

        // Extraer probabilidades del output dinámico
        let (prob_shape, prob_data) =
            outputs[self.prob_output_name.as_str()].try_extract_tensor::<f32>()?;

        // Crear mapa de resultados
        let mut scores = HashMap::new();
        let num_classes = if prob_shape.len() >= 2 {
            prob_shape[1] as usize
        } else {
            prob_shape[0] as usize
        };

        for (i, label) in self.labels.iter().enumerate().take(num_classes) {
            let score = prob_data[i];
            scores.insert(label.clone(), score);
        }

        Ok(scores)
    }

    /// Sistema de votación: predice usando 5 ventanas y vota
    pub fn predict_with_voting(
        &mut self,
        windows: &VotingWindows,
    ) -> Result<(String, f32), ClassifierError> {
        let mut vote_counts: HashMap<String, usize> = HashMap::new();
        let mut accumulated_scores: HashMap<String, f32> = HashMap::new();

        // Predecir con cada ventana
        for window in windows {
            let (predicted_label, confidence) = self.predict_single(window)?;

            // Incrementar voto
            *vote_counts.entry(predicted_label.clone()).or_insert(0) += 1;

            // Acumular score
            *accumulated_scores.entry(predicted_label).or_insert(0.0) += confidence;
        }

        // Encontrar ganador por votos
        let mut winner_label: Option<String> = None;
        let mut winner_votes: usize = 0;
        let mut winner_score_sum: f32 = f32::MIN;
        for (label, &votes) in &vote_counts {
            let score_sum = *accumulated_scores.get(label).unwrap_or(&0.0);
            let better =
                votes > winner_votes || (votes == winner_votes && score_sum > winner_score_sum);
            if winner_label.is_none() || better {
                winner_label = Some(label.clone());
                winner_votes = votes;
                winner_score_sum = score_sum;
            }
        }
        let winner = winner_label.ok_or(ClassifierError::NoOutputTensor)?;
        let votes = winner_votes.max(1) as f32;
        let avg_confidence = winner_score_sum / votes;

        Ok((winner, avg_confidence))
    }

    /// Obtiene las etiquetas de clases
    pub fn get_labels(&self) -> &[String] {
        &self.labels
    }
}
