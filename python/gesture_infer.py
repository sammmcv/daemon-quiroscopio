# -*- coding: utf-8 -*-
"""
gesture_infer.py
Inferencia de gestos a partir de ventanas IMU (64x5x7) usando el pipeline entrenado (.joblib).

Requisitos:
  pip install numpy scipy scikit-learn joblib

Artefactos esperados (ajusta rutas por --artifacts):
  - best_pipeline__time-summary__svm_linear.joblib   (o el nombre que guardaste)
  - robust_scaler_acc.joblib                         (del build del dataset)
  - labels.json                                      (lista de clases en orden)
Opcionales para probabilidad calibrada:
  - preproc.joblib
  - clf_calibrated.joblib

Autor: tú :)
"""

import json
import math
import argparse
from pathlib import Path
from typing import Optional, Tuple, List, Deque

import numpy as np
from joblib import load
from collections import deque

# -----------------------------
# Config por defecto (ajústalo o usa argumentos CLI)
# -----------------------------
DEFAULT_ARTIFACTS = "./"  # cambia a tu ARTIFACTS_DIR
DEFAULT_OUT_DIR = "./"    # cambia a tu OUT_DIR si está distinto

WINDOW_T = 64
SENSORS = 5
CHANNELS = 7  # [ax,ay,az,w,i,j,k]

# -----------------------------
# Utilidades de preproceso
# -----------------------------
def normalize_quats(Q: np.ndarray) -> np.ndarray:
    """Normaliza cuaterniones y fuerza hemisferio w >= 0.
    Q: [...,4] -> [...,4]
    """
    Q = Q / (np.linalg.norm(Q, axis=-1, keepdims=True) + 1e-9)
    flip = (Q[...,0] < 0)[...,None]
    return Q * np.where(flip, -1.0, 1.0)

def time_summary_features(tensor: np.ndarray) -> np.ndarray:
    """Extrae las mismas features time-summary usadas en el entrenamiento.
    tensor: [T,S,C] con C=[ax,ay,az,w,i,j,k]
    return: (F,) vector 1D np.float32
    """
    from scipy.stats import skew, kurtosis
    T, S, C = tensor.shape
    assert T == WINDOW_T and S == SENSORS and C == CHANNELS, f"tensor shape esperado (64,5,7), recibido {tensor.shape}"
    out = []
    for s in range(S):
        X = tensor[:, s, :]  # [T,7]
        mean_ = X.mean(0)
        std_  = X.std(0) + 1e-9
        min_  = X.min(0); max_ = X.max(0)
        p25 = np.percentile(X, 25, axis=0); p50 = np.percentile(X, 50, axis=0); p75 = np.percentile(X, 75, axis=0)
        energy = (X**2).sum(0)/T
        rms = np.sqrt((X**2).mean(0))
        t = np.arange(T)[:, None]
        beta = ((t - t.mean()) * (X - X.mean(0))).sum(0) / (((t - t.mean())**2).sum() + 1e-9)
        sk = skew(X, axis=0, bias=False)
        ku = kurtosis(X, axis=0, bias=False)
        out.append(np.concatenate([mean_, std_, min_, max_, p25, p50, p75, energy, rms, beta, sk, ku]))
    return np.concatenate(out).astype(np.float32)

# -----------------------------
# Clasificador
# -----------------------------
class GestureClassifier:
    def __init__(self,
                 artifacts_dir: Path,
                 pipeline_filename: Optional[str] = None,
                 robust_scaler_filename: str = "robust_scaler_acc.joblib",
                 labels_filename: str = "labels.json",
                 try_calibrated: bool = True):
        """
        artifacts_dir: carpeta con .joblib y labels.json
        pipeline_filename: nombre del pipeline, si None busca el primero que matchee 'best_pipeline__*.joblib'
        try_calibrated: si existen preproc.joblib + clf_calibrated.joblib, usa eso para probas calibradas
        """
        self.artifacts_dir = Path(artifacts_dir)
        self.pipeline_path = None
        if pipeline_filename is None:
            cand = sorted(self.artifacts_dir.glob("best_pipeline__*.joblib"))
            if not cand:
                raise FileNotFoundError("No se encontró ningún best_pipeline__*.joblib en artifacts_dir.")
            self.pipeline_path = cand[0]
        else:
            self.pipeline_path = self.artifacts_dir / pipeline_filename
        if not self.pipeline_path.exists():
            raise FileNotFoundError(f"No existe pipeline: {self.pipeline_path}")

        self.pipeline = load(self.pipeline_path)

        self.robust_scaler = load(self.artifacts_dir / robust_scaler_filename)
        labels_path = self.artifacts_dir / labels_filename
        self.labels = json.loads(labels_path.read_text(encoding="utf-8"))["classes"]

        # Opcional: calibrado
        self._cal_preproc = None
        self._cal_clf = None
        if try_calibrated:
            preproc_path = self.artifacts_dir / "preproc.joblib"
            cal_path = self.artifacts_dir / "clf_calibrated.joblib"
            if preproc_path.exists() and cal_path.exists():
                self._cal_preproc = load(preproc_path)
                self._cal_clf = load(cal_path)

    # ---------- Prepro de ventana cruda ----------
    def _preprocess_window(self, window_imu: np.ndarray) -> np.ndarray:
        """window_imu: [64,5,7] -> features (1,F)"""
        ten = window_imu.astype(np.float32).copy()
        if ten.shape != (WINDOW_T, SENSORS, CHANNELS):
            raise ValueError(f"Esperado (64,5,7), recibido {ten.shape}")

        # 1) Normalizar cuats
        ten[...,3:7] = normalize_quats(ten[...,3:7])

        # 2) RobustScaler en acelerómetros
        acc = ten[...,:3].reshape(-1, 3)                  # [64*5,3]
        ten[...,:3] = self.robust_scaler.transform(acc).reshape(WINDOW_T, SENSORS, 3)

        # 3) Features time-summary
        x = time_summary_features(ten).reshape(1, -1)
        return x

    # ---------- Predicciones ----------
    def predict(self, window_imu: np.ndarray, tau: Optional[float]=None) -> Tuple[str, float]:
        """
        Predice gesto para UNA ventana [64,5,7].
        Retorna (label, confidence). Si no hay calibrador y SVM no tiene probas, confidence=1.0.
        Si se pasa tau y hay probas, retorna ("desconocido", conf) si max_prob < tau.
        """
        X = self._preprocess_window(window_imu)

        # Ruta calibrada si está disponible
        if self._cal_preproc is not None and self._cal_clf is not None:
            Z = self._cal_preproc.transform(X)
            probs = self._cal_clf.predict_proba(Z)[0]
            k = int(np.argmax(probs))
            conf = float(probs[k])
            label = self._cal_clf.classes_[k]
            if tau is not None and conf < tau:
                return "desconocido", conf
            return str(label), conf

        # Ruta pipeline directa (SVM lineal sin probas)
        y = self.pipeline.predict(X)[0]
        # Confianza aproximada: usa distancia de decisión si existe
        conf = 1.0
        if hasattr(self.pipeline, "decision_function"):
            try:
                df = self.pipeline.decision_function(X)  # (1,K) o (1,)
                if isinstance(df, list) or isinstance(df, tuple):
                    df = np.array(df)
                df = np.atleast_1d(df)
                # mapea márgenes a [0,1] con una logística suave
                m = float(np.max(df))
                conf = 1.0 / (1.0 + math.exp(-m))
            except Exception:
                pass
        return str(y), float(conf)

    # ---------- Streaming deslizante ----------
    def stream_predict(self,
                       samples_iter,
                       hop: int = 1,
                       vote: int = 5,
                       tau: Optional[float] = None):
        """
        Predicción en streaming sobre un iterador de muestras IMU de forma (5,7).
        'samples_iter' debe producir tuplas/arrays shape (5,7) en orden temporal.
        Mantiene una ventana deslizante de 64 pasos; cada 'hop' hace una predicción.
        'vote': salida aplica mayoría en las últimas 'vote' predicciones para estabilizar.
        'tau': umbral de rechazo si hay probabilidades calibradas.

        yield dict(timestamp=idx, label=..., conf=..., voted_label=...)
        """
        buf = deque(maxlen=WINDOW_T)  # [(5,7), ...]
        pred_hist: Deque[str] = deque(maxlen=vote)
        conf_hist: Deque[float] = deque(maxlen=vote)
        i = 0
        for sample in samples_iter:
            arr = np.asarray(sample, dtype=np.float32)
            if arr.shape != (SENSORS, CHANNELS):
                raise ValueError(f"Cada muestra debe ser (5,7); recibido {arr.shape}")
            buf.append(arr)
            if len(buf) < WINDOW_T:
                i += 1
                continue
            if (i % hop) == 0:
                window = np.stack(list(buf), axis=0)  # [64,5,7]
                label, conf = self.predict(window, tau=tau)
                pred_hist.append(label)
                conf_hist.append(conf)
                # votación
                voted = max(set(pred_hist), key=pred_hist.count)
                yield {
                    "timestamp": i,
                    "label": label,
                    "conf": conf,
                    "voted_label": voted,
                    "vote_window": list(pred_hist),
                    "vote_conf_mean": float(np.mean(conf_hist)) if conf_hist else None
                }
            i += 1

# -----------------------------
# Helpers CLI
# -----------------------------
def load_window_from_csv(path: Path) -> np.ndarray:
    """
    Carga una ventana desde CSV tipo *_plot.csv con columnas:
      sample(0..63), sensor(0..4), ax,ay,az,w,i,j,k
    Retorna [64,5,7]
    """
    import pandas as pd
    # Si la ruta no es absoluta, buscar en ../gesto-...
    if not path.is_absolute():
        base = Path(__file__).parent.parent  # rust/
        path = base / path
    df = pd.read_csv(path)
    req = {"sample","sensor","ax","ay","az","w","i","j","k"}
    if not req.issubset(df.columns):
        raise ValueError(f"CSV {path} no contiene columnas requeridas {req}")
    # validar sensores y samples
    samples = sorted(df["sample"].unique().tolist())
    sensors = sorted(df["sensor"].unique().tolist())
    if samples != list(range(WINDOW_T)) or sensors != list(range(SENSORS)):
        raise ValueError("CSV no tiene las 64 muestras y 5 sensores esperados.")
    ten = np.zeros((WINDOW_T, SENSORS, CHANNELS), dtype=np.float32)
    for s in range(SENSORS):
        sdf = df[df.sensor == s].sort_values("sample")
        ten[:, s, 0:3] = sdf[["ax","ay","az"]].values
        ten[:, s, 3:7] = sdf[["w","i","j","k"]].values
    return ten

def load_batch_from_npz(path: Path) -> List[np.ndarray]:
    """
    Carga un .npz con múltiples ventanas bajo clave 'windows' shape [N,64,5,7]
    """
    data = np.load(path)
    windows = data["windows"]
    if windows.ndim != 4 or windows.shape[1:] != (WINDOW_T,SENSORS,CHANNELS):
        raise ValueError(f"npz debe contener 'windows' con shape [N,64,5,7], recibido {windows.shape}")
    return [w for w in windows]

def main():
    ap = argparse.ArgumentParser(description="Inferencia de gestos desde ventanas IMU (64x5x7).")
    ap.add_argument("--artifacts", type=str, default=DEFAULT_ARTIFACTS, help="Carpeta con .joblib y labels.json")
    ap.add_argument("--pipeline", type=str, default=None, help="Nombre del pipeline .joblib (opcional)")
    ap.add_argument("--csv", type=str, default=None, help="Ruta a CSV *_plot.csv de UNA ventana")
    ap.add_argument("--npz", type=str, default=None, help="Ruta a .npz con 'windows' [N,64,5,7]")
    ap.add_argument("--tau", type=float, default=None, help="Umbral de rechazo si hay clasificador calibrado")
    args = ap.parse_args()

    clf = GestureClassifier(
        artifacts_dir=Path(args.artifacts),
        pipeline_filename=args.pipeline,
        try_calibrated=True
    )

    if args.csv:
        ten = load_window_from_csv(Path(args.csv))
        label, conf = clf.predict(ten, tau=args.tau)
        print(json.dumps({"file": args.csv, "label": label, "confidence": conf}, ensure_ascii=False))
        return

    if args.npz:
        outs = []
        for ten in load_batch_from_npz(Path(args.npz)):
            label, conf = clf.predict(ten, tau=args.tau)
            outs.append({"label": label, "confidence": conf})
            print(f"{label}\n", end="")
        print(json.dumps({"count": len(outs), "predictions": outs}, ensure_ascii=False))
        return

    print("Nada que inferir. Usa --csv o --npz. Ejemplos:\n"
          "  python gesture_infer.py --artifacts /content/drive/MyDrive/Quiroscopio_TrainingFinal "
          "--csv /content/drive/MyDrive/Dataset/gestosS1/gesto-zoom-in/000012_plot.csv\n"
          "  python gesture_infer.py --artifacts /... --npz ventanas.npz")

if __name__ == "__main__":
    main()