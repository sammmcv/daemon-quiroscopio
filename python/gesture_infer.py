# -*- coding: utf-8 -*-
# gesture_infer.py — usa pipeline time+fft (joblib) con ventana [64,5,7]
# Artefactos esperados en el mismo directorio:
#   - best_pipeline__time+fft__svm_poly.joblib
#   - classes.json  {"index_to_class": {"0":"...", ...}}

import json, math
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from joblib import load

# ====== parámetros del stream ======
WINDOW_T = 64
SENSORS  = 5
CHANNELS = 7   # [ax,ay,az,w,i,j,k] en ese orden

# ====== features idénticas al entrenamiento ======
from scipy.fft import rfft, rfftfreq
from scipy.stats import iqr, entropy

FS = 64
BANDS = [(0.5,2),(2,4),(4,8),(8,12),(12,20)]

def _bandpower(psd, freqs, fmin, fmax):
    m = (freqs>=fmin) & (freqs<fmax)
    if not np.any(m): return 0.0
    return np.trapz(psd[m], freqs[m])

def features_from_tensor(T: np.ndarray) -> np.ndarray:
    """T: [5,64,7] -> vector de features time+fft.
    Ajustado como en la referencia en C: FFT con ventana de Hamming y +5 features finales.
    """
    feats = []
    for s in range(5):
        ax, ay, az, w, i_, j_, k_ = [T[s, :, k] for k in range(7)]
        # --- Temporal (aceleros)
        for sig in (ax, ay, az):
            feats += [
                float(np.mean(sig)), float(np.std(sig)), float(np.median(sig)),
                float(iqr(sig)), float(np.max(sig) - np.min(sig)),
                float(np.sqrt(np.mean(sig**2))),
                float(np.mean(np.abs(np.diff(sig))))
            ]
        # --- Cuaterniones: delta
        dq = np.sqrt(np.diff(w)**2 + np.diff(i_)**2 + np.diff(j_)**2 + np.diff(k_)**2)
        feats += [float(np.mean(dq)), float(np.std(dq)), float(np.max(dq)), float(np.median(dq))]
        # --- FFT (aceleros) con Hamming y corrección de potencia
        for sig in (ax, ay, az):
            sig0 = sig - np.mean(sig)
            if sig0.size > 1:
                n = sig0.size
                n_arr = np.arange(n, dtype=np.float32)
                win = 0.54 - 0.46 * np.cos(2.0 * np.pi * n_arr / (n - 1))
                X = rfft(sig0 * win)
                w2 = float(np.mean(win.astype(np.float64)**2)) if n > 0 else 1.0
                denom = n * (w2 if w2 > 1e-12 else 1.0)
                psd = (np.abs(X)**2) / denom
                freqs = rfftfreq(n, d=1.0 / FS)
            else:
                freqs = np.array([0.0], dtype=np.float32)
                psd = np.array([0.0], dtype=np.float32)
            if psd.size > 1:
                dom_idx = int(np.argmax(psd[1:]) + 1)
            else:
                dom_idx = 0
            dom_f = float(freqs[dom_idx]) if dom_idx < freqs.size else 0.0
            ps_sum = float(np.sum(psd))
            centroid = float(np.sum(freqs * psd) / ps_sum) if ps_sum > 0 else 0.0
            pnorm = psd / ps_sum if ps_sum > 0 else np.zeros_like(psd)
            ent = float(entropy(pnorm + 1e-12, base=2))
            feats += [dom_f, centroid, ent]
            for (f1, f2) in BANDS:
                feats += [float(_bandpower(psd, freqs, f1, f2))]

    # === +5 features adicionales (sensor 0) ===
    ax0 = T[0, :, 0]
    ay0 = T[0, :, 1]
    az0 = T[0, :, 2]
    sum_ax = float(np.sum(ax0))
    sum_ay = float(np.sum(ay0))
    sum_az = float(np.sum(az0))
    delta_ax = float(ax0[-1] - ax0[0])
    delta_ay = float(ay0[-1] - ay0[0])
    feats += [sum_ax, sum_ay, sum_az, delta_ax, delta_ay]

    return np.array(feats, dtype=np.float32)

# --- Umbrales anti-falsos ---
MOTION_TAU = 0.040   # Reducido: el filtrado en Rust ya maneja la detección
CONF_TAU   = 0.70    # confianza mínima del clasificador (0..1)

# --- Medida rápida de movimiento ---
def motion_score(window_64x5x7: np.ndarray) -> float:
    """
    Medida simple de actividad: media del |Δ| en aceleraciones + delta de cuaterniones.
    Escala ~0 en quieto; sube cuando hay gesto.
    """
    a = np.asarray(window_64x5x7, dtype=np.float32)  # [64,5,7]
    accel = a[:, :, 0:3]                  # ax,ay,az
    q     = a[:, :, 3:7]                  # w,i,j,k

    # |Δ| temporal en aceleros
    dacc  = np.abs(np.diff(accel, axis=0))          # [63,5,3]
    s_acc = float(np.mean(dacc))

    # Δ euclídeo en cuaterniones
    dq    = np.sqrt(np.sum(np.diff(q, axis=0)**2, axis=2))  # [63,5]
    s_q   = float(np.mean(dq))

    # mezcla (ajusta ponderaciones si quieres)
    return 0.7 * s_acc + 0.3 * s_q

class GestureClassifier:
    """
    Interfaz mantenida: predict(window_64x5x7) -> (label:str, conf:float)
    """
    def __init__(
        self,
        artifacts_dir: Optional[str|Path]=None,
        pipeline_filename: Optional[str]=None,
        try_calibrated: bool=True,   # <- NUEVO: se acepta pero no se usa
        **kwargs                     # <- por si llegan más kwargs desde Rust
    ):
        adir = Path(artifacts_dir or Path(__file__).parent)
        if pipeline_filename:
            model_path = adir / pipeline_filename
        else:
            cands = sorted(adir.glob("best_pipeline__time+fft__*.joblib"))
            if not cands:
                raise FileNotFoundError("No se encontró best_pipeline__time+fft__*.joblib en artifacts_dir")
            model_path = cands[0]
        self.pipeline = load(model_path)

        mapping = json.loads((adir / "classes.json").read_text(encoding="utf-8"))["index_to_class"]
        self.idx2label = [mapping[str(i)] for i in range(len(mapping))]

    def _pre(self, window_64x5x7: np.ndarray) -> np.ndarray:
        a = np.asarray(window_64x5x7, dtype=np.float32)
        if a.shape != (WINDOW_T, SENSORS, CHANNELS):
            raise ValueError(f"Esperado (64,5,7), recibido {a.shape}")
        # el entrenamiento usa [5,64,7]
        T = np.transpose(a, (1,0,2))
        x = features_from_tensor(T).reshape(1, -1)
        return x

    def predict(self, window_64x5x7: np.ndarray) -> Tuple[str, float]:
        # 1) Gate por “hay movimiento”
        ms = motion_score(window_64x5x7)
        if ms < MOTION_TAU:
            return "desconocido", 0.0

        # 2) Extraer features y predecir
        X = self._pre(window_64x5x7)
        y_idx = int(self.pipeline.predict(X)[0])
        label = self.idx2label[y_idx]

        # 3) Confianza (decision_function → sigmoide)
        conf = 1.0
        try:
            df = self.pipeline.decision_function(X)
            m = float(np.max(np.atleast_1d(df)))
            conf = 1.0 / (1.0 + math.exp(-m))
        except Exception:
            pass

        # 4) Gate por confianza mínima
        if conf < CONF_TAU:
            return "desconocido", conf

        return label, conf
    
    def predict_all_scores(self, window_64x5x7: np.ndarray) -> Tuple[str, float, dict]:
        """
        Predice el gesto y devuelve scores para TODAS las clases.
        Retorna: (label, conf, all_scores) donde all_scores = {"gesto-drop": 0.95, ...}
        """
        # 1) Gate por "hay movimiento"
        ms = motion_score(window_64x5x7)
        if ms < MOTION_TAU:
            return "desconocido", 0.0, {}

        # 2) Extraer features
        X = self._pre(window_64x5x7)
        
        # 3) Predecir clase principal
        y_idx = int(self.pipeline.predict(X)[0])
        label = self.idx2label[y_idx]

        # 4) Obtener scores para todas las clases
        all_scores = {}
        conf = 1.0
        
        try:
            # Para SVM multiclase (OvR), decision_function devuelve scores por clase
            df = self.pipeline.decision_function(X)[0]  # shape: (n_classes,)
            
            # Convertir scores a pseudo-probabilidades con softmax (para mostrar distribución)
            exp_scores = np.exp(df - np.max(df))  # estabilidad numérica
            probs = exp_scores / np.sum(exp_scores)
            
            for i, score in enumerate(probs):
                all_scores[self.idx2label[i]] = float(score)
            
            # Confianza del gesto predicho usando la misma lógica que predict()
            # (sigmoide del score máximo, igual que antes)
            m = float(np.max(np.atleast_1d(df)))
            conf = 1.0 / (1.0 + math.exp(-m))
        except Exception:
            # Fallback si decision_function no está disponible
            conf = 1.0
            all_scores = {label: conf}

        # 5) Gate por confianza mínima
        if conf < CONF_TAU:
            return "desconocido", conf, all_scores

        return label, conf, all_scores

# Helpers CLI para probar con CSV/NPZ
def load_window_from_csv(path: Path) -> np.ndarray:
    import pandas as pd
    if not path.is_absolute():
        base = Path(__file__).parent.parent
        path = base / path
    df = pd.read_csv(path)
    req = {"sample","sensor","ax","ay","az","w","i","j","k"}
    if not req.issubset(df.columns):
        raise ValueError(f"CSV {path} no contiene columnas requeridas {req}")
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

def load_batch_from_npz(path: Path):
    data = np.load(path)
    windows = data["windows"]
    if windows.ndim != 4 or windows.shape[1:] != (WINDOW_T,SENSORS,CHANNELS):
        raise ValueError(f"npz debe contener 'windows' con shape [N,64,5,7], recibido {windows.shape}")
    return [w for w in windows]

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Inferencia de gestos desde ventanas IMU (64x5x7) usando time+fft.")
    ap.add_argument("--artifacts", type=str, default=None, help="Carpeta con .joblib y classes.json")
    ap.add_argument("--pipeline", type=str, default=None, help="Nombre del pipeline .joblib (opcional)")
    ap.add_argument("--csv", type=str, default=None, help="Ruta a CSV *_plot.csv de UNA ventana")
    ap.add_argument("--npz", type=str, default=None, help="Ruta a .npz con 'windows' [N,64,5,7]")
    args = ap.parse_args()

    clf = GestureClassifier(
        artifacts_dir=args.artifacts,
        pipeline_filename=args.pipeline
    )

    if args.csv:
        ten = load_window_from_csv(Path(args.csv))
        label, conf = clf.predict(ten)
        print(json.dumps({"file": args.csv, "label": label, "confidence": conf}, ensure_ascii=False))
        return

    if args.npz:
        outs = []
        for ten in load_batch_from_npz(Path(args.npz)):
            label, conf = clf.predict(ten)
            outs.append({"label": label, "confidence": conf})
            print(f"{label}\n", end="")
        print(json.dumps({"count": len(outs), "predictions": outs}, ensure_ascii=False))
        return

    print("Nada que inferir. Usa --csv o --npz. Ejemplos:\n"
          "  python gesture_infer.py --artifacts . --csv gesto-drop/000001_plot.csv\n"
          "  python gesture_infer.py --artifacts . --npz ventanas.npz")

if __name__ == "__main__":
    main()