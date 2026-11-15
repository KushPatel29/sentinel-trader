"""Calibrated tree and sequence challengers with strictly chronological fitting.

TFT uses the actual pytorch-forecasting implementation, with a binary direction
head. Decoder inputs contain known calendar variables only, never future prices.
Heavy dependencies are imported only inside neural training/inference functions.
"""

from __future__ import annotations
from dataclasses import dataclass
import copy

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

FEATURES = [
    "ret_1",
    "ret_3",
    "ret_6",
    "ma_distance_5d",
    "ma_distance_20d",
    "ma_distance_60d",
    "vol_1d",
    "vol_20d",
    "atr",
    "range_z",
    "body",
    "wick_skew",
    "rsi_14",
    "breakout",
    "drawdown",
    "vol_regime",
    "gap_hours",
    "flow_volume_ratio",
    "flow_imbalance",
    "flow_buy_share",
]
CALENDAR = ["hour_sin", "hour_cos", "weekday_sin", "weekday_cos"]


def split_positions(f, y, ends, boundary, cfg):
    cal = boundary - pd.Timedelta(days=cfg["calibration_days"])
    early = cal - pd.Timedelta(days=cfg["early_stop_days"])
    valid = y.notna() & ends.notna() & (np.arange(len(f)) >= cfg["lookback"] - 1)
    train = valid & (f.index >= boundary - pd.Timedelta(days=cfg["train_days"])) & (ends < early)
    stop = valid & (f.index >= early) & (ends < cal)
    calibration = valid & (f.index >= cal) & (ends < boundary)
    positions = [np.flatnonzero(m) for m in (train, stop, calibration)]
    if min(map(len, positions)) < 25 or len(positions[0]) < 128:
        raise ValueError("Insufficient purged train/early-stop/calibration samples")
    return positions


def preprocessing(f, train):
    columns = [c for c in FEATURES if c in f and f[c].iloc[train].notna().any()]
    medians = f[columns].iloc[train].median().fillna(0)
    filled = f[columns].fillna(medians)
    scaler = StandardScaler().fit(filled.iloc[train])
    return columns, medians, scaler


def arrays(f, columns, medians, scaler):
    market = np.clip(scaler.transform(f[columns].fillna(medians)), -8, 8).astype("float32")
    calendar = f[CALENDAR].fillna(0).to_numpy(dtype="float32")
    return np.concatenate([market, calendar], axis=1)


def sequence_batch(values, positions, lookback):
    # Each sequence ends at its own decision bar, not one step beyond it.
    offsets = np.arange(-lookback + 1, 1)
    return values[np.asarray(positions)[:, None] + offsets]


def network(name, inputs, cfg):
    from torch import nn

    hidden = cfg["hidden_size"]
    if name == "lstm":
        return nn.Sequential() if inputs == 0 else LSTMFactory.build(inputs, hidden)
    if name == "tft":
        from pytorch_forecasting import TemporalFusionTransformer
        from pytorch_forecasting.data import NaNLabelEncoder
        from pytorch_forecasting.metrics import CrossEntropy

        names = [f"v{i}" for i in range(inputs)]
        return TemporalFusionTransformer(
            hidden_size=hidden,
            hidden_continuous_size=4,
            attention_head_size=2,
            lstm_layers=1,
            dropout=0.15,
            output_size=2,
            loss=CrossEntropy(),
            max_encoder_length=cfg["lookback"],
            x_reals=names,
            time_varying_reals_encoder=names,
            time_varying_reals_decoder=names[-4:],
            learning_rate=cfg["learning_rate"],
            logging_metrics=nn.ModuleList(),
            causal_attention=True,
            output_transformer=NaNLabelEncoder().fit(np.array([0, 1])),
        )
    raise ValueError(name)


class LSTMFactory:
    @staticmethod
    def build(inputs, hidden):
        from torch import nn

        class DirectionLSTM(nn.Module):
            def __init__(self):
                super().__init__()
                self.recurrent = nn.LSTM(inputs, hidden, num_layers=2, dropout=0.15, batch_first=True)
                self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Dropout(0.15), nn.Linear(hidden, 2))

            def forward(self, x):
                return self.head(self.recurrent(x)[0][:, -1])

        return DirectionLSTM()


def forward(net, name, x):
    if name == "lstm":
        return net(x)
    import torch

    n, length, features = x.shape
    decoder = torch.zeros((n, 1, features), dtype=x.dtype, device=x.device)
    # These are decision-time calendar fields, all known at the forecast origin.
    decoder[:, 0, -4:] = x[:, -1, -4:]
    z = {
        "encoder_cont": x,
        "decoder_cont": decoder,
        "encoder_cat": torch.empty(n, length, 0, dtype=torch.long, device=x.device),
        "decoder_cat": torch.empty(n, 1, 0, dtype=torch.long, device=x.device),
        "encoder_lengths": torch.full((n,), length, dtype=torch.long, device=x.device),
        "decoder_lengths": torch.ones(n, dtype=torch.long, device=x.device),
        "target_scale": torch.ones(n, 2, device=x.device),
    }
    return net(z)["prediction"][:, 0, :]


def raw_logits(net, name, values, positions, cfg):
    import torch

    net.eval()
    output = []
    with torch.no_grad():
        for begin in range(0, len(positions), cfg["batch_size"]):
            ids = positions[begin : begin + cfg["batch_size"]]
            x = torch.from_numpy(sequence_batch(values, ids, cfg["lookback"]))
            logits = forward(net, name, x)
            output.extend((logits[:, 1] - logits[:, 0]).numpy().tolist())
    return np.asarray(output)


@dataclass
class Challenger:
    name: str
    columns: list
    medians: object
    scaler: object
    model: object
    calibrator: object
    prior: float
    cfg: dict
    audit: dict
    _net: object = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_net"] = None
        return state

    def predict(self, f, positions=None):
        positions = np.asarray(positions if positions is not None else [len(f) - 1])
        values = arrays(f, self.columns, self.medians, self.scaler)
        if self.name in ("lstm", "tft"):
            import torch

            torch.set_num_threads(self.cfg["torch_threads"])
            if self._net is None:
                self._net = network(self.name, values.shape[1], self.cfg)
                self._net.load_state_dict(self.model)
            raw = raw_logits(self._net, self.name, values, positions, self.cfg)
        else:
            with threadpool_limits(limits=1):
                p = self.model.predict_proba(values[positions])[:, 1].clip(1e-5, 1 - 1e-5)
            raw = np.log(p / (1 - p))
        return self.calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]


def fit_challenger(name, f, y, ends, boundary, cfg):
    train, stop, cal = split_positions(f, y, ends, boundary, cfg)
    columns, medians, scaler = preprocessing(f, train)
    values = arrays(f, columns, medians, scaler)
    labels = (y > 0).astype("int64").to_numpy()
    audit = {
        "boundary": str(boundary),
        "train_samples": len(train),
        "stop_samples": len(stop),
        "calibration_samples": len(cal),
        "train_last_label_end": str(ends.iloc[train].max()),
        "stop_last_label_end": str(ends.iloc[stop].max()),
        "calibration_last_label_end": str(ends.iloc[cal].max()),
    }
    if min(len(np.unique(labels[p])) for p in (train, stop, cal)) < 2:
        raise ValueError("A chronological segment lacks both direction classes")
    if name in ("lstm", "tft"):
        import torch

        torch.set_num_threads(cfg["torch_threads"])
        torch.manual_seed(cfg["seed"])
        rng = np.random.default_rng(cfg["seed"])
        net = network(name, values.shape[1], cfg)
        optimizer = torch.optim.AdamW(net.parameters(), lr=cfg["learning_rate"], weight_decay=0.01)
        loss_fn = torch.nn.CrossEntropyLoss()
        # Uniform chronological subsampling controls CPU cost, never selects by return.
        if len(train) > cfg["max_training_sequences"]:
            train = train[np.linspace(0, len(train) - 1, cfg["max_training_sequences"]).astype(int)]
        best, state, bad, history = float("inf"), None, 0, []
        for epoch in range(cfg["max_epochs"]):
            net.train()
            order = rng.permutation(train)
            losses = []
            for begin in range(0, len(order), cfg["batch_size"]):
                ids = order[begin : begin + cfg["batch_size"]]
                x = torch.from_numpy(sequence_batch(values, ids, cfg["lookback"]))
                loss = loss_fn(forward(net, name, x), torch.from_numpy(labels[ids]))
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite neural training loss")
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.detach()))
            raw = raw_logits(net, name, values, stop, cfg)
            validation = float(np.mean(np.logaddexp(0, raw) - labels[stop] * raw))
            history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "stop_loss": validation})
            if validation < best - 1e-4:
                best, state, bad = validation, copy.deepcopy(net.state_dict()), 0
            else:
                bad += 1
            if bad >= cfg["patience"]:
                break
        net.load_state_dict(state)
        raw = raw_logits(net, name, values, cal, cfg)
        model = {k: v.detach().cpu() for k, v in state.items()}
        audit.update(
            {
                "epochs": history,
                "parameters": sum(p.numel() for p in net.parameters()),
                "training_sequences_used": len(train),
                "implementation": "pytorch_forecasting.TemporalFusionTransformer" if name == "tft" else "torch.nn.LSTM",
            }
        )
    else:
        if name.startswith("random_forest"):
            model = RandomForestClassifier(
                n_estimators=180,
                max_depth=8 if name.endswith("deep") else 4,
                min_samples_leaf=40,
                max_features=0.7,
                n_jobs=1,
                random_state=cfg["seed"],
            )
        elif name == "extra_trees":
            model = ExtraTreesClassifier(
                n_estimators=180, max_depth=5, min_samples_leaf=40, max_features=0.7, n_jobs=1, random_state=cfg["seed"]
            )
        elif name.startswith("lightgbm"):
            from lightgbm import LGBMClassifier

            model = LGBMClassifier(
                n_estimators=160,
                learning_rate=0.025,
                num_leaves=7 if name.endswith("small") else 15,
                max_depth=-1,
                min_child_samples=80,
                reg_lambda=20,
                colsample_bytree=0.8,
                verbosity=-1,
                deterministic=True,
                force_col_wise=True,
                n_jobs=1,
                random_state=cfg["seed"],
            )
        elif name == "logistic":
            model = LogisticRegression(C=0.02, max_iter=500, random_state=cfg["seed"])
        else:
            raise ValueError(name)
        with threadpool_limits(limits=1):
            model.fit(values[train], labels[train])
            p = model.predict_proba(values[cal])[:, 1].clip(1e-5, 1 - 1e-5)
        raw = np.log(p / (1 - p))
    calibrator = LogisticRegression(C=1.0, random_state=cfg["seed"]).fit(raw.reshape(-1, 1), labels[cal])
    return Challenger(
        name, columns, medians, scaler, model, calibrator, float(labels[np.r_[train, stop, cal]].mean()), cfg, audit
    )


def hybrid_weights(probabilities, actual, cfg):
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(actual, dtype=float)
    k = p.shape[1]
    uniform = np.full(k, 1 / k)

    def objective(w):
        return np.mean((p @ w - y) ** 2) + cfg["hybrid_ridge"] * np.sum((w - uniform) ** 2)

    result = minimize(
        objective,
        uniform,
        bounds=[(0, 1)] * k,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
        method="SLSQP",
    )
    if not result.success or not np.isfinite(result.x).all():
        raise ValueError("Hybrid weight fit failed")
    return result.x / result.x.sum()
