import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os, random, numpy as np, torch

seed = 11
os.environ["PYTHONHASHSEED"] = str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)


class LSTMcombineTwoFeatureLnY(nn.Module):
    """
    Learns a single latent volatility factor from (bsIV, MFIV)
    using an LSTM. Designed for direct feeding into MS-GARCH.
    """

    def __init__(self, seq_len=20, hidden_dim=32, lr=1e-3, epochs=30):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs

        self.lstm = nn.LSTM(
            input_size=2,
            hidden_size=hidden_dim,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])  # last timestep

    # -------------------------
    # Fit model
    # -------------------------
    def fit(self, X: pd.DataFrame):
        X = X.copy().dropna()

        feats = X[["bsIV", "MFIV"]].to_numpy(dtype=np.float32)
        rt = X["rt"].to_numpy(dtype=np.float32)

        X_seq, y = [], []
        for i in range(self.seq_len, len(X)):
            X_seq.append(feats[i - self.seq_len:i])
            # y.append(rt[i] ** 2)  # supervise on realized variance
            y.append(np.log(rt[i]**2 + 1e-12)) # optimizing on log once 


        X_seq = torch.tensor(np.stack(X_seq))
        y = torch.tensor(y).view(-1, 1)

        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        self.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            y_hat = self(X_seq)
            loss = loss_fn(y_hat, y)
            loss.backward()
            opt.step()

        return self

    # -------------------------
    # Transform X → xx
    # -------------------------
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy().dropna()

        feats = X[["bsIV", "MFIV"]].to_numpy(dtype=np.float32)
        rt = X["rt"].to_numpy(dtype=np.float32)
        idx = X.index

        X_seq = []
        for i in range(self.seq_len, len(X)):
            X_seq.append(feats[i - self.seq_len:i])

        X_seq = torch.tensor(np.stack(X_seq))

        self.eval()
        with torch.no_grad():
            iv_latent = self(X_seq).numpy().flatten()

        xx = pd.DataFrame(
            {
                "rt": rt[self.seq_len:],
                "iv_lstm": iv_latent
            },
            index=idx[self.seq_len:]
        )

        return xx


# 4) LSTM correction model (input_size=1)
class LSTMcorrNoIV(nn.Module):
    def __init__(self, hidden_dim=32, lr=1e-3, epochs=30):
        super().__init__()
        self.lr = lr
        self.epochs = epochs
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_dim, batch_first=True)
        self.fc   = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

    def fit_xy(self, X_seq, y):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        self.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            loss = loss_fn(self(X_seq), y)
            loss.backward()
            opt.step()
        return self

    def predict(self, X_seq):
        self.eval()
        with torch.no_grad():
            return self(X_seq).numpy().flatten()

# 4) LSTM model (unchanged)
class LSTMcombineTwoFeature(nn.Module):
    def __init__(self, hidden_dim=32, lr=1e-3, epochs=30):
        super().__init__()
        self.lr = lr
        self.epochs = epochs
        self.lstm = nn.LSTM(input_size=2, hidden_size=hidden_dim, batch_first=True)
        self.fc   = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

    def fit_xy(self, X_seq, y):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        self.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            loss = loss_fn(self(X_seq), y)
            loss.backward()
            opt.step()
        return self

    def predict(self, X_seq):
        self.eval()
        with torch.no_grad():
            return self(X_seq).numpy().flatten()


# put this in src/LSTM_lib.py (or in a notebook cell)

import numpy as np
import torch
import torch.nn as nn
import numpy as np
import torch
import torch.nn as nn


class LSTMGatedIVCorrection(nn.Module):
    """
    ADDITIVE gated blend in LEVELS:
        h_hat_t = (1 - w_t) * h_base_t + w_t * h_iv_t

    To keep your caller code almost unchanged, predict() returns g_hat such that:
        h_final = h_base2 * np.exp(g_hat)
    matches the additive blend exactly by setting:
        g_hat_t = log( h_hat_t / h_base_t )
    """

    def __init__(self, hidden_dim=32, lr=1e-3, epochs=30, batch_size=256, seed=42, device=None):
        super().__init__()
        torch.manual_seed(seed)
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # input: [bsIV, MFIV] -> LSTM -> gate w_t
        self.lstm = nn.LSTM(input_size=2, hidden_size=hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, 1)

        self.to(self.device)
        self.last_w = None

    @staticmethod
    def _to_tensor(x, device):
        if isinstance(x, torch.Tensor):
            return x.to(device)
        return torch.tensor(x, dtype=torch.float32, device=device)

    def forward(self, X_seq):
        out, _ = self.lstm(X_seq)          # (B, T, H)
        hT = out[:, -1, :]                 # (B, H)
        w = torch.sigmoid(self.head(hT))   # (B, 1) in [0,1]
        return w

    def fit_xy(self, X_seq, rv, h_base2, h_iv2):
        """
        X_seq:   (N, SEQ_LEN, 2)
        rv:      (N, 1) or (N,) realized variance target in LEVELS (NOT logs)
        h_base2: (N,)    base variance aligned to the same N rows
        h_iv2:   (N,)    IV variance proxy aligned to the same N rows
        """
        X_seq = self._to_tensor(X_seq, self.device)

        rv = np.asarray(rv, float).reshape(-1)
        hb = np.asarray(h_base2, float).reshape(-1)
        hi = np.asarray(h_iv2,   float).reshape(-1)

        eps = 1e-12
        rv_t = self._to_tensor(rv.reshape(-1, 1), self.device)
        hb_t = self._to_tensor(np.maximum(hb, eps).reshape(-1, 1), self.device)
        hi_t = self._to_tensor(np.maximum(hi, eps).reshape(-1, 1), self.device)

        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        N = X_seq.shape[0]
        for _ in range(self.epochs):
            perm = torch.randperm(N, device=self.device)
            for s in range(0, N, self.batch_size):
                ii = perm[s:s+self.batch_size]
                Xb, rvb, hbb, hib = X_seq[ii], rv_t[ii], hb_t[ii], hi_t[ii]

                w = self.forward(Xb)
                h_hat = (1.0 - w) * hbb + w * hib   # ADDITIVE blend in levels

                loss = loss_fn(h_hat, rvb)
                opt.zero_grad()
                loss.backward()
                opt.step()

        return self

    @torch.no_grad()
    def predict(self, X_seq, h_base2, h_iv2):
        """
        Returns g_hat so your existing caller line can stay:
            h_final = h_base2 * np.exp(g_hat)

        Internally we compute:
            h_hat = (1-w)*h_base + w*h_iv
            g_hat = log(h_hat / h_base)
        """
        X_seq = self._to_tensor(X_seq, self.device)

        hb = np.asarray(h_base2, float).reshape(-1)
        hi = np.asarray(h_iv2,   float).reshape(-1)

        eps = 1e-12
        hb_t = self._to_tensor(np.maximum(hb, eps).reshape(-1, 1), self.device)
        hi_t = self._to_tensor(np.maximum(hi, eps).reshape(-1, 1), self.device)

        w = self.forward(X_seq)
        self.last_w = w.detach().cpu().numpy()

        h_hat = (1.0 - w) * hb_t + w * hi_t
        g_hat = torch.log(torch.clamp(h_hat / hb_t, min=eps))

        return g_hat.detach().cpu().numpy()


# class LSTMGatedIVCorrection(nn.Module):
#     """
#     Gated blend: model learns a weight w_t in [0,1] that decides how much to trust IV vs base.
#     Returns g_hat so your downstream line stays the same:
#         h_final = h_base2 * np.exp(g_hat)

#     Constraint:
#         g_hat_t = w_t * ( log(h_iv_t) - log(h_base_t) )
#     so the correction is bounded by the IV-vs-base log gap.
#     """

#     def __init__(self, hidden_dim=32, lr=1e-3, epochs=30, batch_size=256, seed=42, device=None):
#         super().__init__()
#         torch.manual_seed(seed)
#         self.hidden_dim = hidden_dim
#         self.lr = lr
#         self.epochs = epochs
#         self.batch_size = batch_size
#         self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

#         # input: your seq features (e.g., [bsIV, MFIV]) -> hidden -> scalar gate logit
#         self.lstm = nn.LSTM(input_size=2, hidden_size=hidden_dim, batch_first=True)
#         self.head = nn.Linear(hidden_dim, 1)

#         self.to(self.device)
#         self.last_w = None

#     @staticmethod
#     def _to_tensor(x, device):
#         if isinstance(x, torch.Tensor):
#             return x.to(device)
#         return torch.tensor(x, dtype=torch.float32, device=device)

#     def forward(self, X_seq):
#         out, _ = self.lstm(X_seq)          # (B, T, H)
#         hT = out[:, -1, :]                 # (B, H)
#         w = torch.sigmoid(self.head(hT))   # (B, 1) in [0,1]
#         return w

#     def fit_xy(self, X_seq, y, h_base2, h_iv2):
#         """
#         X_seq: (N, SEQ_LEN, 2)
#         y:     (N, 1)   target g = log(rv) - log(h_base)
#         h_base2: (N,)   aligned base variance for same N rows
#         h_iv2:   (N,)   aligned IV variance proxy for same N rows
#         """
#         X_seq = self._to_tensor(X_seq, self.device)
#         y     = self._to_tensor(y,     self.device)

#         hb = np.asarray(h_base2, float).reshape(-1)
#         hi = np.asarray(h_iv2,   float).reshape(-1)

#         eps = 1e-12
#         log_gap = np.log(np.maximum(hi, eps)) - np.log(np.maximum(hb, eps))  # (N,)
#         log_gap = self._to_tensor(log_gap.reshape(-1, 1), self.device)       # (N,1)

#         opt = torch.optim.Adam(self.parameters(), lr=self.lr)
#         loss_fn = nn.MSELoss()

#         N = X_seq.shape[0]
#         for _ in range(self.epochs):
#             perm = torch.randperm(N, device=self.device)
#             for s in range(0, N, self.batch_size):
#                 ii = perm[s:s+self.batch_size]
#                 Xb, yb, gb = X_seq[ii], y[ii], log_gap[ii]

#                 w = self.forward(Xb)
#                 g_hat = w * gb

#                 loss = loss_fn(g_hat, yb)
#                 opt.zero_grad()
#                 loss.backward()
#                 opt.step()

#         return self

#     @torch.no_grad()
#     def predict(self, X_seq, h_base2, h_iv2):
#         """
#         Returns g_hat (N,1) so you can keep:
#             h_final = h_base2 * np.exp(g_hat)
#         """
#         X_seq = self._to_tensor(X_seq, self.device)

#         hb = np.asarray(h_base2, float).reshape(-1)
#         hi = np.asarray(h_iv2,   float).reshape(-1)

#         eps = 1e-12
#         log_gap = np.log(np.maximum(hi, eps)) - np.log(np.maximum(hb, eps))
#         log_gap_t = self._to_tensor(log_gap.reshape(-1, 1), self.device)

#         w = self.forward(X_seq)
#         self.last_w = w.detach().cpu().numpy()

#         g_hat = (w * log_gap_t).detach().cpu().numpy()
#         return g_hat
