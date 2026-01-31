import numpy as np
import numpy as np


class lnMsGARCHIV:
    """
    Same interface as msGARCHIV, but IV enters multiplicatively:
        h_t = base_t * exp(delta * iv_t)
    so h_t stays > 0 by construction.
    """

    def __init__(self):
        self.R = 3

        self.P = np.array([[0.90, 0.05, 0.05],
                           [0.10, 0.80, 0.10],
                           [0.05, 0.10, 0.85]])

        self.pi_0 = np.full(self.R, 1 / self.R)

        self.params = {
            0: [0.01, 0.05, 0.90],
            1: [0.02, 0.10, 0.80],
            2: [0.03, 0.07, 0.85]
        }

        self.theta0 = self._build_theta0(self.R)

    def _build_theta0(self, R):
        def _lgit(x): return np.log(x / (1 - x))
        theta0 = np.r_[np.log(0.01) * np.ones(R),
                       _lgit(0.05) * np.ones(R),
                       _lgit(0.90) * np.ones(R),
                       np.zeros(R),
                       np.zeros(R * R),
                       np.log(0.1),
                       0.0,
                       np.log(0.02),
                       _lgit(0.4),
                       0.0
                       ]
        return theta0

    def sigmoid(self, x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def row_softmax(self, M):
        M = M - M.max(axis=1, keepdims=True)
        E = np.exp(M)
        return E / E.sum(axis=1, keepdims=True)

    def softplus(self, z):
        return np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0.0)

    def unpack_theta(self, theta, R=3):
        k = 4 * R
        w_raw, a_raw, b_raw, g_raw = theta[:R], theta[R:2*R], theta[2*R:3*R], theta[3*R:4*R]
        P_logits = theta[k:k+R*R].reshape(R, R)
        rest = theta[k+R*R:]

        if rest.size >= 4:
            lam_raw, muJ, sJ_raw, d_raw = rest[:4]
            d2 = 1.0 / (1.0 + np.exp(-d_raw))
        elif rest.size >= 3:
            lam_raw, muJ, sJ_raw = rest[:3]
            d2 = 0.0
        else:
            lam_raw, muJ, sJ_raw = np.log(0.1), 0.0, np.log(0.02)
            d2 = 0.0

        omega = np.exp(w_raw)
        alpha = 1/(1+np.exp(-a_raw))
        beta  = 1/(1+np.exp(-b_raw))
        gamma = 1/(1+np.exp(-g_raw))

        s = alpha + beta + gamma
        mask = s >= 0.999
        mask02 = mask & np.array([True, False, True])
        if mask02.any():
            alpha[mask02] *= 0.999 / s[mask02]
            beta[mask02]  *= 0.999 / s[mask02]
            gamma[mask02] *= 0.999 / s[mask02]

        if mask[1]:
            beta[1] = beta[1] * (0.999 / s[1])

        P = np.exp(P_logits - P_logits.max(axis=1, keepdims=True))
        P = P / P.sum(axis=1, keepdims=True)

        lam     = np.minimum(self.softplus(lam_raw), 5.0)
        sigmaJ  = np.minimum(self.softplus(sJ_raw),  1.0)
        sigmaJ2 = sigmaJ * sigmaJ

        delta = rest[4] if rest.size >= 5 else 0.0
        return omega, alpha, beta, gamma, P, lam, muJ, sigmaJ2, d2, delta

    def _iv_mult(self, delta, iv_x):
        # stable exp(delta * iv_x)
        return float(np.exp(np.clip(delta * iv_x, -50.0, 50.0)))

    def nll_ms_garch(self, theta, r, iv, R=3, tol=1e-12):
        r  = np.asarray(r, float)
        iv = np.asarray(iv, float)

        omega, alpha, beta, gamma, P, lam, muJ, sigmaJ2, d2, delta = self.unpack_theta(theta, R)

        pi = np.full(R, 1.0 / R)
        h  = (omega + alpha * lam * sigmaJ2) / (1 - alpha - beta + 1e-9)
        h  = np.maximum(h, 1e-8)
        h_init_full = h.copy()
        g2 = h[2].copy() if h.size > 2 else 0.0
        ll = 0.0

        for i, x in enumerate(r):
            iv_x = float(iv[i]) if (i < len(iv) and np.isfinite(iv[i])) else 0.0
            m = self._iv_mult(delta, iv_x)

            u   = x - lam * muJ
            var = np.maximum(h + lam * sigmaJ2, 1e-12)
            dens = (1.0 / np.sqrt(2 * np.pi * var)) * np.exp(-0.5 * (u * u) / var)
            mix  = pi @ dens + tol
            ll  += np.log(mix)
            post = (pi * dens) / mix

            neg = float(u < 0.0)
            h_old = h.copy()

            base0 = omega[0] + alpha[0] * (u*u + lam*sigmaJ2) + beta[0] * h_old[0]
            base1 = omega[1] + (alpha[1] + gamma[1]*neg) * (u*u + lam*sigmaJ2) + beta[1] * h_old[1]
            h0 = np.maximum(base0, 1e-12) * m
            h1 = np.maximum(base1, 1e-12) * m

            if d2 >= 1e-4:
                if h_old.size < 3:
                    h_old = np.r_[h_old, h_init_full[2]]
                    h     = np.r_[h,     h_init_full[2]]
                    pi    = np.r_[pi,    0.0]

                base2 = omega[2] + alpha[2] * (u*u + lam*sigmaJ2) + beta[2] * ((1 - d2) * h_old[2] + d2 * g2)
                g2 = h_old[2].copy()
                h2 = np.maximum(base2, 1e-12) * m

                h = np.array([h0, h1, h2])
                P_use = P[:3, :3]
            else:
                h = np.array([h0, h1])
                P_use = P[:2, :2]

            n_active = P_use.shape[0]
            pi = post[:n_active].flatten() @ P_use

        return -ll

    def forecast_var_k(self, r, iv, theta, R=3, k=3):
        omega, alpha, beta, gamma, P, lam, muJ, sigmaJ2, d2, delta = self.unpack_theta(theta, R)
        pi = np.full(R, 1.0 / R)
        h  = (omega + alpha * lam * sigmaJ2) / (1 - alpha - beta + 1e-9)
        h  = np.maximum(h, 1e-8)

        regime = []
        g2 = h[2].copy() if len(h) > 2 else 0.0
        thresh = 1e-4

        iv_x = 0.0
        m = 1.0

        for j, x in enumerate(r):
            iv_x = float(iv[j]) if (j < len(iv) and np.isfinite(iv[j])) else 0.0
            m = self._iv_mult(delta, iv_x)

            Jt = lam * muJ
            u  = x - Jt
            neg = float(u < 0.0)
            u2 = u * u

            if d2 >= thresh:
                h_old = h.copy()
                base = omega + alpha*u2 + gamma*u2*neg + beta*h_old
                base = np.maximum(base, 1e-12)
                h = base * m

                base2 = omega[2] + alpha[2]*(u2 + lam*sigmaJ2) + beta[2]*((1 - d2)*h_old[2] + d2*g2)
                g2 = h_old[2].copy()
                h[2] = np.maximum(base2, 1e-12) * m

                pi = pi @ P
            else:
                base = omega[:2] + alpha[:2]*u2 + gamma[:2]*u2*neg + beta[:2]*h[:2]
                base = np.maximum(base, 1e-12)
                h = base * m

                P = P[:2, :2]
                pi = pi[:2] @ P

        out = []
        for _ in range(k):
            out.append(float(pi @ h))

            if d2 >= thresh and h.size == 3:
                h_old = h.copy()
                base = omega + alpha*out[-1] + beta*h_old
                base = np.maximum(base, 1e-12)
                h = base * m

                g2 = h_old[2].copy()
                base2 = omega[2] + alpha[2]*out[-1] + beta[2]*((1 - d2)*h_old[2] + d2*g2)
                h[2] = np.maximum(base2, 1e-12) * m

                pi = pi @ P
            else:
                base = omega[:2] + alpha[:2]*out[-1] + beta[:2]*h
                base = np.maximum(base, 1e-12)
                h = base * m
                pi = pi @ P[:2, :2]

            regime.append(pi.copy())

        return np.array(out), regime
####################################################################################

class msGARCHIV:
    """
    Wrapper class for your plain Markov-switching / MS-GARCH functions.

    IMPORTANT:
    - The actual function bodies are unchanged.
    - Everything is accessible via this class (attributes + methods).
    - Later you can instantiate and call methods; you asked to do that later.
    """

    def __init__(self):
        # ######### PLAIN MARKOV SWITCHING MODEL
        self.R = 3  # number of regimes

        self.P = np.array([[0.90, 0.05, 0.05],
                           [0.10, 0.80, 0.10],
                           [0.05, 0.10, 0.85]])

        self.pi_0 = np.full(self.R, 1 / self.R)

        self.params = {
            0: [0.01, 0.05, 0.90],
            1: [0.02, 0.10, 0.80],
            2: [0.03, 0.07, 0.85]
        }

        # preserve your _lgit + theta0 exactly, but make them available on the instance
        self.theta0 = self._build_theta0(self.R)

    # -------------------------
    # Helper to build theta0 (no change to your construction)
    # -------------------------
    def _build_theta0(self, R):
        def _lgit(x): return np.log(x / (1 - x))
        theta0 = np.r_[np.log(0.01) * np.ones(R),
                       _lgit(0.05) * np.ones(R),
                       _lgit(0.90) * np.ones(R),
                       np.zeros(R),
                       np.zeros(R * R),
                       np.log(0.1),
                       0.0,
                       np.log(0.02),
                       _lgit(0.4),
                       0.0          # delta initial value
                       ]
        return theta0

    # -------------------------
    # Your functions (unchanged)
    # -------------------------
    def sigmoid(self, x):
        x = np.clip(x, -500, 500)     # avoid overflow
        return 1 / (1 + np.exp(-x))

    def row_softmax(self, M):
        M = M - M.max(axis=1, keepdims=True)
        E = np.exp(M)
         # we substract the maximum in each value from all entries in that row
         # so the largest becomes 0, then we exponentiate.
         # that gives exp(0)=1 and others <1. to mimic softmax results but safely scaled.
        return E / E.sum(axis=1, keepdims=True)

    # Below uses soft plus, the above doesnt. USE BELOW ONE
    def softplus(self, z):  # stable: log(1+e^z)
        return np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0.0)

    def unpack_theta(self, theta, R=3):
        k = 4*R
        w_raw, a_raw, b_raw, g_raw = theta[:R], theta[R:2*R], theta[2*R:3*R], theta[3*R:4*R]
        P_logits = theta[k:k+R*R].reshape(R, R)
        rest = theta[k+R*R:]
        if rest.size >= 4:
            lam_raw, muJ, sJ_raw, d_raw = rest[:4]
            d2 = 1.0 / (1.0 + np.exp(-d_raw))
        elif rest.size >= 3:
            lam_raw, muJ, sJ_raw = rest[:3]
            d2 = 0.0
        else:
            lam_raw, muJ, sJ_raw = np.log(0.1), 0.0, np.log(0.02)
            d2 = 0.0

        omega = np.exp(w_raw)
        alpha = 1/(1+np.exp(-a_raw))
        beta  = 1/(1+np.exp(-b_raw))
        gamma = 1/(1+np.exp(-g_raw))
        
        s = alpha + beta + gamma
        mask = s >= 0.999
        mask02 = mask & np.array([True, False, True])
        if mask02.any():
            alpha[mask02] *= 0.999 / s[mask02]
            beta[mask02]  *= 0.999 / s[mask02]
            gamma[mask02] *= 0.999 / s[mask02]

        ####### mdl07b  regime 1: only shrink beta1 (keep alpha1, gamma1)
        if mask[1]:
            beta[1] = beta[1] * (0.999 / s[1])

        # Markov transitions
        P = np.exp(P_logits - P_logits.max(axis=1, keepdims=True))
        P = P / P.sum(axis=1, keepdims=True)

        # Jumps with stable positivity and caps
        lam     = np.minimum(self.softplus(lam_raw), 5.0)          # λ ∈ (0,5]
        sigmaJ  = np.minimum(self.softplus(sJ_raw),  1.0)          # σ_J ∈ (0,1]
        sigmaJ2 = sigmaJ*sigmaJ
        ############################################################################################################################## ADDED
        delta = rest[4] if rest.size >= 5 else 0.0
        return omega, alpha, beta, gamma, P, lam, muJ, sigmaJ2, d2, delta

    def nll_ms_garch(self, theta, r, iv, R=3, tol=1e-12):
        r  = np.asarray(r, float)
        iv = np.asarray(iv, float)

        omega, alpha, beta, gamma, P, lam, muJ, sigmaJ2, d2, delta = self.unpack_theta(theta, R)

        pi = np.full(R, 1.0/R)
        h  = (omega + alpha*lam*sigmaJ2) / (1 - alpha - beta + 1e-9)
        h  = np.maximum(h, 1e-8)
        h_init_full = h.copy()
        g2 = h[2].copy() if h.size > 2 else 0.0
        ll = 0.0

        for i, x in enumerate(r):
            iv_x = float(iv[i]) if (i < len(iv) and np.isfinite(iv[i])) else 0.0 ################### CHECK

            u   = x - lam*muJ
            var = np.maximum(h + lam*sigmaJ2, 1e-12)
            dens = (1.0/np.sqrt(2*np.pi*var)) * np.exp(-0.5*(u*u)/var)
            mix  = pi @ dens + tol
            ll  += np.log(mix)
            post = (pi * dens) / mix

            neg = float(u < 0.0)
            h_old = h.copy()

            h0  = omega[0] + delta*iv_x + alpha[0]*(u*u + lam*sigmaJ2) + beta[0]*h_old[0]
            h1  = omega[1] + delta*iv_x + (alpha[1] + gamma[1]*neg)*(u*u + lam*sigmaJ2) + beta[1]*h_old[1]

            if d2 >= 1e-4:
                if h_old.size < 3:
                    h_old = np.r_[h_old, h_init_full[2]]
                    h     = np.r_[h,     h_init_full[2]]
                    pi    = np.r_[pi,    0.0]

                h2 = omega[2] + delta*iv_x + alpha[2]*(u*u + lam*sigmaJ2) + beta[2]*((1 - d2)*h_old[2] + d2*g2)
                g2 = h_old[2].copy()

                h  = np.array([h0, h1, h2])
                P_use = P[:3, :3]
            else:
                h  = np.array([h0, h1])
                P_use = P[:2, :2]

            n_active = P_use.shape[0]
            pi = post[:n_active].flatten() @ P_use

        return -ll

    def forecast_var_k(self, r, iv, theta, R=3, k=3):
        omega, alpha, beta, gamma, P, lam, muJ, sigmaJ2, d2, delta = self.unpack_theta(theta, R)
        pi = np.full(R, 1.0/R)
        h  = (omega + alpha*lam*sigmaJ2) / (1 - alpha - beta + 1e-9)
        regime = []
        g2 = h[2].copy() if len(h) > 2 else 0.0
        thresh = 1e-4

        

        for j,x in enumerate(r):
            iv_x = float(iv[j])  if (j < len(iv) and np.isfinite(iv[j])) else 0.0  ################### CHECK
            Jt = lam * muJ
            u  = x - Jt
            neg = float(u < 0.0)
            u2 = u*u

            if d2 >= thresh:
                h_old = h.copy()
                h = omega + delta*iv_x + alpha*u2 + gamma*u2*neg + beta*h_old

                # overwrite regime-3 with FIGARCH-style recursion, but INCLUDE iv term too
                h[2] = omega[2] + delta*iv_x + alpha[2]*(u2 + lam*sigmaJ2) + beta[2]*((1 - d2)*h_old[2] + d2*g2)
                g2 = h_old[2].copy()

                pi = pi @ P
            else:
                # 2-regime update, INCLUDE iv term
                h = omega[:2] + delta*iv_x + alpha[:2]*u2 + gamma[:2]*u2*neg + beta[:2]*h[:2]
                P = P[:2, :2]
                pi = pi[:2] @ P

        out = []
        for _ in range(k):
            out.append(float(pi @ h))

            if d2 >= thresh and h.size == 3:
                h_old = h.copy()
                h = omega + delta*iv_x + alpha*out[-1] + beta*h_old

                # keep regime-3 recursion, INCLUDE iv term
                g2 = h_old[2].copy()
                h[2] = omega[2] + delta*iv_x + alpha[2]*out[-1] + beta[2]*((1 - d2)*h_old[2] + d2*g2)

                pi = pi @ P
            else:
                h = omega[:2] + delta*iv_x + alpha[:2]*out[-1] + beta[:2]*h
                pi = pi @ P[:2, :2]

            regime.append(pi.copy())

        return np.array(out), regime


####################################################################################
import numpy as np

class msGARCH:
    """
    Wrapper class for your plain Markov-switching / MS-GARCH functions.

    IMPORTANT:
    - The actual function bodies are unchanged.
    - Everything is accessible via this class (attributes + methods).
    """

    def __init__(self):
        # ######### PLAIN MARKOV SWITCHING MODEL
        self.R = 3  # number of regimes

        self.P = np.array([[0.90, 0.05, 0.05],
                           [0.10, 0.80, 0.10],
                           [0.05, 0.10, 0.85]])

        self.pi_0 = np.full(self.R, 1 / self.R)

        self.params = {
            0: [0.01, 0.05, 0.90],
            1: [0.02, 0.10, 0.80],
            2: [0.03, 0.07, 0.85]
        }

        # preserve your _lgit + theta0 exactly, but make them available on the instance
        self.theta0 = self._build_theta0(self.R)

    # -------------------------
    # Helper to build theta0 (no change to your construction)
    # -------------------------
    def _build_theta0(self, R):
        def _lgit(x): return np.log(x / (1 - x))
        theta0 = np.r_[np.log(0.01) * np.ones(R),
                       _lgit(0.05) * np.ones(R),
                       _lgit(0.90) * np.ones(R),
                       np.zeros(R),                  # γ
                       np.zeros(R * R),
                       np.log(0.1),   # lam_raw
                       0.0,           # muJ
                       np.log(0.02),  # sJ_raw
                       _lgit(0.4)
                       ]
        return theta0

    # -------------------------
    # Your functions (unchanged)
    # -------------------------
    def sigmoid(self, x):
        x = np.clip(x, -500, 500)     # avoid overflow
        return 1 / (1 + np.exp(-x))

    def row_softmax(self, M):
        M = M - M.max(axis=1, keepdims=True)
        E = np.exp(M)
         # we substract the maximum in each value from all entries in that row
         # so the largest becomes 0, then we exponentiate.
         # that gives exp(0)=1 and others <1. to mimic softmax results but safely scaled.
        return E / E.sum(axis=1, keepdims=True)

    # Below uses soft plus, the above doesnt. USE BELOW ONE
    def softplus(self, z):  # stable: log(1+e^z)
        return np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0.0)

    def unpack_theta(self, theta, R=3):
        k = 4*R
        w_raw, a_raw, b_raw, g_raw = theta[:R], theta[R:2*R], theta[2*R:3*R], theta[3*R:4*R]
        P_logits = theta[k:k+R*R].reshape(R, R)
        rest = theta[k+R*R:]
        if rest.size >= 4:
            lam_raw, muJ, sJ_raw, d_raw = rest[:4]
            d2 = 1.0 / (1.0 + np.exp(-d_raw))
        elif rest.size >= 3:
            lam_raw, muJ, sJ_raw = rest[:3]
            d2 = 0.0
        else:
            lam_raw, muJ, sJ_raw = np.log(0.1), 0.0, np.log(0.02)
            d2 = 0.0

        omega = np.exp(w_raw)
        alpha = 1/(1+np.exp(-a_raw))
        beta  = 1/(1+np.exp(-b_raw))
        gamma = 1/(1+np.exp(-g_raw))
        
        # gamma[0] = 0 # This allows for the second regime to have no leverage effect
        # single unified stationarity cap (incl. leverage)
        s = alpha + beta + gamma
        mask = s >= 0.999
        if mask.any():
            alpha[mask] *= 0.999 / s[mask]
            beta[mask]  *= 0.999 / s[mask]
            gamma[mask] *= 0.999 / s[mask]

        # Markov transitions
        P = np.exp(P_logits - P_logits.max(axis=1, keepdims=True))
        P = P / P.sum(axis=1, keepdims=True)

        # Jumps with stable positivity and caps
        lam     = np.minimum(self.softplus(lam_raw), 5.0)          # λ ∈ (0,5]
        sigmaJ  = np.minimum(self.softplus(sJ_raw),  1.0)          # σ_J ∈ (0,1]
        sigmaJ2 = sigmaJ*sigmaJ
        return omega, alpha, beta, gamma, P, lam, muJ, sigmaJ2, d2

    def nll_ms_garch(self, theta, r, R=3, tol=1e-12):
        r = np.asarray(r, float)
        omega, alpha, beta, gamma, P, lam, muJ, sigmaJ2, d2 = self.unpack_theta(theta, R)
        pi = np.full(R, 1.0/R)
        h  = (omega + alpha*lam*sigmaJ2) / (1 - alpha - beta + 1e-9)
        h  = np.maximum(h, 1e-8)
        h_init_full = h.copy()
        g2 = h[2].copy() if h.size > 2 else 0.0
        ll = 0.0

        for x in r:
            u   = x - lam*muJ
            var = np.maximum(h + lam*sigmaJ2, 1e-12)                 # include jump variance
            dens = (1.0/np.sqrt(2*np.pi*var)) * np.exp(-0.5*(u*u)/var)
            mix  = pi @ dens + tol
            ll  += np.log(mix)
            post = (pi * dens) / mix

            neg = float(u < 0.0)
            h_old = h.copy()                     # save previous state for FIGARCH lag
            h0  = omega[0] + alpha[0]*(u*u + lam*sigmaJ2) + beta[0]*h_old[0]
            h1  = omega[1] + (alpha[1] + gamma[1]*neg)*(u*u + lam*sigmaJ2) + beta[1]*h_old[1]

            
            if d2 >= 1e-4:
                if h_old.size < 3:
                    h_old = np.r_[h_old, h_init_full[2]]
                    h      = np.r_[h,      h_init_full[2]]
                    pi     = np.r_[pi,     0.0]       # append zero weight for regime 3
                # h2 = omega[2] + alpha[2]*(u*u + lam*sigmaJ2) + beta[2]*((1 - d2)*h_old[2] + d2*h_old[2])
                h2 = omega[2] + alpha[2]*(u*u + lam*sigmaJ2) + beta[2]*((1 - d2)*h_old[2] + d2*g2)
                g2 = h_old[2].copy()
                h  = np.array([h0, h1, h2])
                P_use = P[:3, :3] 
            else:
                h  = np.array([h0, h1])
                P_use = P[:2, :2]
            n_active = P_use.shape[0]
            pi = post[:n_active].flatten() @ P_use

        return -ll

    # Poisson-jump recursive variance update (structural form)
    ################# changed in mdl07
    def forecast_var_k(self, r, theta, R=3, k=15):
        omega, alpha, beta, gamma, P, lam, muJ, sigmaJ2, d2 = self.unpack_theta(theta, R)
        pi = np.full(R, 1.0/R)
        h  = (omega + alpha*lam*sigmaJ2) / (1 - alpha - beta + 1e-9)
        regime = []
        g2 = h[2].copy() if len(h) > 2 else 0.0     # long-memory state for regime 3
        thresh = 1e-4
        
        for x in r:
            Jt = lam * muJ
            u  = x - Jt
            neg = (u < 0).astype(float)
            u2 = u*u

            if d2 >= thresh:
                h_old = h.copy()
                # full 3-regime update (regime 3 uses FIGARCH recursion)
                h = omega + alpha*u2 + gamma*u2*neg + beta* h_old
                # h2 = omega[2] + alpha[2]*(u2 + lam*sigmaJ2) + beta[2]*((1 - d2)*h_old[2] + d2*h_old[2])
                # TWO NEW LINES BELOW
                h[2] = omega[2] + alpha[2]*(u2 + lam*sigmaJ2) + beta[2]*((1 - d2)*h_old[2] + d2*g2)
                g2 = h_old[2].copy()

                ####
                pi = pi @ P
                regime.append(pi.copy())
            else:
                # FIGARCH inactive -> operate only on first two regimes (drop regime 3)
                h = omega[:2] + alpha[:2]*u2 + gamma[:2]*u2*neg + beta[:2]*h[:2]
                P = P[:2, :2]            # overwrite P to the active 2x2 submatrix
                pi = pi[:2] @ P          # use first-two probabilities (not renormalized)
                regime.append(pi.copy())

        out = []
        for _ in range(k): # USING THROW AWAY LOOP
            out.append(float(pi @ h))
            if d2 >= thresh and h.size == 3:
                h_old = h.copy()     
                # 3-regime forecast (keep FIGARCH recursion for regime 3)
                h = omega + alpha*out[-1] + beta*h_old
                # g2 = (1.0 - d2)*(out[-1] + lam*sigmaJ2) + d2*g2
                # h[2] = omega[2] + alpha[2]*out[-1] + beta[2]*((1 - d2)*h_old[2] + d2*h_old[2])
                # TWO NEW LINES
                g2 = h_old[2].copy()
                h[2] = omega[2] + alpha[2]*out[-1] + beta[2]*((1 - d2)*h_old[2] + d2*g2)

                ####
                pi = pi @ P
            else:
                # 2-regime forecast (use first-two slices)
                h = omega[:2] + alpha[:2]*out[-1] + beta[:2]*h
                pi = pi @ P[:2, :2]

        return np.array(out), regime
