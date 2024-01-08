import numpy as np
import math


class SVM:
    def __init__(self, x_train, y_train, kkt_thr=1e-3, max_iter=1e3):
        self.max_iter = max_iter
        self.kkt_thr = kkt_thr
        self.x_train = x_train
        self.y_train = y_train
        self.N, self.D = x_train.shape
        self.alpha = np.zeros(self.N)
        self.w = np.zeros(self.D)
        self.b = 0
        self.kernelMatrix = np.dot(x_train, x_train.T)  # Precompute the kernel matrix
        self.Q = np.outer(y_train, y_train) * self.kernelMatrix
        self.E = -y_train.astype(float)
        self.G = np.full(self.N, -1.0)
        self.support_vectors_idx = None

    def select_i_j(self):
        # Improved selection of i and j using vectorized operations
        yG = -self.y_train * self.G
        eligible_i = (self.y_train == 1) | ((self.y_train == -1) & (self.alpha > 0))
        eligible_j = ((self.y_train == 1) & (self.alpha > 0)) | (self.y_train == -1)
        if not np.any(eligible_i) or not np.any(eligible_j):
            return -1, -1

        G_max = yG[eligible_i].max()
        G_min = yG[eligible_j].min()
        
        if G_max - G_min < self.kkt_thr:
            return -1, -1

        i = np.where(yG == G_max)[0][0]
        j_candidates = np.where(yG == G_min)[0]
        obj_min = math.inf
        j = -1
        for t in j_candidates:
            b = G_max + yG[t]
            a = (
                self.Q[i, i]
                + self.Q[t, t]
                - 2 * self.y_train[i] * self.y_train[t] * self.Q[i, t]
            )
            a = max(a, 0)
            if -(b * b) / a < obj_min:
                j = t
                obj_min = -(b * b) / a

        return i, j

    def fit(self):
        idx = 0
        while idx < int(self.max_iter):
            idx += 1
            i, j = self.select_i_j()
            if j == -1:
                break

            # Compute 'a' and 'b' and update alphas
            a = (
                self.Q[i, i]
                + self.Q[j, j]
                - 2 * self.y_train[i] * self.y_train[j] * self.Q[i, j]
            )
            a = max(a, 0)
            b = -self.y_train[i] * self.G[i] + self.y_train[j] * self.G[j]

            old_ai, old_aj = self.alpha[i], self.alpha[j]
            self.alpha[i] += self.y_train[i] * b / a
            self.alpha[j] -= self.y_train[j] * b / a

            # Project alpha back to the feasible region
            sum_alpha = self.y_train[i] * old_ai + self.y_train[j] * old_aj
            self.alpha[i] = max(0, self.alpha[i])
            self.alpha[j] = self.y_train[j] * (
                sum_alpha - self.y_train[i] * self.alpha[i]
            )
            self.alpha[j] = max(0, self.alpha[j])
            self.alpha[i] = self.y_train[i] * (
                sum_alpha - self.y_train[j] * self.alpha[j]
            )

            # Update gradient
            delta_ai = self.alpha[i] - old_ai
            delta_aj = self.alpha[j] - old_aj
            self.G += self.Q[:, i] * delta_ai + self.Q[:, j] * delta_aj

            # Update weights
            self.w += (
                self.y_train[i] * delta_ai * self.x_train[i]
                + self.y_train[j] * delta_aj * self.x_train[j]
            )

        # Update support vectors and bias
        self.support_vectors_idx = np.where(self.alpha > 0)[0]
        self.b = np.mean(
            self.y_train[self.support_vectors_idx]
            - np.dot(self.x_train[self.support_vectors_idx], self.w)
        )
        print(
            f"training completed!\niterations:{idx}\nnumber of support vectors:{self.support_vectors_idx.shape[0]}"
        )

    def predict(self, x):
        scores = np.dot(self.w, x.T) + self.b
        pred = np.sign(scores)
        return pred, scores

    def check_kkt(self, check_idx: int) -> np.ndarray:
        """
        This function checks if sample_idx satisfies KKT conditions.

        Arguments:
            check_idx: Indices of alphas to check (scalar or vector)

        Return:
            kkt_condition_satisfied: Boolean array per alpha
        """

        alpha = self.alpha[check_idx]
        _, score_i = self.predict(self.x_train[check_idx, :])
        y_i = self.y_train[check_idx]
        r_i = y_i * score_i - 1
        cond_1 = (alpha == 0) & (r_i >= -self.kkt_thr)
        cond_2 = (alpha > 0) & ((-self.kkt_thr / 2 <= r_i) & (r_i <= self.kkt_thr / 2))

        return ~(cond_1 | cond_2)
