import numpy as np

class SVM:
    def __init__(self, C=1.0, tol=0.001, max_passes=5):
        self.C = C
        self.tol = tol
        self.max_passes = max_passes

    def fit(self, X, y):
        m, n = X.shape
        self.alphas = np.zeros(m)
        self.b = 0
        self.X = X
        self.y = y
        passes = 0

        while passes < self.max_passes:
            num_changed_alphas = 0
            for i in range(m):
                Ei = self._decision_function(X[i]) - y[i]
                if (y[i] * Ei < -self.tol and self.alphas[i] < self.C) or (y[i] * Ei > self.tol and self.alphas[i] > 0):
                    j = self._select_j(i, m)
                    Ej = self._decision_function(X[j]) - y[j]

                    old_alpha_i = self.alphas[i]
                    old_alpha_j = self.alphas[j]

                    if y[i] != y[j]:
                        L = max(0, self.alphas[j] - self.alphas[i])
                        H = min(self.C, self.C + self.alphas[j] - self.alphas[i])
                    else:
                        L = max(0, self.alphas[i] + self.alphas[j] - self.C)
                        H = min(self.C, self.alphas[i] + self.alphas[j])

                    if L == H:
                        continue

                    eta = 2.0 * self._kernel(X[i], X[j]) - self._kernel(X[i], X[i]) - self._kernel(X[j], X[j])
                    if eta >= 0:
                        continue

                    self.alphas[j] -= y[j] * (Ei - Ej) / eta
                    self.alphas[j] = self._clip_alpha(self.alphas[j], H, L)

                    if abs(self.alphas[j] - old_alpha_j) < 1e-5:
                        continue

                    self.alphas[i] += y[i] * y[j] * (old_alpha_j - self.alphas[j])

                    b1 = self.b - Ei - y[i] * (self.alphas[i] - old_alpha_i) * self._kernel(X[i], X[i]) - y[j] * (self.alphas[j] - old_alpha_j) * self._kernel(X[i], X[j])
                    b2 = self.b - Ej - y[i] * (self.alphas[i] - old_alpha_i) * self._kernel(X[i], X[j]) - y[j] * (self.alphas[j] - old_alpha_j) * self._kernel(X[j], X[j])

                    if 0 < self.alphas[i] < self.C:
                        self.b = b1
                    elif 0 < self.alphas[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2

                    num_changed_alphas += 1

            if num_changed_alphas == 0:
                passes += 1
            else:
                passes = 0

    def predict(self, X):
        return np.sign(self._decision_function(X))

    def _decision_function(self, X):
        result = 0
        for i in range(len(self.X)):
            result += self.alphas[i] * self.y[i] * self._kernel(self.X[i], X)
        return result + self.b

    def _kernel(self, x1, x2):
        return np.dot(x1, x2)

    def _select_j(self, i, m):
        j = i
        while j == i:
            j = np.random.randint(0, m)
        return j

    def _clip_alpha(self, alpha, H, L):
        if alpha > H:
            return H
        elif alpha < L:
            return L
        else:
            return alpha

if __name__ == "__main__":
    X = np.array([
        [1, 2],
        [2, 3],
        [3, 3],
        [2, 1],
        [3, 2]
    ])
    y = np.array([1, 1, 1, -1, -1])

    svm = SVM(C=1.0, tol=0.001, max_passes=1000)
    svm.fit(X, y)

    test_point = np.array([2.5, 2.5])
    prediction = svm.predict(test_point)
    print("Prediction for [2.5, 2.5]:", prediction)
