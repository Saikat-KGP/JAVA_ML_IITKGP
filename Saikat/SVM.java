import java.util.Random;

public class SVM {
    private double C;
    private double tol;
    private int maxPasses;
    private double[] alphas;
    private double b;
    private double[][] X;
    private double[] y;
    private Random random;

    public SVM(double C, double tol, int maxPasses) {
        this.C = C;
        this.tol = tol;
        this.maxPasses = maxPasses;
        this.random = new Random();
    }

    public void fit(double[][] X, double[] y) {
        int m = X.length;
        int n = X[0].length;
        this.alphas = new double[m];
        this.b = 0;
        this.X = X;
        this.y = y;
        int passes = 0;

        while (passes < this.maxPasses) {
            int numChangedAlphas = 0;
            for (int i = 0; i < m; i++) {
                double Ei = decisionFunction(X[i]) - y[i];
                if ((y[i] * Ei < -this.tol && this.alphas[i] < this.C) || (y[i] * Ei > this.tol && this.alphas[i] > 0)) {
                    int j = selectJ(i, m);
                    double Ej = decisionFunction(X[j]) - y[j];

                    double oldAlphaI = this.alphas[i];
                    double oldAlphaJ = this.alphas[j];

                    double L, H;
                    if (y[i] != y[j]) {
                        L = Math.max(0, this.alphas[j] - this.alphas[i]);
                        H = Math.min(this.C, this.C + this.alphas[j] - this.alphas[i]);
                    } else {
                        L = Math.max(0, this.alphas[i] + this.alphas[j] - this.C);
                        H = Math.min(this.C, this.alphas[i] + this.alphas[j]);
                    }

                    if (L == H) continue;

                    double eta = 2.0 * kernel(X[i], X[j]) - kernel(X[i], X[i]) - kernel(X[j], X[j]);
                    if (eta >= 0) continue;

                    this.alphas[j] -= y[j] * (Ei - Ej) / eta;
                    this.alphas[j] = clipAlpha(this.alphas[j], H, L);

                    if (Math.abs(this.alphas[j] - oldAlphaJ) < 1e-5) continue;

                    this.alphas[i] += y[i] * y[j] * (oldAlphaJ - this.alphas[j]);

                    double b1 = this.b - Ei - y[i] * (this.alphas[i] - oldAlphaI) * kernel(X[i], X[i]) - y[j] * (this.alphas[j] - oldAlphaJ) * kernel(X[i], X[j]);
                    double b2 = this.b - Ej - y[i] * (this.alphas[i] - oldAlphaI) * kernel(X[i], X[j]) - y[j] * (this.alphas[j] - oldAlphaJ) * kernel(X[j], X[j]);

                    if (0 < this.alphas[i] && this.alphas[i] < this.C) {
                        this.b = b1;
                    } else if (0 < this.alphas[j] && this.alphas[j] < this.C) {
                        this.b = b2;
                    } else {
                        this.b = (b1 + b2) / 2.0;
                    }

                    numChangedAlphas++;
                }
            }

            if (numChangedAlphas == 0) {
                passes++;
            } else {
                passes = 0;
            }
        }
    }

    public double predict(double[] X) {
        return Math.signum(decisionFunction(X));
    }

    private double decisionFunction(double[] X) {
        double result = 0;
        for (int i = 0; i < this.X.length; i++) {
            result += this.alphas[i] * this.y[i] * kernel(this.X[i], X);
        }
        return result + this.b;
    }

    private double kernel(double[] x1, double[] x2) {
        double sum = 0;
        for (int i = 0; i < x1.length; i++) {
            sum += x1[i] * x2[i];
        }
        return sum;
    }

    private int selectJ(int i, int m) {
        int j = i;
        while (j == i) {
            j = this.random.nextInt(m);
        }
        return j;
    }

    private double clipAlpha(double alpha, double H, double L) {
        if (alpha > H) {
            return H;
        } else if (alpha < L) {
            return L;
        } else {
            return alpha;
        }
    }

    public static void main(String[] args) {
        double[][] X = {
            {1, 2},
            {2, 3},
            {3, 3},
            {2, 1},
            {3, 2}
        };
        double[] y = {1, 1, 1, -1, -1};

        SVM svm = new SVM(1.0, 0.001, 1000);
        svm.fit(X, y);

        double[] testPoint = {2.5, 2.5};
        double prediction = svm.predict(testPoint);
        System.out.println("Prediction for [2.5, 2.5]: " + prediction);
    }
}


