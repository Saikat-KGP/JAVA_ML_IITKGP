import java.util.Random;

public class RandomForestTest {
    public static void main(String[] args) {
        double[][] X = generateSyntheticData(1000, 5);
        int[] y = generateLabels(X);


        double[][] XTrain = new double[800][];
        double[][] XTest = new double[200][];
        int[] yTrain = new int[800];
        int[] yTest = new int[200];

        System.arraycopy(X, 0, XTrain, 0, 800);
        System.arraycopy(X, 800, XTest, 0, 200);
        System.arraycopy(y, 0, yTrain, 0, 800);
        System.arraycopy(y, 800, yTest, 0, 200);


        RandomForest rf = new RandomForest(10, 10, 2, null);
        rf.fit(XTrain, yTrain);


        int[] predictions = rf.predict(XTest);

        double accuracy = calculateAccuracy(yTest, predictions);
        System.out.println("Accuracy: " + accuracy * 100 + "%");
    }

    private static double[][] generateSyntheticData(int nSamples, int nFeatures) {
        Random rand = new Random();
        double[][] data = new double[nSamples][nFeatures];
        for (int i = 0; i < nSamples; i++) {
            for (int j = 0; j < nFeatures; j++) {
                data[i][j] = rand.nextDouble();
            }
        }
        return data;
    }

    private static int[] generateLabels(double[][] X) {
        int[] labels = new int[X.length];
        for (int i = 0; i < X.length; i++) {
            double sum = 0.0;
            for (double val : X[i]) {
                sum += val;
            }
            labels[i] = sum > 2.5 ? 1 : 0; 
        }
        return labels;
    }

    private static double calculateAccuracy(int[] trueLabels, int[] predictions) {
        int correct = 0;
        for (int i = 0; i < trueLabels.length; i++) {
            if (trueLabels[i] == predictions[i]) {
                correct++;
            }
        }
        return (double) correct / trueLabels.length;
    }
}
