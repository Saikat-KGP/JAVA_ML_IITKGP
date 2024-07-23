import java.util.Arrays;
import java.util.Random;

public class MLP {

    private int inputSize;
    private int hiddenSize;
    private int outputSize;
    private double[][] weightsInputHidden;
    private double[] biasHidden;
    private double[][] weightsHiddenOutput;
    private double[] biasOutput;
    private double[] hiddenLayerOutput;
    private double[] outputLayerOutput;
    private Random random;

    public MLP(int inputSize, int hiddenSize, int outputSize) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;
        this.random = new Random();

        // Initialize weights and biases
        weightsInputHidden = new double[inputSize][hiddenSize];
        biasHidden = new double[hiddenSize];
        weightsHiddenOutput = new double[hiddenSize][outputSize];
        biasOutput = new double[outputSize];
        hiddenLayerOutput = new double[hiddenSize];
        outputLayerOutput = new double[outputSize];

        initializeWeights(weightsInputHidden);
        initializeWeights(weightsHiddenOutput);
        initializeBias(biasHidden);
        initializeBias(biasOutput);
    }

    private void initializeWeights(double[][] weights) {
        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[i].length; j++) {
                weights[i][j] = random.nextDouble() - 0.5;
            }
        }
    }

    private void initializeBias(double[] bias) {
        for (int i = 0; i < bias.length; i++) {
            bias[i] = random.nextDouble() - 0.5;
        }
    }

    private double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    private double sigmoidDerivative(double x) {
        return x * (1 - x);
    }

    public double[] forward(double[] input) {
        // Input to Hidden Layer
        for (int j = 0; j < hiddenSize; j++) {
            double sum = 0;
            for (int i = 0; i < inputSize; i++) {
                sum += input[i] * weightsInputHidden[i][j];
            }
            sum += biasHidden[j];
            hiddenLayerOutput[j] = sigmoid(sum);
        }

        // Hidden to Output Layer
        for (int k = 0; k < outputSize; k++) {
            double sum = 0;
            for (int j = 0; j < hiddenSize; j++) {
                sum += hiddenLayerOutput[j] * weightsHiddenOutput[j][k];
            }
            sum += biasOutput[k];
            outputLayerOutput[k] = sigmoid(sum);
        }

        return outputLayerOutput;
    }

    public void backward(double[] input, double[] target, double[] output, double learningRate) {
        double[] outputErrors = new double[outputSize];
        double[] outputDeltas = new double[outputSize];

        for (int k = 0; k < outputSize; k++) {
            outputErrors[k] = target[k] - output[k];
            outputDeltas[k] = outputErrors[k] * sigmoidDerivative(output[k]);
        }

        double[] hiddenErrors = new double[hiddenSize];
        double[] hiddenDeltas = new double[hiddenSize];

        for (int j = 0; j < hiddenSize; j++) {
            double sum = 0;
            for (int k = 0; k < outputSize; k++) {
                sum += outputDeltas[k] * weightsHiddenOutput[j][k];
            }
            hiddenErrors[j] = sum;
            hiddenDeltas[j] = hiddenErrors[j] * sigmoidDerivative(hiddenLayerOutput[j]);
        }

        for (int j = 0; j < hiddenSize; j++) {
            for (int k = 0; k < outputSize; k++) {
                weightsHiddenOutput[j][k] += learningRate * outputDeltas[k] * hiddenLayerOutput[j];
            }
        }

        for (int k = 0; k < outputSize; k++) {
            biasOutput[k] += learningRate * outputDeltas[k];
        }

        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < hiddenSize; j++) {
                weightsInputHidden[i][j] += learningRate * hiddenDeltas[j] * input[i];
            }
        }

        for (int j = 0; j < hiddenSize; j++) {
            biasHidden[j] += learningRate * hiddenDeltas[j];
        }
    }

    public void train(double[][] inputs, double[][] targets, int epochs, double learningRate) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int i = 0; i < inputs.length; i++) {
                double[] output = forward(inputs[i]);
                backward(inputs[i], targets[i], output, learningRate);
            }
        }
    }

    public double[] predict(double[] input) {
        return forward(input);
    }

    public static void main(String[] args) {
        // XOR Problem
        double[][] inputs = {
            {0, 0},
            {0, 1},
            {1, 0},
            {1, 1}
        };
        double[][] targets = {
            {0},
            {1},
            {1},
            {0}
        };

        MLP mlp = new MLP(2, 2, 1);
        mlp.train(inputs, targets, 10000, 0.1);

        for (double[] input : inputs) {
            double[] output = mlp.predict(input);
            System.out.println(Arrays.toString(output));
        }
    }
}