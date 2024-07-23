import java.util.*;
import java.util.stream.Collectors;

public class RandomForest {
    private int nTrees;
    private int maxDepth;
    private int minSamplesSplit;
    private Integer nFeatures;
    private List<DecisionTree> trees;

    public RandomForest(int nTrees, int maxDepth, int minSamplesSplit, Integer nFeatures) {
        this.nTrees = nTrees;
        this.maxDepth = maxDepth;
        this.minSamplesSplit = minSamplesSplit;
        this.nFeatures = nFeatures;
        this.trees = new ArrayList<>();
    }

    public void fit(double[][] X, int[] y) {
        trees.clear();
        for (int i = 0; i < nTrees; i++) {
            DecisionTree tree = new DecisionTree(minSamplesSplit, maxDepth, nFeatures);
            int[][] bootstrappedSamples = bootstrapSamples(X, y);
            double[][] XSample = getSubMatrix(X, bootstrappedSamples[0]);
            int[] ySample = getSubArray(y, bootstrappedSamples[0]);
            tree.fit(XSample, ySample);
            trees.add(tree);
        }
    }

    private int[][] bootstrapSamples(double[][] X, int[] y) {
        Random rand = new Random();
        int nSamples = X.length;
        int[] indices = rand.ints(nSamples, 0, nSamples).toArray();
        return new int[][]{indices};
    }

    private double[][] getSubMatrix(double[][] matrix, int[] rows) {
        return Arrays.stream(rows).mapToObj(row -> matrix[row]).toArray(double[][]::new);
    }

    private int[] getSubArray(int[] array, int[] indices) {
        return Arrays.stream(indices).map(i -> array[i]).toArray();
    }

    public int[] predict(double[][] X) {
        int[][] predictions = trees.stream().map(tree -> tree.predict(X)).toArray(int[][]::new);
        int[][] transposedPredictions = transpose(predictions);
        return Arrays.stream(transposedPredictions).mapToInt(this::mostCommonLabel).toArray();
    }

    private int[][] transpose(int[][] matrix) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        int[][] transposed = new int[cols][rows];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                transposed[j][i] = matrix[i][j];
            }
        }

        return transposed;
    }

    private int mostCommonLabel(int[] array) {
        Map<Integer, Long> counts = Arrays.stream(array).boxed().collect(Collectors.groupingBy(e -> e, Collectors.counting()));
        return Collections.max(counts.entrySet(), Map.Entry.comparingByValue()).getKey();
    }
}
