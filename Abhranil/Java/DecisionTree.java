import java.util.*;
import java.util.stream.Collectors;

class Node {
    Integer feature;
    Double threshold;
    Node left;
    Node right;
    Integer value;

    Node(Integer feature, Double threshold, Node left, Node right, Integer value) {
        this.feature = feature;
        this.threshold = threshold;
        this.left = left;
        this.right = right;
        this.value = value;
    }

    boolean isLeafNode() {
        return value != null;
    }
}

public class DecisionTree {
    private int minSamplesSplit;
    private int maxDepth;
    private Integer nFeatures;
    private Node root;

    public DecisionTree(int minSamplesSplit, int maxDepth, Integer nFeatures) {
        this.minSamplesSplit = minSamplesSplit;
        this.maxDepth = maxDepth;
        this.nFeatures = nFeatures;
    }

    public void fit(double[][] X, int[] y) {
        int nFeatures = X[0].length;
        this.nFeatures = (this.nFeatures == null) ? nFeatures : Math.min(nFeatures, this.nFeatures);
        this.root = growTree(X, y, 0);
    }

    private Node growTree(double[][] X, int[] y, int depth) {
        int nLabels = (int) Arrays.stream(y).distinct().count();

        if (depth >= maxDepth || nLabels == 1 || X.length < minSamplesSplit) {
            int leafValue = mostCommonLabel(y);
            return new Node(null, null, null, null, leafValue);
        }

        int[] featIdxs = randomSubset(nFeatures, X[0].length);

        double[] bestSplit = bestSplit(X, y, featIdxs);
        int bestFeature = (int) bestSplit[0];
        double bestThresh = bestSplit[1];

        int[][] splitIndices = split(X, bestFeature, bestThresh);
        int[] leftIdxs = splitIndices[0];
        int[] rightIdxs = splitIndices[1];

        double[][] XLeft = getSubMatrix(X, leftIdxs);
        int[] yLeft = getSubArray(y, leftIdxs);

        double[][] XRight = getSubMatrix(X, rightIdxs);
        int[] yRight = getSubArray(y, rightIdxs);

        Node left = growTree(XLeft, yLeft, depth + 1);
        Node right = growTree(XRight, yRight, depth + 1);

        return new Node(bestFeature, bestThresh, left, right, null);
    }

    private double[] bestSplit(double[][] X, int[] y, int[] featIdxs) {
        double bestGain = -1;
        int splitIdx = -1;
        double splitThresh = -1;

        for (int featIdx : featIdxs) {
            double[] XColumn = getColumn(X, featIdx);
            double[] thresholds = Arrays.stream(XColumn).distinct().toArray();

            for (double thr : thresholds) {
                double gain = informationGain(y, XColumn, thr);

                if (gain > bestGain) {
                    bestGain = gain;
                    splitIdx = featIdx;
                    splitThresh = thr;
                }
            }
        }

        return new double[]{splitIdx, splitThresh};
    }

    private double informationGain(int[] y, double[] XColumn, double threshold) {
        double parentEntropy = entropy(y);

        int[][] splitIndices = split(XColumn, threshold);
        int[] leftIdxs = splitIndices[0];
        int[] rightIdxs = splitIndices[1];

        if (leftIdxs.length == 0 || rightIdxs.length == 0) {
            return 0;
        }

        int n = y.length;
        int nL = leftIdxs.length;
        int nR = rightIdxs.length;

        double eL = entropy(getSubArray(y, leftIdxs));
        double eR = entropy(getSubArray(y, rightIdxs));

        double childEntropy = ((double) nL / n) * eL + ((double) nR / n) * eR;

        return parentEntropy - childEntropy;
    }

    private int[][] split(double[] XColumn, double splitThresh) {
        List<Integer> leftIdxs = new ArrayList<>();
        List<Integer> rightIdxs = new ArrayList<>();

        for (int i = 0; i < XColumn.length; i++) {
            if (XColumn[i] <= splitThresh) {
                leftIdxs.add(i);
            } else {
                rightIdxs.add(i);
            }
        }

        return new int[][]{leftIdxs.stream().mapToInt(i -> i).toArray(), rightIdxs.stream().mapToInt(i -> i).toArray()};
    }

    private int[][] split(double[][] X, int splitFeature, double splitThresh) {
        List<Integer> leftIdxs = new ArrayList<>();
        List<Integer> rightIdxs = new ArrayList<>();

        for (int i = 0; i < X.length; i++) {
            if (X[i][splitFeature] <= splitThresh) {
                leftIdxs.add(i);
            } else {
                rightIdxs.add(i);
            }
        }

        return new int[][]{leftIdxs.stream().mapToInt(i -> i).toArray(), rightIdxs.stream().mapToInt(i -> i).toArray()};
    }

    private double entropy(int[] y) {
        Map<Integer, Long> counts = Arrays.stream(y).boxed().collect(Collectors.groupingBy(e -> e, Collectors.counting()));
        double entropy = 0.0;
        for (long count : counts.values()) {
            double p = (double) count / y.length;
            entropy -= p * Math.log(p);
        }
        return entropy;
    }

    private int mostCommonLabel(int[] y) {
        Map<Integer, Long> counts = Arrays.stream(y).boxed().collect(Collectors.groupingBy(e -> e, Collectors.counting()));
        return Collections.max(counts.entrySet(), Map.Entry.comparingByValue()).getKey();
    }

    public int[] predict(double[][] X) {
        return Arrays.stream(X).mapToInt(x -> traverseTree(x, root)).toArray();
    }

    private int traverseTree(double[] x, Node node) {
        if (node.isLeafNode()) {
            return node.value;
        }

        if (x[node.feature] <= node.threshold) {
            return traverseTree(x, node.left);
        } else {
            return traverseTree(x, node.right);
        }
    }

    private double[] getColumn(double[][] matrix, int col) {
        return Arrays.stream(matrix).mapToDouble(row -> row[col]).toArray();
    }

    private double[][] getSubMatrix(double[][] matrix, int[] rows) {
        return Arrays.stream(rows).mapToObj(row -> matrix[row]).toArray(double[][]::new);
    }

    private int[] getSubArray(int[] array, int[] indices) {
        return Arrays.stream(indices).map(i -> array[i]).toArray();
    }

    private int[] randomSubset(int size, int total) {
        Random rand = new Random();
        return rand.ints(0, total).distinct().limit(size).toArray();
    }
}
