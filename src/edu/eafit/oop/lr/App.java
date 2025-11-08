package edu.eafit.oop.lr;

import java.util.*;

/**
 * CLI demo to train/evaluate LinearRegression from a CSV.
 *
 * Example:
 *   java -cp out edu.eafit.oop.lr.App --file data.csv --target-col y --method normal --scale standard
 *   java -cp out edu.eafit.oop.lr.App --file data.csv --target-col final --method gd --alpha 0.01 --epochs 20000
 */
public class App {

    private static void printHelp() {
        System.out.println("Usage:");
        System.out.println("  --file <path.csv>              CSV file (comma or semicolon is OK)");
        System.out.println("  --target-col <name|index>      Target column (header name or zero-based index)");
        System.out.println("  --method <normal|gd>           Training: normal equation or gradient descent");
        System.out.println("  --scale <none|standard>        Feature scaling (default: standard for GD, none for normal)");
        System.out.println("  --alpha <float>                Learning rate for GD (default 0.01)");
        System.out.println("  --epochs <int>                 Epochs for GD (default 10000)");
        System.out.println("  --split <float>                Train ratio (default 0.8)");
        System.out.println("  --seed <long>                  Random seed for split (default 42)");
        System.out.println();
        System.out.println("Example:");
        System.out.println("  java -cp out edu.eafit.oop.lr.App --file ice_cream.csv --target-col sales --method normal");
    }

    public static void main(String[] args) {
        Map<String,String> flags = parseArgs(args);
        if (flags.isEmpty() || flags.containsKey("--help")) {
            printHelp();
            return;
        }
        String file = flags.getOrDefault("--file", "");
        String targetCol = flags.getOrDefault("--target-col", "");
        String method = flags.getOrDefault("--method", "normal").toLowerCase(Locale.ROOT);
        String scale = flags.getOrDefault("--scale", method.equals("gd") ? "standard" : "none").toLowerCase(Locale.ROOT);
        double alpha = Double.parseDouble(flags.getOrDefault("--alpha", "0.01"));
        int epochs = Integer.parseInt(flags.getOrDefault("--epochs", "10000"));
        double split = Double.parseDouble(flags.getOrDefault("--split", "0.8"));
        long seed = Long.parseLong(flags.getOrDefault("--seed", "42"));

        if (file.isEmpty() || targetCol.isEmpty()) {
            System.err.println("ERROR: --file and --target-col are required.");
            printHelp();
            return;
        }
        if (!method.equals("normal") && !method.equals("gd")) {
            System.err.println("ERROR: --method must be 'normal' or 'gd'.");
            return;
        }
        if (!scale.equals("none") && !scale.equals("standard")) {
            System.err.println("ERROR: --scale must be 'none' or 'standard'.");
            return;
        }

        // Load CSV
        CSVReader.DataFrame df;
        try {
            df = CSVReader.read(file);
        } catch (Exception e) {
            System.err.println("ERROR reading CSV: " + e.getMessage());
            return;
        }

        // Resolve target column index
        int yIdx = -1;
        try {
            yIdx = Integer.parseInt(targetCol);
        } catch (NumberFormatException ignored) {
            yIdx = df.indexOfColumn(targetCol);
        }
        if (yIdx < 0) {
            System.err.println("ERROR: Could not find target column '" + targetCol + "'");
            return;
        }

        // Split train/test
        CSVReader.DataFrame[] splitDF = df.trainTestSplit(split, seed);
        CSVReader.DataFrame train = splitDF[0];
        CSVReader.DataFrame test  = splitDF[1];

        // Separate X and y
        double[][] Xtrain = train.featuresExcept(yIdx);
        double[]   ytrain = train.columnAsVector(yIdx);

        double[][] Xtest  = test.featuresExcept(yIdx);
        double[]   ytest  = test.columnAsVector(yIdx);

        // Optional scaling
        StandardScaler scaler = null;
        if (scale.equals("standard")) {
            scaler = new StandardScaler();
            scaler.fit(Xtrain);
            Xtrain = scaler.transform(Xtrain);
            Xtest  = scaler.transform(Xtest);
        }

        // Train model
        LinearRegression.TrainingMethod tm = method.equals("gd")
                ? LinearRegression.TrainingMethod.GRADIENT_DESCENT
                : LinearRegression.TrainingMethod.NORMAL_EQUATION;

        LinearRegression model = new LinearRegression(tm);
        if (tm == LinearRegression.TrainingMethod.GRADIENT_DESCENT) {
            model.setLearningRate(alpha);
            model.setEpochs(epochs);
        }
        model.fit(Xtrain, ytrain);

        // Report
        System.out.println("== Model parameters ==");
        System.out.println("bias: " + model.getBias());
        System.out.println("weights: " + Arrays.toString(model.getWeights()));

        double[] yhatTest = model.predict(Xtest);
        System.out.println("\n== Predictions (first 5) ==");
        for (int i = 0; i < Math.min(5, yhatTest.length); i++) {
            System.out.printf(Locale.US, "y_hat[%d] = %.6f (y=%.6f)%n", i, yhatTest[i], ytest[i]);
        }

        double r2  = model.score(Xtest, ytest); // R^2
        double mse = VectorStats.mse(ytest, yhatTest);
        double mae = VectorStats.mae(ytest, yhatTest);

        System.out.println("\n== Scores ==");
        System.out.printf(Locale.US, "R2  : %.6f%n", r2);
        System.out.printf(Locale.US, "MSE : %.6f%n", mse);
        System.out.printf(Locale.US, "MAE : %.6f%n", mae);

        if (scaler != null) {
            System.out.println("\n(Features were scaled with StandardScaler: mean=0, std=1)");
        }
    }

    private static Map<String,String> parseArgs(String[] args) {
        Map<String,String> m = new LinkedHashMap<>();
        for (int i=0; i<args.length; i++) {
            String a = args[i];
            if (a.startsWith("--")) {
                if (i+1 < args.length && !args[i+1].startsWith("--")) {
                    m.put(a, args[i+1]);
                    i++;
                } else {
                    m.put(a, "true");
                }
            }
        }
        return m;
    }
}
