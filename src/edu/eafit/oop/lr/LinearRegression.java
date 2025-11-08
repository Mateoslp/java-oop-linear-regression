package edu.eafit.oop.lr;

import java.util.Arrays;
import java.util.Locale;

/** Linear Regression model with two training methods (Normal Equation / Gradient Descent). */
public class LinearRegression implements RegressionModel {

    public enum TrainingMethod { NORMAL_EQUATION, GRADIENT_DESCENT }

    private final TrainingMethod method;
    private double[] weights; // size = n_features
    private double bias;      // scalar
    // GD params
    private double learningRate = 0.01;
    private int epochs = 10000;
    private double ridge = 1e-8; // small L2 for numerical stability in normal equation

    public LinearRegression(TrainingMethod method) {
        this.method = method;
    }

    public void setLearningRate(double lr) { this.learningRate = lr; }
    public void setEpochs(int e) { this.epochs = e; }
    public void setRidge(double lambda) { this.ridge = lambda; }

    public double[] getWeights() { return Arrays.copyOf(weights, weights.length); }
    public double getBias() { return bias; }

    @Override
    public void fit(double[][] X, double[] y) {
        int m = X.length;
        if (m == 0) throw new IllegalArgumentException("Empty X");
        int n = X[0].length;

        if (method == TrainingMethod.NORMAL_EQUATION) {
            // Add bias column of ones
            double[][] Xb = new double[m][n+1];
            for (int i=0; i<m; i++) {
                Xb[i][0] = 1.0;
                for (int j=0; j<n; j++) Xb[i][j+1] = X[i][j];
            }
            double[][] Xt = Matrix.transpose(Xb);
            double[][] XtX = Matrix.dot(Xt, Xb);
            // Add tiny ridge term to diagonal for stability
            for (int i=0; i<XtX.length; i++) XtX[i][i] += ridge;
            double[][] XtXinv = Matrix.inverse(XtX);
            double[] Xty = Matrix.dot(Xt, y);
            double[] theta = Matrix.dot(XtXinv, Xty);
            bias = theta[0];
            weights = new double[n];
            for (int j=0; j<n; j++) weights[j] = theta[j+1];
        } else {
            // Gradient Descent (assumes X is already scaled for best results)
            weights = new double[n];
            bias = 0.0;
            double lr = learningRate;
            for (int epoch = 0; epoch < epochs; epoch++) {
                // Predictions
                double[] yhat = predict(X);
                // Gradients
                double gradB = 0.0;
                double[] gradW = new double[n];
                for (int i=0; i<m; i++) {
                    double e = yhat[i] - y[i];
                    gradB += e;
                    for (int j=0; j<n; j++) gradW[j] += e * X[i][j];
                }
                gradB /= m;
                for (int j=0; j<n; j++) gradW[j] /= m;

                // Update params
                bias   -= lr * gradB;
                for (int j=0; j<n; j++) weights[j] -= lr * gradW[j];

                // (Optional) simple early stopping on tiny gradient
                double gradNorm = Math.abs(gradB);
                for (int j=0; j<n; j++) gradNorm += Math.abs(gradW[j]);
                if (gradNorm < 1e-8) break;
            }
        }
    }

    @Override
    public double[] predict(double[][] X) {
        int m = X.length;
        if (m == 0) return new double[0];
        int n = X[0].length;
        if (weights == null || weights.length != n) {
            throw new IllegalStateException("Model not fitted or feature size mismatch.");
        }
        double[] yhat = new double[m];
        for (int i=0; i<m; i++) {
            double s = bias;
            for (int j=0; j<n; j++) s += weights[j] * X[i][j];
            yhat[i] = s;
        }
        return yhat;
    }

    @Override
    public double score(double[][] X, double[] y) {
        double[] yhat = predict(X);
        return VectorStats.r2(y, yhat);
    }

    // Convenience overload used internally
    private static double[] dot(double[][] A, double[] v) {
        return Matrix.dot(A, v);
    }
}
