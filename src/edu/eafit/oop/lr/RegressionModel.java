package edu.eafit.oop.lr;

public interface RegressionModel {
    /** Train the model. */
    void fit(double[][] X, double[] y);

    /** Predict outputs for rows in X. */
    double[] predict(double[][] X);

    /** Score the model on (X, y). Returns R^2 for regression. */
    double score(double[][] X, double[] y);

    /** Optional utility to expose scaled data (handled by StandardScaler class in this project). */
}
