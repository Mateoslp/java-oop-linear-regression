package edu.eafit.oop.lr;

import java.util.Arrays;

/** Standard (z-score) scaler: mean=0, std=1 per feature. */
public class StandardScaler {
    private double[] means;
    private double[] stds;
    private boolean fitted = false;

    public void fit(double[][] X) {
        int m = X.length;
        if (m == 0) throw new IllegalArgumentException("Empty X");
        int n = X[0].length;
        means = new double[n];
        stds  = new double[n];
        // means
        for (int j=0; j<n; j++) {
            double s = 0;
            for (int i=0; i<m; i++) s += X[i][j];
            means[j] = s / m;
        }
        // stds
        for (int j=0; j<n; j++) {
            double s2 = 0;
            for (int i=0; i<m; i++) {
                double d = X[i][j] - means[j];
                s2 += d*d;
            }
            stds[j] = Math.sqrt(s2 / Math.max(1, m-1));
            if (stds[j] == 0) stds[j] = 1.0; // avoid division by zero
        }
        fitted = true;
    }

    public double[][] transform(double[][] X) {
        if (!fitted) throw new IllegalStateException("Call fit() first.");
        int m = X.length;
        if (m == 0) return new double[0][0];
        int n = X[0].length;
        double[][] Z = new double[m][n];
        for (int i=0; i<m; i++) {
            for (int j=0; j<n; j++) {
                Z[i][j] = (X[i][j] - means[j]) / stds[j];
            }
        }
        return Z;
    }

    public double[] getMeans() { return Arrays.copyOf(means, means.length); }
    public double[] getStds()  { return Arrays.copyOf(stds, stds.length); }
}
