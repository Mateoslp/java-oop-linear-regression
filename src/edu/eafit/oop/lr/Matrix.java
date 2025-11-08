package edu.eafit.oop.lr;

import java.util.Arrays;

/** Minimal matrix utilities (no external libraries). */
public class Matrix {

    public static double[][] transpose(double[][] A) {
        int m = A.length, n = A[0].length;
        double[][] T = new double[n][m];
        for (int i=0; i<m; i++)
            for (int j=0; j<n; j++)
                T[j][i] = A[i][j];
        return T;
    }

    public static double[][] dot(double[][] A, double[][] B) {
        int m = A.length, n = A[0].length, p = B[0].length;
        if (B.length != n) throw new IllegalArgumentException("A.cols != B.rows");
        double[][] C = new double[m][p];
        for (int i=0; i<m; i++) {
            for (int k=0; k<p; k++) {
                double s = 0;
                for (int j=0; j<n; j++) s += A[i][j] * B[j][k];
                C[i][k] = s;
            }
        }
        return C;
    }

    public static double[] dot(double[][] A, double[] v) {
        int m = A.length, n = A[0].length;
        if (v.length != n) throw new IllegalArgumentException("A.cols != v.length");
        double[] r = new double[m];
        for (int i=0; i<m; i++) {
            double s = 0;
            for (int j=0; j<n; j++) s += A[i][j] * v[j];
            r[i] = s;
        }
        return r;
    }

    public static double[][] add(double[][] A, double[][] B) {
        int m = A.length, n = A[0].length;
        if (B.length != m || B[0].length != n) throw new IllegalArgumentException("size mismatch");
        double[][] C = new double[m][n];
        for (int i=0; i<m; i++)
            for (int j=0; j<n; j++)
                C[i][j] = A[i][j] + B[i][j];
        return C;
    }

    public static double[][] identity(int n) {
        double[][] I = new double[n][n];
        for (int i=0; i<n; i++) I[i][i] = 1.0;
        return I;
    }

    /** Gauss-Jordan inverse with partial pivoting. */
    public static double[][] inverse(double[][] A) {
        int n = A.length;
        if (A[0].length != n) throw new IllegalArgumentException("Matrix must be square");
        double[][] aug = new double[n][2*n];
        for (int i=0; i<n; i++) {
            for (int j=0; j<n; j++) aug[i][j] = A[i][j];
            aug[i][n+i] = 1.0;
        }
        // Forward elimination
        for (int col=0; col<n; col++) {
            // pivot
            int pivot = col;
            double max = Math.abs(aug[pivot][col]);
            for (int r=col+1; r<n; r++) {
                double v = Math.abs(aug[r][col]);
                if (v > max) { max = v; pivot = r; }
            }
            if (Math.abs(aug[pivot][col]) < 1e-12) {
                throw new IllegalArgumentException("Singular matrix (or near-singular).");
            }
            // swap
            if (pivot != col) {
                double[] tmp = aug[pivot];
                aug[pivot] = aug[col];
                aug[col] = tmp;
            }
            // scale to 1
            double div = aug[col][col];
            for (int j=0; j<2*n; j++) aug[col][j] /= div;
            // eliminate others
            for (int r=0; r<n; r++) {
                if (r == col) continue;
                double factor = aug[r][col];
                for (int j=0; j<2*n; j++) {
                    aug[r][j] -= factor * aug[col][j];
                }
            }
        }
        // Extract inverse
        double[][] inv = new double[n][n];
        for (int i=0; i<n; i++) {
            System.arraycopy(aug[i], n, inv[i], 0, n);
        }
        return inv;
    }
}
