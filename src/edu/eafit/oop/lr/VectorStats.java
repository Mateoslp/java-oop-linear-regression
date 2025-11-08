package edu.eafit.oop.lr;

public class VectorStats {

    public static double mean(double[] v) {
        double s=0; for (double x : v) s += x; return s / v.length;
    }

    public static double mse(double[] y, double[] yhat) {
        double s=0; int n=y.length;
        for (int i=0; i<n; i++) {
            double d = yhat[i] - y[i];
            s += d*d;
        }
        return s / n;
    }

    public static double mae(double[] y, double[] yhat) {
        double s=0; int n=y.length;
        for (int i=0; i<n; i++) s += Math.abs(yhat[i] - y[i]);
        return s / n;
    }

    /** R^2 = 1 - SSE/SST */
    public static double r2(double[] y, double[] yhat) {
        double ym = mean(y);
        double sse=0, sst=0;
        for (int i=0; i<y.length; i++) {
            double e = yhat[i] - y[i];
            sse += e*e;
            double d = y[i] - ym;
            sst += d*d;
        }
        if (sst == 0) return 0.0;
        return 1.0 - (sse / sst);
    }
}
