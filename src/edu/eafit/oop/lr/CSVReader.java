package edu.eafit.oop.lr;

import java.io.*;
import java.util.*;

/** Minimal CSV loader that accepts comma or semicolon as delimiter. */
public class CSVReader {

    public static class DataFrame {
        public final String[] headers;
        public final double[][] data; // rows x cols

        DataFrame(String[] headers, double[][] data) {
            this.headers = headers;
            this.data = data;
        }

        public int rows() { return data.length; }
        public int cols() { return headers.length; }

        public int indexOfColumn(String name) {
            for (int i=0; i<headers.length; i++) {
                if (headers[i].equalsIgnoreCase(name)) return i;
            }
            return -1;
        }

        public double[] columnAsVector(int colIdx) {
            double[] v = new double[rows()];
            for (int i=0; i<rows(); i++) v[i] = data[i][colIdx];
            return v;
        }

        public double[][] featuresExcept(int skipCol) {
            double[][] X = new double[rows()][cols()-1];
            for (int i=0; i<rows(); i++) {
                int t = 0;
                for (int j=0; j<cols(); j++) {
                    if (j == skipCol) continue;
                    X[i][t++] = data[i][j];
                }
            }
            return X;
        }

        public DataFrame[] trainTestSplit(double trainRatio, long seed) {
            int n = rows();
            Integer[] idx = new Integer[n];
            for (int i=0; i<n; i++) idx[i] = i;
            Random r = new Random(seed);
            Collections.shuffle(Arrays.asList(idx), r);
            int nTrain = Math.max(1, (int)Math.round(n * trainRatio));

            double[][] train = new double[nTrain][cols()];
            double[][] test  = new double[n - nTrain][cols()];

            for (int i=0; i<nTrain; i++) train[i] = data[idx[i]];
            for (int i=nTrain; i<n; i++) test[i - nTrain] = data[idx[i]];

            return new DataFrame[] {
                new DataFrame(headers, train),
                new DataFrame(headers, test)
            };
        }
    }

    public static DataFrame read(String path) throws IOException {
        try (BufferedReader br = new BufferedReader(new FileReader(path))) {
            String headerLine = br.readLine();
            if (headerLine == null) throw new IOException("Empty file");
            String delim = headerLine.contains(";") ? ";" : ",";
            String[] headers = Arrays.stream(headerLine.trim().split("\\s*" + java.util.regex.Pattern.quote(delim) + "\\s*"))
                                     .toArray(String[]::new);
            List<double[]> rows = new ArrayList<>();
            String line;
            int expectedCols = headers.length;
            int lineNo = 1;
            while ((line = br.readLine()) != null) {
                lineNo++;
                line = line.trim();
                if (line.isEmpty()) continue;
                String[] parts = line.split(java.util.regex.Pattern.quote(delim));
                if (parts.length != expectedCols) {
                    throw new IOException("Row " + lineNo + " has " + parts.length +
                        " columns, expected " + expectedCols);
                }
                double[] row = new double[expectedCols];
                for (int i=0; i<expectedCols; i++) {
                    String s = parts[i].trim().replace("\"","");
                    if (s.isEmpty() || s.equalsIgnoreCase("NA")) {
                        row[i] = Double.NaN;
                    } else {
                        row[i] = Double.parseDouble(s);
                    }
                }
                rows.add(row);
            }
            // Drop rows with NaN
            List<double[]> clean = new ArrayList<>();
            for (double[] r : rows) {
                boolean ok = true;
                for (double v : r) if (Double.isNaN(v)) { ok = false; break; }
                if (ok) clean.add(r);
            }
            double[][] data = clean.toArray(new double[0][]);
            return new DataFrame(headers, data);
        }
    }
}
