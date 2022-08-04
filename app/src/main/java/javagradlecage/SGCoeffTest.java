package javagradlecage;

import tech.tablesaw.api.Table;
import tech.tablesaw.columns.numbers.fillers.DoubleRangeIterable;
import tech.tablesaw.conversion.TableConverter;
import tech.tablesaw.index.Index;

import java.io.IOException;
import java.util.Arrays;
import java.util.Comparator;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.DecompositionSolver;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.linear.SingularValueDecomposition;

import com.jlibrosa.audio.JLibrosa;
import com.jlibrosa.audio.exception.FileFormatNotSupportedException;
import com.jlibrosa.audio.wavFile.WavFileException;

import tech.tablesaw.api.DoubleColumn;
import tech.tablesaw.api.FloatColumn;

import com.github.psambit9791.jdsp.*;
import com.github.psambit9791.jdsp.signal.Convolution;

public class SGCoeffTest {
    public static void main(String args[]) throws IOException, WavFileException, FileFormatNotSupportedException{
        //get savgol coefficients
        int window_length = 49;
        int polyorder = 3;
        int deriv = 0;
        int delta = 1;
        int pos = -1;
        String use = "conv";

        int axis = -1;
        String mode = "interp";
        double cval = 0.0;

        Table coeffs = savgol_coeffs(window_length, polyorder, deriv, delta, pos, use);
        System.out.println(coeffs);

        //get MFCC from cough

        String cough_sample = "data/audio_test/Wu0427/cough_0.wav";

        JLibrosa librosa = new JLibrosa();
        float[] original_features = librosa.loadAndRead(cough_sample, 48000, -1);
        float[][] MFCC = librosa.generateMFCCFeatures(original_features, FeatureExtraction.sr, FeatureExtraction.N_MFCC);

        Table t_MFCC = convertMatrixToTable("coeffs", convertFloatsToDoubles2D(MFCC));

        System.out.println(t_MFCC.shape());
        System.out.println(t_MFCC.column(0).print());

        savgol_filter(t_MFCC, window_length, polyorder, deriv, delta, axis, mode, cval);
    }

    public static Table savgol_filter(Table x, int window_length, int polyorder, int deriv, double delta, int axis, String mode, double cval){

        String[] modes = {"mirror", "constant", "nearest", "interp", "wrap"};
        if(!Arrays.stream(modes).anyMatch(mode::equals)){
            throw new IllegalArgumentException("mode must be mirror, constant, nearest, wrap or interp."); 
        }

        Table coeffs = savgol_coeffs(window_length, polyorder, deriv, delta, -1, "conv");

        System.out.println(coeffs.print());

        int x_size = x.columnCount() * x.rowCount();

        //System.out.println(x_size);

        Table y = Table.create();

        if(mode == "interp"){
            if(window_length > x_size){
                throw new IllegalArgumentException("If mode is 'interp', window_length must be less than or equal to the size of x."); //falscher fehlertyp
            }
            //convolve1d
            //y = convolve1d(x, coeffs, axis, -1, mode="constant", cval=0.0, 0); //coeffs.doubleColumn(0)
            //fit:edge
            
            //convolve1d(x, coeffs, axis, -1, mode="constant", cval=0.0, 0); //testing

            //test with jdsp
            y = convolve1d_jdsp(x, coeffs, "constant");
        }
        else{
            //convolve1d
            y = convolve1d_jdsp(x, coeffs, "constant");
        }

        //debug
        System.out.println(y);

        return y;
    }

    public static Table convolve1d_jdsp(Table input, Table weights, String mode){

        Table output = Table.create();

        TableConverter conv_input = new TableConverter(input);
        double[][] d_input = conv_input.doubleMatrix();

        TableConverter conv_weights = new TableConverter(weights);
        double[][] d_weights = conv_weights.doubleMatrix();

        for (int i = 0; i < d_input.length; i++) {
            Convolution conv = new Convolution(d_input[i], d_weights[0]);
            double[] d_output = conv.convolve1d(mode);
            output.addColumns(convertArrayToColumn("conv_"+i, d_output));
        }

        //System.out.println(output.print());
        
        //Convolution conv = new Convolution(d_input[0], d_weights[0]);
        //double[] d_output = conv.convolve1d("constant");

        //debug
        /* System.out.println("mfcc");
        for (int i = 0; i < d_input.length; i++) {
            System.out.println(d_input[i][0]);
        }
        System.out.println("weights");
        for (int i = 0; i < d_weights.length; i++) {
            System.out.println(d_weights[i][0]);
        }
        System.out.println("convolution");
        System.out.println("conv size="+d_output.length);
        for (int i = 0; i < d_output.length; i++) {
            System.out.println(d_output[i]);
        } */
        //debug end

        return output;
    }

    public static Table convolve1d(Table input, Table weights, int axis, int output, String mode, double cval, int origin){ //void

        weights = reverse1dTable(weights);
        origin = -origin;
        
        System.out.println(weights);

        if(weights.rowCount() % 2 == 0){
            origin -= 1;
        }

        /* double[] correlate1d = correlate1d(input, weights, axis, output, mode, cval, origin);
        System.out.println("Correlated");
        for (int i = 0; i < correlate1d.length; i++) {
            System.out.println(correlate1d[i]);
        }
        System.out.println(correlate1d.length); */

        return correlate1d(input, weights, axis, output, mode, cval, origin);
    }

    public static Table correlate1d(Table input, Table weights, int axis, int output, String mode, double cval, int origin){ //double[]

        /* TableConverter conv_input = new TableConverter(input);
        double[][] d_input = conv_input.doubleMatrix();

        TableConverter conv_weights = new TableConverter(weights);
        double[][] d_weights = conv_weights.doubleMatrix();

        return correlate1d_func(d_input[0], d_weights[0]); */

        //output=None, input=input

        //System.out.println(input.shape());

        Table corr_output = Table.create();
        float[] output_column = new float[input.rowCount()]; 
        Arrays.fill(output_column, 0, input.rowCount(), (float)(0.0));
        for (int i = 0; i < input.columnCount(); i++) {
            FloatColumn t_float_column = FloatColumn.create("zero_"+i, output_column);
            corr_output.addColumns(t_float_column);
        }

        //System.out.println(corr_output.shape());

        return null;
    }

    //zum testen ausm netz. scheint nicht das richtige zu sein
    private static double[] correlate1d_func(double[] input, double[] weights) {
        double[] output = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            double sum = 0.0;
            int radius = weights.length / 2;
            for (int j = 0; j < weights.length; j++) {
                if (i - radius + j < 0) {
                    sum += input[-(i - radius + j) - 1] * weights[j];
                } else if (i - radius + j >= input.length) {
                    sum += input[2 * input.length - (i - radius + j) - 1] * weights[j];
                } else {
                    sum += input[i - radius + j] * weights[j];
                }
            }
            output[i] = sum;
        }

        return output;
    }

    public static Table reverse1dTable(Table table){

        int table_size = table.rowCount();
        int table_order_el = table_size-1;
        String[] order = new String[table_size];
        for (int i = 0; i < order.length; i++) {
            order[i] = Integer.toString(table_order_el--);
        }

        table = table.transpose().reorderColumns(order);

        return table.transpose();
    }

    public static Table savgol_coeffs(int window_length, int polyorder, int deriv, double delta, double pos, String use){

        if(polyorder >= window_length){
            throw new IllegalArgumentException("polyorder must be less than window_length."); 
        }

        int[] div = divmod(window_length, 2);
        int halflen = div[0];
        int rem = div[1];

        if(pos==-1){
            if(rem == 0){
                pos = halflen - 0.5;
            }
            else{
                pos = halflen;
            }
        }

        if(pos < 0 || pos > window_length){
            throw new IllegalArgumentException("pos must be nonnegative and less than window_length."); 
        }

        if(use != "conv" && use != "dot"){
            throw new IllegalArgumentException("use must be conv or dot"); 
        }

        Table coeffs = Table.create();
        Double[] coeff_column = new Double[window_length]; 
        
        if(deriv > polyorder){
            coeffs.addColumns(DoubleColumn.create("zeroes", coeff_column).fillWith(0));
        }

        Table x = Table.create();
        Double[] x_col = new Double[window_length]; 
        DoubleColumn x_column = DoubleColumn.create("x", x_col);

        x.addColumns(x_column.fillWith(DoubleRangeIterable.range(-pos, window_length - pos)));

        if(use == "conv"){
            x_column.sortDescending(); //wahrscheinlich falsche sortierung, da nicht reserved, sondern sortiert nach groeße: wobei es in diesem fall passen sollte
        }
        
        Table order = Table.create();
        Double[] order_col = new Double[polyorder + 1]; 
        DoubleColumn order_column = DoubleColumn.create("order", order_col);
        order.addColumns(order_column.fillWith(DoubleRangeIterable.range(0, polyorder + 1)));
        order = order.transpose();

        Table A = Table.create();
        A = matrix_exp(x_column, order_column);

        Table y = Table.create();
        Double[] y_column = new Double[polyorder + 1]; 
        double facorial = factorial(deriv) / Math.pow(delta, deriv);
        y.addColumns(DoubleColumn.create("zeroes", y_column).fillWith(0).set(deriv, facorial));

        TableConverter convert_A = new TableConverter(A);
        double[][] d_A = convert_A.doubleMatrix();

        TableConverter convert_y = new TableConverter(y.transpose());
        double[][] d_y = convert_y.doubleMatrix();

        double[][] data = d_A;
        RealMatrix matrix = MatrixUtils.createRealMatrix(data);
        SingularValueDecomposition svd = new SingularValueDecomposition(matrix);
        DecompositionSolver ds=svd.getSolver();
        RealVector b = new ArrayRealVector(d_y[0]);
        RealVector v_coeffs = ds.solve(b);
        double[] d_coeffs = v_coeffs.toArray();

        coeffs.addColumns(convertArrayToColumn("coeffs", d_coeffs));

        return coeffs;
    }

    public static DoubleColumn convertArrayToColumn(String name, double[] array){
        DoubleColumn column = DoubleColumn.create(name, array);

        return column;
    }

    public static Table convertMatrixToTable(String name, double[][] matrix){
        Table table = Table.create();

        for (int i = 0; i < matrix.length; i++) {
            table.addColumns(convertArrayToColumn(name+"_"+i, matrix[i]));
        }

        return table.transpose();
    }

    /* public static Table convertDoubleArrayToTable(String name, double[] matrix){
        Table table = Table.create();

        int i = 0;
        for (double column : matrix) {
            table.addColumns(DoubleColumn.create(name+"_"+i, column));
            i++;
        }

        return table;
    } */

    public static double[][] convertFloatsToDoubles2D(float[][] input){
        if (input == null)
        {
            return null; // Or throw an exception - your choice
        }
        double[][] output = new double[input.length][input[0].length];
        for (int i = 0; i < input.length; i++)
        {
            for (int j = 0; j < input[i].length; j++)
            {
                output[i][j] = input[i][j];
            }
        }
        return output;
    }

    public static int[] divmod(int dividend, int divisor){
        int[] result = new int[2];

        int halflen = Math.floorDiv(dividend, divisor);
        int rem = dividend % divisor;

        result[0] = halflen;
        result[1] = rem;

        return result;
    }

    public static double factorial(int factorial){
        double d_factorial = 1.0;
        for (int i = factorial; i != 0; i--) {
            d_factorial=factorial*(int)factorial(factorial-i);
        }

        return d_factorial;
    }

    public static Table matrix_exp(DoubleColumn base, DoubleColumn exponent){

        Table computed_matrix = Table.create();

        Double[][] exp_matrix = new Double[base.size()][exponent.size()];

        for(int i = 0; i < base.size(); i++){
            for (int j = 0; j < exponent.size(); j++) {
                Double exp = Math.pow(base.get(i), exponent.get(j));
                exp_matrix[i][j] = exp;
            }
        }

        for (int i = 0; i < exp_matrix.length; i++) {
            DoubleColumn a_column = DoubleColumn.create("col"+i, exp_matrix[i]);
            computed_matrix.addColumns(a_column);
        }
        
        return computed_matrix;
    }
}
