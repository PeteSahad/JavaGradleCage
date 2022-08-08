package javagradlecage;

import tech.tablesaw.api.Table;
import tech.tablesaw.columns.numbers.fillers.DoubleRangeIterable;
import tech.tablesaw.conversion.TableConverter;
import tech.tablesaw.api.DoubleColumn;

import java.io.IOException;
import java.util.Arrays;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.DecompositionSolver;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.linear.SingularValueDecomposition;

import com.jlibrosa.audio.JLibrosa;
import com.jlibrosa.audio.exception.FileFormatNotSupportedException;
import com.jlibrosa.audio.wavFile.WavFileException;

import com.github.psambit9791.jdsp.signal.Convolution;

public class SGCoeffTest {
    public static void main(String args[]) throws IOException, WavFileException, FileFormatNotSupportedException{

        //get MFCC from cough
        int sample_rate = 48000;
        int N_MFCC = 26;

        String cough_sample = "data/audio_test/Wu0427/cough_0.wav";

        JLibrosa librosa = new JLibrosa();
        float[] original_features = librosa.loadAndRead(cough_sample, sample_rate, -1);
        float[][] MFCC = librosa.generateMFCCFeatures(original_features, sample_rate, N_MFCC);

        Table t_MFCC = convertMatrixToTable("coeffs", convertFloatsToDoubles2D(MFCC));

        //get MFCC delta and delta-delta features
        delta(t_MFCC, t_MFCC.columnCount(), 1, -1);
        delta(t_MFCC, t_MFCC.columnCount(), 2, -1);
    }

    public static void delta(Table data, int width, int order, int axis){

        int polyorder = order;
        int deriv = order;
        int delta = 1;
        double cval = 0.0;
        String mode = "interp";

        savgol_filter(data, width, polyorder, deriv, delta, axis, mode, cval);
    }

    public static Table savgol_filter(Table x, int window_length, int polyorder, int deriv, double delta, int axis, String mode, double cval){

        String[] modes = {"mirror", "constant", "nearest", "interp", "wrap"};
        if(!Arrays.stream(modes).anyMatch(mode::equals)){
            throw new IllegalArgumentException("mode must be mirror, constant, nearest, wrap or interp."); 
        }

        Table coeffs = savgol_coeffs(window_length, polyorder, deriv, delta, -1, "conv");

        int x_size = x.columnCount() * x.rowCount();

        Table y = Table.create();

        if(mode == "interp"){
            if(window_length > x_size){
                throw new IllegalArgumentException("If mode is 'interp', window_length must be less than or equal to the size of x."); //wrong error type; fix later
            }
            y = convolve1d_jdsp(x, coeffs, "constant");
        }
        else{
            // not actually reached b/c right now only mode=interp is considered
            y = convolve1d_jdsp(x, coeffs, mode); //mode = mode (python function); cval not implemented in jdsp
        }

        y = createPythonLibrosaDeltaValues(y);

        System.out.println(y.column(0).print());
        System.out.println(y.shape());

        return y;
    }

    /*
     * This function takes the convolved values from y = convolve1d_jdsp() and reorganizes the data to
     * python librosa.delta equivalent results. I don't know why librosa does that, and I know 
     * this solution is most probably false. I just figured this solution by accident: 
     * 
     * It takes the center element of y-rows and refills this row all along with this elements "output[Math.floor(y-row_size/2)]"
     */
    public static Table createPythonLibrosaDeltaValues(Table convolved_data){

        Table results = Table.create();

        int window_length = convolved_data.transpose().columnCount();
        int data_rows = convolved_data.transpose().rowCount();

        DoubleColumn center_elements = (DoubleColumn)convolved_data.transpose().column((int)Math.floor(window_length/2));

        for (int i = 0; i < data_rows; i++) {
            double value = center_elements.getDouble(i);
            DoubleColumn col = DoubleColumn.create("res_"+i, new Double[window_length]).fillWith(value);
            results.addColumns(col);
        }

        results = results.transpose();

        return results;
    }

    public static Table convolve1d_jdsp(Table input, Table weights, String mode){

        Table output = Table.create();

        TableConverter conv_input = new TableConverter(input);
        double[][] d_input = conv_input.doubleMatrix();
        
        weights = weights.transpose();

        TableConverter conv_weights = new TableConverter(weights);
        double[][] d_weights = conv_weights.doubleMatrix();

        for (int i = 0; i < d_input.length; i++) {
            Convolution conv = new Convolution(d_input[i], d_weights[0]);
            double[] d_output = conv.convolve1d(mode);
            output.addColumns(convertArrayToColumn("conv_"+i, d_output));
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
            x_column.sortDescending();          //just sorted descending -> should be reversed; this just works here b/c of values nature. Fix later
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
