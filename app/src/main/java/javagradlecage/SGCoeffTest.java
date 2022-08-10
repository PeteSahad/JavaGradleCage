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
import org.apache.commons.math3.stat.descriptive.moment.Kurtosis;

import com.jlibrosa.audio.JLibrosa;
import com.jlibrosa.audio.exception.FileFormatNotSupportedException;
import com.jlibrosa.audio.wavFile.WavFileException;

import com.github.psambit9791.jdsp.signal.Convolution;

public class SGCoeffTest {
    public static void main(String args[]) throws IOException, WavFileException, FileFormatNotSupportedException{

        //input audio file
        String cough_sample = "data/audio_test/Wu0427/cough_0.wav";
        
        //extraction parameters
        int sample_rate = 48000;
        int N_MFCC = 26;

        //parameters for frame
        int n_fft = 2048;
        int hop_length = 2048;

        //get original audio features
        JLibrosa librosa = new JLibrosa();
        float[] original_features = librosa.loadAndRead(cough_sample, sample_rate, -1);

        //debug
        //System.out.println("1. "+original_features[0]+", last: "+original_features[original_features.length-1]);

        //get mfcc
        float[][] librosa_mfcc = librosa.generateMFCCFeatures(original_features, sample_rate, N_MFCC);
        Table mfcc = convertMatrixToTable("mfcc", convertFloatsToDoubles2D(librosa_mfcc));

        //get MFCC delta and delta-delta features
        Table mfcc_delta = delta(mfcc, mfcc.columnCount(), 1, -1);
        Table mfcc_delta_delta = delta(mfcc, mfcc.columnCount(), 2, -1);

        //stack mfcc, mfcc_delta, mfcc_delta_delta
        Table mfcc_vec = renameColums(mfcc.transpose());
        mfcc_vec.setName("mfcc_delta_stack");
        mfcc_vec.concat(renameColums(mfcc_delta.transpose()));
        mfcc_vec.concat(renameColums(mfcc_delta_delta.transpose()));
        mfcc_vec = mfcc_vec.transpose();
        renameColums(mfcc_vec);

        //debug
        /* System.out.println(mfcc_vec.shape());
        System.out.println(mfcc_vec.print());
        System.out.println(mfcc_vec.column(0).size());
        System.out.println(mfcc_vec.column(0).print()); */

        /*
         * get frames from cough audio file
         */

        //get frames from original audio features
        double[][] d_frames = frames(convertFloatsToDoubles(original_features), n_fft, hop_length);
        Table frames = convertMatrixToTable("frames", d_frames);

        //prepare frames for applying hamming window
        frames = frames.transpose();
        TableConverter conv_frames = new TableConverter(frames);
        d_frames = conv_frames.doubleMatrix(); 

        //create hamming window for frame size
        double[] hamming = hammingWindow(frames.columnCount());

        /*
         * loop over every frame and:
         * 
         * - apply hamming window
         * - get zero crossing rate
         * - get kurtosis
         * - get LogEnergy
         */

        Table vec = Table.create("vec");

        for (int i = 0; i < d_frames.length; i++) {
            //apply hamming window to frames
            RealVector vector_a = new ArrayRealVector(hamming);
            RealVector vector_b = new ArrayRealVector(d_frames[i]);
            RealVector result = vector_a.ebeMultiply(vector_b);
            double[] d_result = result.toArray();

            //debug applied hamming window
            //System.out.println("Window "+i+": 1. "+d_result[0]+", last ("+d_result.length+"): "+d_result[d_result.length-1]);

            //zero crossing rate
            double zcr = zero_crossing_rate(sample_rate, d_frames[i]);
            
            //log energy
            double logE = logEnergy(d_frames[i]);
            
            //kurtosis
            Kurtosis kurtosis = new Kurtosis();
            kurtosis.setData(d_frames[i]);
            double kurt = kurtosis.evaluate();

            //build feature vector
            double[] d_vec = {kurt, zcr, logE};

            vec.addColumns(DoubleColumn.create("frame_"+i, d_vec));
            
        }

        //debug vec
        System.out.println(vec.transpose().print());

    }

    /*
     * rename columns:
     * 
     * after transposing a table the column names are lost and set to 0 ... n. 
     * This prevents the issue concating two transposed tables b/c the colnames are redundant.
     * 
     * This function just renames the columns of the given table to the provided "col_names"
     * with increment suffix. e.g. col_0, ..., col_n 
     */
    public static Table renameColums(Table table){

        for (int i = 0; i < table.columnCount(); i++) {
            table.column(i).setName(table.name()+"_"+i);
        }

        return table;
    }

    public static Table delta(Table data, int width, int order, int axis){

        int polyorder = order;
        int deriv = order;
        int delta = 1;
        double cval = 0.0;
        String mode = "interp";

        return savgol_filter(data, width, polyorder, deriv, delta, axis, mode, cval);
    }

    public static Table savgol_filter(Table x, int window_length, int polyorder, int deriv, double delta, int axis, String mode, double cval){

        String[] modes = {"mirror", "constant", "nearest", "interp", "wrap"};
        if(!Arrays.stream(modes).anyMatch(mode::equals)){
            throw new IllegalArgumentException("mode must be mirror, constant, nearest, wrap or interp."); 
        }

        Table coeffs = savgol_coeffs(window_length, polyorder, deriv, delta, -1, "conv");

        int x_size = x.columnCount() * x.rowCount();
           
        Table y = Table.create(); //just for naming the table and cols

        String tableName;

        if(polyorder == 1){
            tableName = "delta";
        }
        else if(polyorder > 1){
            tableName = "delta_delta";
        }
        else{
            throw new IllegalArgumentException("Polyorder must by > 0");
        }

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

        y.setName(tableName);

        //System.out.println(y.column(0).print());
        //System.out.println("returned Table with shape: "+y.shape());

        return y;
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
        A = matrixPower(x_column, order_column);

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

        //System.out.println(results.print());

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

    public static DoubleColumn convertArrayToColumn(String name, double[] array){
        DoubleColumn column = DoubleColumn.create(name, array);

        return column;
    }

    public static Table convertMatrixToTable(String name, double[][] matrix){
        Table table = Table.create(name);

        for (int i = 0; i < matrix.length; i++) {
            table.addColumns(convertArrayToColumn(name+"_"+i, matrix[i]));
        }

        return table.transpose();       // I believe it's wrong to transpose here. It generates the weird data that createPythonLibrosaDeltaValues alters to get librosa data. I think thats incorrect.
    }

    public static double[] convertFloatsToDoubles(float[] input){
        if (input == null)
        {
            return null; // Or throw an exception - your choice
        }
        double[] output = new double[input.length];
        for (int i = 0; i < input.length; i++)
        {
            output[i] = input[i];
        }
        return output;
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

    public static float[][] matrixSubtract(float[][] source, float[] substract){
        float[][] results = new float[source.length][source[0].length];

        for (int i = 0; i < source.length; i++) {
            for (int j = 0; j < source[i].length; j++) {
                results[i][j] = (source[i][j] - substract[i]);
            }
        }

        return results;
    }

    public static Table matrixPower(DoubleColumn base, DoubleColumn power){ // ---> vectorPower ...

        Table computed_matrix = Table.create();

        Double[][] exp_matrix = new Double[base.size()][power.size()];

        for(int i = 0; i < base.size(); i++){
            for (int j = 0; j < power.size(); j++) {
                Double exp = Math.pow(base.get(i), power.get(j));
                exp_matrix[i][j] = exp;
            }
        }

        for (int i = 0; i < exp_matrix.length; i++) {
            DoubleColumn a_column = DoubleColumn.create("col_"+i, exp_matrix[i]);
            computed_matrix.addColumns(a_column);
        }
        
        return computed_matrix;
    }

    /* public static Table vectorMultiplication(Table vector_a, Table vector_b){

        Table computed_vector = Table.create();

        Double[] multiplies_vector = new Double[vector_a.rowCount()];

        TableConverter conv_vector_a = new TableConverter(vector_a);
        double[][] d_vector_a = conv_vector_a.doubleMatrix();
        TableConverter conv_vector_b = new TableConverter(vector_b);
        double[][] d_vector_b = conv_vector_b.doubleMatrix();

        for(int i = 0; i < d_vector_a.length; i++){
            multiplies_vector[i] = d_vector_a[0][i] * d_vector_b[0][i];
        }
                                                                //weiter machen!
        for (int i = 0; i < multiplies_vector.length; i++) {
            DoubleColumn column = DoubleColumn.create("col_"+i, multiplies_vector[i]);
            computed_vector.addColumns(column);
        }
        
        return computed_vector;
    } */

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

    /*
     * from MFCC.java
     */
    public static double[][] frames(double[] ypad, int n_fft, int hop_length){
		final int n_frames = 1 + (ypad.length - n_fft) / hop_length;
		double[][] winFrames = new double[n_fft][n_frames];
		for (int i = 0; i < n_fft; i++){
			for (int j = 0; j < n_frames; j++){
				winFrames[i][j] = ypad[j*hop_length+i];
			}
		}
		return winFrames;
	}

    public static double[] hannWindow(int n_fft){
		//Return a Hann window for even n_fft.
		//The Hann window is a taper formed by using a raised cosine or sine-squared
		//with ends that touch zero.
		double[] win = new double[n_fft];
		for (int i = 0; i < n_fft; i++){
			win[i] = 0.5 - 0.5 * Math.cos(2.0*Math.PI*i/n_fft);
		}
		return win;
	}

    public static double[] hammingWindow(int n_fft){
		//Return a Hamming window for even n_fft.
		double[] win = new double[n_fft];
		for (int i = 0; i < n_fft; i++){
			win[i] = 0.54 - 0.46 * Math.cos(2.0*Math.PI*i/(n_fft-1));
		}
		return win;
	}

    /*
     * end MFCC.java
     */

    /*
     * from https://github.com/gast-lib/gast-lib/blob/master/library/src/root/gast/audio/processing/ZeroCrossing.java
     * 
     * adapted to madhu data
     */
    public static double zero_crossing_rate(int sampleRate, double[] audioData)
    {
        int numSamples = audioData.length;
        int numCrossing = 0;
        for (int p = 0; p < numSamples-1; p++)
        {
            if ((audioData[p] > 0 && audioData[p + 1] <= 0) || 
                (audioData[p] < 0 && audioData[p + 1] >= 0))
            {
                numCrossing++;
            }
        }

        /* float numSecondsRecorded = (float)numSamples/(float)sampleRate;
        float numCycles = numCrossing/2;
        float frequency = numCycles/numSecondsRecorded; */
        double frequency = 1 + numCrossing / (double)numSamples-1;

        return frequency;
    }

    /*
     * translated to java from madhu python
     */
    public static double logEnergy(double[] audio){

        double eps = 0.0001;
        double N = audio.length;

        RealVector audio_vector = new ArrayRealVector(audio);
        RealVector energy_vector = audio_vector.ebeMultiply(audio_vector);
        double[] energy = energy_vector.toArray();

        double logE = 10 * Math.log10(eps + Arrays.stream(energy).sum() / N);

        return logE;
    }

}
