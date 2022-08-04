package javagradlecage;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import com.jlibrosa.audio.JLibrosa;
import com.jlibrosa.audio.exception.FileFormatNotSupportedException;
import com.jlibrosa.audio.wavFile.WavFileException;

public class FeatureExtraction {

    //feature extraction parameters
    static Integer  N_MFCC = 26;
    static Integer  N_frame = 2048;//Math.pow(2, 11); //2048 nFFT
    static Integer  B = 1;
    static Integer  Avg = 1;

    static Integer  M_Mels = 128;

    //input sample rate, default audio duration (for JLibrosa.loadAndRead): -1, for unspecified duration
    static Integer sr = 48000;
    static Integer defaultAudioDuration = -1;

    static int hop_length = 512;//2048;

    //read original audio features from source audio files
    public static float[][] getAudioFeatures(File[] audio_files) throws IOException, WavFileException, FileFormatNotSupportedException{

        float[][] audio_features = new float[audio_files.length][];

        //sort files by name (numeric file name extension)
        Arrays.sort(audio_files, (a, b) -> getFileId(a).compareTo(getFileId(b)));

        //get audio features for every audio file in source directory
        for(int i=0; i< audio_files.length; i++){

            JLibrosa librosa = new JLibrosa();

            audio_features[i] = librosa.loadAndRead(audio_files[i].toString(), sr, defaultAudioDuration);

            //console output
            System.out.println( "Features: [File: "+audio_files[i]+
                                "] size: "+audio_features[i].length+
                                " First value: "+audio_features[i][0]+
                                " Last value: "+audio_features[i][audio_features[i].length-1]
            );
        
        }

        return audio_features;
    }

    public static float[] getAudioFeaturesFlat(File[] audio_files) throws IOException, WavFileException, FileFormatNotSupportedException{

        float[][] audio_features = new float[audio_files.length][];

        //sort files by name (numeric file name extension)
        Arrays.sort(audio_files, (a, b) -> getFileId(a).compareTo(getFileId(b)));

        int totalSize = 0;

        //get audio features for every audio file in source directory
        for(int i=0; i< audio_files.length; i++){

            JLibrosa librosa = new JLibrosa();

            audio_features[i] = librosa.loadAndRead(audio_files[i].toString(), sr, defaultAudioDuration);

            //console output
            System.out.println( "Features: [File: "+audio_files[i]+
                                "] size: "+audio_features[i].length+
                                " First value: "+audio_features[i][0]+
                                " Last value: "+audio_features[i][audio_features[i].length-1]
            );

            totalSize += audio_features[i].length;
        
        }

        float[] flattedFeatures = new float[totalSize];

        int index = 0;
        for (int i = 0; i < audio_features.length; i++) {
            for (int j = 0; j < audio_features[i].length; j++) {
                flattedFeatures[index] = audio_features[i][j];
                index++;
            }
        }

        return flattedFeatures;
    }

    public static float[] delta_smooth(float[] data, int nl, int nr, int order){
        SGFilter filter = new SGFilter(nl, nr);

        //float[] smoothed_data;
            
        System.out.println("\n----- sgfilter coeffs -----");
        double[] coeffs = SGFilter.computeSGCoefficients(nl, nr, order);
        for (int i = 0; i < coeffs.length; i++) {
            System.out.print(coeffs[i]+", ");
        }

        return filter.smooth(data, nl, nr, coeffs);
    }

    private static Integer getFileId(File path){
        String reg = path.toString().split("_")[2];
        Integer id = Integer.parseInt(reg.split(".wav")[0]);

        return id;
    }
}
