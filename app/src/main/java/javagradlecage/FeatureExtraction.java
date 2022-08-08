package javagradlecage;

import java.io.*;
import java.util.Arrays;
import com.jlibrosa.audio.JLibrosa;
import com.jlibrosa.audio.exception.FileFormatNotSupportedException;
import com.jlibrosa.audio.wavFile.WavFileException;

public class FeatureExtraction {

    //feature extraction parameters
    static Integer  N_MFCC = 26;
    static Integer  N_frame = 2048;//Math.pow(2, 11); //2048 nFFT
    
    //input sample rate, default audio duration (for JLibrosa.loadAndRead): -1, for unspecified duration
    static Integer sr = 48000;
    static Integer defaultAudioDuration = -1;

    //static int hop_length = 512;//2048;
    //static Integer  B = 1;
    //static Integer  Avg = 1;
    //static Integer  M_Mels = 128;

    
    
    /*
     * Helper functions for original audio feature extraction (audio source files --> wav)
     */
    
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

    private static Integer getFileId(File path){
        String reg = path.toString().split("_")[2];
        Integer id = Integer.parseInt(reg.split(".wav")[0]);

        return id;
    }
}
