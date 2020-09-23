package com.touchmediaproductions.xrayornot;

import android.app.Activity;
import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.util.Log;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.TensorOperator;
import org.tensorflow.lite.support.common.TensorProcessor;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp;
import org.tensorflow.lite.support.label.TensorLabel;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

public class MLHelper {


    private static final int MODEL_A = 0;
    private static final int MODEL_B = 1;

    private Context context;

    protected Interpreter tflite;

    private int imageSizeX;
    private int imageSizeY;

    private TensorImage inputImageBuffer;
    private TensorBuffer outputProbabilityBuffer;
    private TensorProcessor probabilityProcessor;

    private float IMAGE_MEAN;
    private float IMAGE_STD;
    private String MODELNAME;

    private float PROBABILITY_MEAN = 0.0f;
    private float PROBABILITY_STD = 1.0f;


    private List<String> labels;

    //Classification underway flag makes sure that classification does not run again till its finished
    private boolean CLASSIFICATION_UNDERWAY = false;

    public MLHelper(Context context, int model) throws IOException {
        this.context = context;

        if(model == MODEL_A){
            //Model a - Quantized
            this.PROBABILITY_MEAN = 0.0f;
            this.PROBABILITY_STD = 255.0f;
            this.IMAGE_MEAN = 0.0f;
            this.IMAGE_STD = 1.0f;
            this.MODELNAME = "model.tflite";
        } else if (model == MODEL_B){
            //Model b - unquantized
            this.PROBABILITY_MEAN = 0.0f;
            this.PROBABILITY_STD = 1.0f;
            this.IMAGE_MEAN = 127.5f;
            this.IMAGE_STD = 127.5f;
            this.MODELNAME = "model_unquant.tflite";
        }

        tflite = new Interpreter(loadmodelfile((Activity) this.context));

    }

    public Prediction runClassification(Bitmap bitmap){
        if(!CLASSIFICATION_UNDERWAY) {
            CLASSIFICATION_UNDERWAY = true;
            Log.i("TFModel", "Classification initiated...");

            int imageTensorIndex = 0;
            int[] imageShape = tflite.getInputTensor(imageTensorIndex).shape();
            imageSizeX = imageShape[1];
            imageSizeY = imageShape[2];
            DataType imageDataType = tflite.getOutputTensor(imageTensorIndex).dataType();

            int probabilityTensorIndex = 0;
            int[] probabilityShape = tflite.getOutputTensor(probabilityTensorIndex).shape();
            DataType probabilityDataType = tflite.getOutputTensor(probabilityTensorIndex).dataType();

            inputImageBuffer = new TensorImage(imageDataType);
            outputProbabilityBuffer = TensorBuffer.createFixedSize(probabilityShape, probabilityDataType);
            probabilityProcessor = new TensorProcessor.Builder().add(getPostprocessNormalizeOp()).build();

            inputImageBuffer = loadImage(bitmap);

            tflite.run(inputImageBuffer.getBuffer(), outputProbabilityBuffer.getBuffer().rewind());

            CLASSIFICATION_UNDERWAY = false;
            return getResult();
        } else {
            Log.i("TFModel", "Classification is already running.");
        }
        return null;
    }

    private MappedByteBuffer loadmodelfile(Activity activity) throws IOException {
        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(MODELNAME);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startoffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startoffset, declaredLength);
    }

    private TensorImage loadImage(final Bitmap bitmap){
        inputImageBuffer.load(bitmap);
        int cropSize = Math.min(bitmap.getWidth(), bitmap.getHeight());
        ImageProcessor imageProcessor =
                new ImageProcessor.Builder()
                        .add(new ResizeWithCropOrPadOp(cropSize, cropSize))
                        .add(new ResizeOp(imageSizeX, imageSizeY, ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
                        .add(getPreprocessNormalizeOp())
                        .build();
        return imageProcessor.process(inputImageBuffer);
    }

    private TensorOperator getPreprocessNormalizeOp() {
        return new NormalizeOp(IMAGE_MEAN, IMAGE_STD);
    }
    private TensorOperator getPostprocessNormalizeOp(){
        return new NormalizeOp(PROBABILITY_MEAN, PROBABILITY_STD);
    }

    private Prediction getResult(){
        String result = "";
        try {
            labels = FileUtil.loadLabels(context, "labels.txt");
        } catch (IOException e) {
            e.printStackTrace();
        }
        Map<String, Float> labeledProbability = new TensorLabel(labels, probabilityProcessor.process(outputProbabilityBuffer)).getMapWithFloatValue();
        tflite.close();

        return new Prediction(labeledProbability);
    }

    public static class Prediction {
        private String first = "";
        private String second = "";
        private float firstValue = 0;
        private float secondValue = 0;
        private Map<String, Float> sortedProbabilities;
        public Prediction(Map<String, Float> probabilityMap){
            this.sortedProbabilities = sortByValue(probabilityMap);
            Object[] keyset = sortedProbabilities.keySet().toArray();
            if(sortedProbabilities.size() > 0) {
                this.first = (String) keyset[0];
                this.firstValue = sortedProbabilities.get(this.first) * 100;
            }
            if(sortedProbabilities.size() > 1) {
                this.second = (String) keyset[1];
                this.secondValue = sortedProbabilities.get(this.second) * 100;
            }
        }
        private static Map<String, Float> sortByValue(Map<String, Float> unsortMap) {
            List<Map.Entry<String, Float>> list =
                    new LinkedList<>(unsortMap.entrySet());
            Collections.sort(list, new Comparator<Map.Entry<String, Float>>() {
                public int compare(Map.Entry<String, Float> o1,
                                   Map.Entry<String, Float> o2) {
                    return (o2.getValue()).compareTo(o1.getValue());
                }
            });
            Map<String, Float> sortedMap = new LinkedHashMap<>();
            for (Map.Entry<String, Float> entry : list) {
                sortedMap.put(entry.getKey(), entry.getValue());
            }
            return sortedMap;
        }
        public String getFirst() {
            return first;
        }
        public String getSecond() {
            return second;
        }
        public float getFirstValue() {
            return firstValue;
        }
        public float getSecondValue() {
            return secondValue;
        }
        @Override
        public String toString(){
            return this.getFirst() + " | " + (int) this.getFirstValue() + "%\n"
                    + this.getSecond() + " | " + (int) this.getSecondValue() + "%";
        }
    }

}
