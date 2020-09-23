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

    private final float PROBABILITY_MEAN = 0.0f;
    private static final float PROBABILITY_STD = 255.0f;


    private List<String> labels;

    //Classification underway flag makes sure that classification does not run again till its finished
    private boolean CLASSIFICATION_UNDERWAY = false;

    public MLHelper(Context context, int model) throws IOException {
        this.context = context;

        if(model == MODEL_A){
            //Model A (float? - no quantisation or compression)
            this.IMAGE_MEAN = 0.0f;
            this.IMAGE_STD = 1.0f;
            this.MODELNAME = "model.tflite";
        } else if (model == MODEL_B){
            //Model B (quant? - as model B underwent default tf quantisation)
            this.IMAGE_MEAN = 127.5f;
            this.IMAGE_STD = 127.5f;
            this.MODELNAME = "model_unquant.tflite";
        }

        tflite = new Interpreter(loadmodelfile((Activity) this.context));

    }

    public String runClassification(Bitmap bitmap){
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
        return "";
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

    private String getResult(){
        String result = "";
        try {
            labels = FileUtil.loadLabels(context, "labels.txt");
        } catch (IOException e) {
            e.printStackTrace();
        }
        Map<String, Float> labeledProbability = new TensorLabel(labels, probabilityProcessor.process(outputProbabilityBuffer)).getMapWithFloatValue();
        float maxValueInMap = (Collections.max(labeledProbability.values()));

        for (Map.Entry<String, Float> entry : labeledProbability.entrySet()){
            if (entry.getValue() == maxValueInMap){
                result = entry.getKey();
            }
        }
        tflite.close();
        return result;
    }

}
