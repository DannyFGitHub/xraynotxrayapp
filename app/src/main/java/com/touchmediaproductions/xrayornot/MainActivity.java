package com.touchmediaproductions.xrayornot;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.net.Uri;
import android.os.Bundle;
import android.os.SystemClock;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.SwitchCompat;

import java.io.IOException;

public class MainActivity extends AppCompatActivity {

    private static final int MODEL_A = 0;
    private static final int MODEL_B = 1;

    ImageView imageView;
    Button btnClassify;
    TextView classifyText;
    TextView predictionDetails;
    SwitchCompat modelSwitch;

    MLHelper mlHelper;

    Uri imageuri;
    private Bitmap bitmap;
    private ProgressBar loadingCircle;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        imageView = findViewById(R.id.image);
        btnClassify = findViewById(R.id.btn_classify);
        classifyText = findViewById(R.id.result);
        predictionDetails = findViewById(R.id.predictiondetails);
        modelSwitch = findViewById(R.id.toggle_modelAorB);

        loadingCircle = findViewById(R.id.loading);
        loadingCircle.setIndeterminate(true);
        loadingCircle.setVisibility(View.GONE);

        imageView.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View v) {
                Intent intent = new Intent();
                intent.setType("image/*");
                intent.setAction(Intent.ACTION_GET_CONTENT);
                startActivityForResult(Intent.createChooser(intent, "Select Picture"), 12);
            }
        });


        btnClassify.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                runClassification();
            }
        });
    }


    private void runClassification() {
        if (bitmap != null) {
            //If mlHelper is null it has been cleared and is ready to be initiated.
            if (mlHelper == null) {
                loadingCircle.setVisibility(View.VISIBLE);
                new Thread(new Runnable() {
                    @Override
                    public void run() {
                        android.os.Process.setThreadPriority(android.os.Process.THREAD_PRIORITY_BACKGROUND);
                        try {
                            //Get whether model is A or B
                            int modelToUse = 0;
                            if (modelSwitch.isChecked()) {
                                modelToUse = MODEL_A;
                            } else {
                                modelToUse = MODEL_B;
                            }

                            //Prepare the Machine Learning Helper
                            mlHelper = new MLHelper(MainActivity.this, modelToUse);

                            //Run Classification against bitmap image input:
                            final MLHelper.Prediction result = mlHelper.runClassification(bitmap);

                            new Thread(new Runnable() {
                                @Override
                                public void run() {
                                    SystemClock.sleep(1000);
                                    loadingCircle.post(new Runnable() {
                                        @Override
                                        public void run() {
                                            loadingCircle.setVisibility(View.GONE);
                                        }
                                    });
                                }
                            }).start();

                            //Display result
                            classifyText.post(new Runnable() {
                                @Override
                                public void run() {
                                    classifyText.setText(result.getFirst());
                                    //If XRay then white else Not X-Ray another color
                                    if (result.getFirst().contains("X-Ray")) {
                                        classifyText.setTextColor(Color.WHITE);
                                    } else if (result.getFirst().contains("Not X-Ray")) {
                                        classifyText.setTextColor(Color.rgb(255, 165, 0));
                                    }
                                }
                            });

                            predictionDetails.post(new Runnable() {
                                @Override
                                public void run() {
                                    predictionDetails.setText(result.toString());
                                }
                            });

                        } catch (IOException e) {
                            e.printStackTrace();
                        }

                        //Clear mHelper as classification is finished.
                        mlHelper = null;
                    }
                }).start();
            } else {
                Toast toast = new Toast(MainActivity.this);
                toast.makeText(MainActivity.this, "Please choose a photo first.", Toast.LENGTH_SHORT).show();
            }
        }
    }


    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if(requestCode == 12 && resultCode == RESULT_OK && data != null){
            imageuri = data.getData();
            try {
                bitmap = MediaStore.Images.Media.getBitmap(getContentResolver(), imageuri);
                imageView.setImageBitmap(bitmap);
            } catch (IOException e){
                e.printStackTrace();
            }
        }
    }


}