# Chest X-Ray / not Chest X-Ray Model
## Simple Model to determine whether an image classifies as a Chest X-Ray or Not

<p align="center">
  <img width="100" src="screenshots/iconlogo.png">
</p>

A (poc) android application to classify images into category of Chest X-Rays or not Chest X-Rays.

Use case intended for when a user provides an image and image input validation is required before processing the image as a Chest X-Ray.

Trained from 500 Chest X-Ray images and 1000 normal random images (500 grayscale and 500 color).

<p align="center">
  <img width="200" src="screenshots/screenshot01.png">
  <img width="200" src="screenshots/screenshot02.png">
  <img width="200" src="screenshots/screenshot03.png">
</p>
API 26 - Oreo

The trained models can be obtained from the assets folder.

|  Type  | Trained on Resolution |  Size |       Model      |
|:------:|:---------------------:|:-----:|:----------------:|
| Float  |        500x500        |  40M  | [model_unquant.tflite](/app/src/main/assets/model_unquant.tflite)|
| Quant  |        500x500        |  12M  | [model.tflite](/app/src/main/assets/model.tflite)|