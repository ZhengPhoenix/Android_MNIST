package com.example.android.tflitecamerademo;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.os.SystemClock;
import android.util.Log;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

public class MnistScanRunner{
    private static final String TAG = "MnistScanRunner";

    private static final String MODEL_FULL_CONNECT = "fc_softmax.tflite";

    private static final String MODEL_FULL_CNN_QUANTIZATION = "cnn.tflite";

    private static final String MODEL_FULL_CNN = "cnn_ori.tflite";

    private static final int DIM_BATCH_SIZE = 1;

    private static final int DIM_PIXEL_SIZE = 3;

    static final int DIM_IMG_SIZE_X = 28;
    static final int DIM_IMG_SIZE_Y = 28;

    /** A ByteBuffer to hold image data, to be feed into Tensorflow Lite as inputs. */
    private ByteBuffer imgData = null;

    /* Preallocated buffers for storing image data in. */
    private int[] intValues = new int[DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y];

    /** An instance of the driver class to run model inference with Tensorflow Lite. */
    private Interpreter points_network;


    /** An array to hold inference results, to be feed into Tensorflow Lite as outputs. */
    private float[][] mnistOutput;
    
    private Context mContext;

    public MnistScanRunner(Context context){
        mContext = context;
    }

    public void loadModel() {
        try {
            MappedByteBuffer mappedByteBuffer = loadModelFile(mContext);
            points_network = new Interpreter(mappedByteBuffer, 4);

            imgData = ByteBuffer.allocateDirect(
                    DIM_BATCH_SIZE * DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y * 4);
            imgData.order(ByteOrder.nativeOrder());
            mnistOutput = new float[1][10];
        } catch (Exception exp) {
            Log.e(TAG, "loadModel: " + exp.getMessage());
        }
    }

    private MappedByteBuffer loadModelFile(Context activity) throws IOException {
        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(MODEL_FULL_CNN_QUANTIZATION);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    public float[][] internalProcess(Bitmap data) {
        if(points_network == null){
            Log.d(TAG,"Image classifier has not been initialized; Skipped.");
            return new float[1][10];
        }

        long startTime = SystemClock.uptimeMillis();
        float[][] result = null;
        data = data.copy(Bitmap.Config.RGB_565, false);
        result = doPredict(data);
        long endTime = SystemClock.uptimeMillis();
        Log.d(TAG,"Timecost to run model inference: " + Long.toString(endTime - startTime));
        return result;
    }

    private synchronized float[][] doPredict(Bitmap bitmap) {
        if (bitmap == null) {
            return null;
        }

        Bitmap small_bitmap = Bitmap.createScaledBitmap(bitmap, 28, 28, true);
        convertBitmapToByteBuffer(small_bitmap);

        points_network.run(imgData, mnistOutput);
        return mnistOutput;
    }


    /** Writes Image data into a {@code ByteBuffer}. */
    private void convertBitmapToByteBuffer(Bitmap bitmap) {
        if (imgData == null) {
            return;
        }
        imgData.rewind();
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        // Convert the image to floating point.
        int pixel = 0;
        long startTime = SystemClock.uptimeMillis();
        for (int i = 0; i < DIM_IMG_SIZE_X; ++i) {
            for (int j = 0; j < DIM_IMG_SIZE_Y; ++j) {
                final int val = intValues[pixel++];

                imgData.putFloat((val & 0xFF));
            }
        }
        long endTime = SystemClock.uptimeMillis();
        Log.d(TAG,"Timecost to convert bitmap into ByteBuffer: " + Long.toString(endTime - startTime));
    }
}
