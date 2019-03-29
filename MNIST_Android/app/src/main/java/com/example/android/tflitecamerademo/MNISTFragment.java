package com.example.android.tflitecamerademo;

import android.app.Fragment;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.support.annotation.Nullable;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.TextView;

import java.io.IOException;
import java.io.InputStream;

public class MNISTFragment extends Fragment {
    private static final String TAG = "MNISTFragment";

    static final String IMAGE_PREFIX = "mnist_set/image_";

    FrameLayout mFrame;
    ImageView mImage;
    TextView mText;

    MnistScanRunner runner;

    int count;

    @Nullable
    @Override
    public View onCreateView(LayoutInflater inflater, @Nullable ViewGroup container, Bundle savedInstanceState) {
        return inflater.inflate(R.layout.fragment_mnist, container, false);
    }

    @Override
    public void onViewCreated(View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);
        mFrame = view.findViewById(R.id.control);
        mImage = view.findViewById(R.id.mnist_container);
        mText = view.findViewById(R.id.text);

        mText.setText("Predict Result: %d");

        runner = new MnistScanRunner(getActivity());

        mText.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Bitmap mnist = loadImage(count++);
                if (count == 9) {
                    count = 0;
                }
                if (mnist != null) {
                    mImage.setImageBitmap(mnist);
                }
                mText.setText("Predict Result:" + detectValue(mnist));
            }
        });
    }

    private Bitmap loadImage(int num) {
        String image_name = IMAGE_PREFIX + String.valueOf(num) + ".png";
        try {
            InputStream ins = getActivity().getAssets().open(image_name);
            Bitmap result = BitmapFactory.decodeStream(ins);
            return result;
        } catch (IOException exp) {
            Log.e(TAG, "loadImage: " + exp.getMessage());
        }
        return null;
    }

    private String detectValue(Bitmap bitmap) {
        try {
            runner.loadModel();
            float[][] result = runner.internalProcess(bitmap);
            String value = argmaxLabel(result);
            Log.d(TAG, "detectValue: " + value);
            return value;
        } catch (Exception exp) {
            Log.d(TAG, "detectValue: " + exp.getMessage());
        }
        return "";
    }

    private String argmaxLabel(float[][] networkOutput){
        int maxIdx = -1;
        float maxValue = -1;

        for(int i = 0 ; i < 10 ; i++){
            if(networkOutput[0][i] > maxValue){
                maxValue = networkOutput[0][i];
                maxIdx = i;
            }
        }

        return String.valueOf(maxIdx);
    }
}
