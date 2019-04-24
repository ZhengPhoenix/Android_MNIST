python3 /usr/local/lib/python3.6/site-packages/tensorflow/contrib/lite/python/tflite_convert.py \
--graph_def_file=original_prune/cnn.pb \
--output_file=original_prune/cnn.tflite \
--input_format=TENSORFLOW_GRAPHDEF \
--output_format=TFLITE \
--input_shape=1,28,28,1 \
--input_array=x_PH \
--output_array=out \
--inference_type=FLOAT \
--input_data_type=FLOAT
