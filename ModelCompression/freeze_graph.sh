python3 /usr/local/lib/python3.6/site-packages/tensorflow/python/tools/freeze_graph.py \
--input_meta_graph=CNN/cnn_quantized.ckpt.meta \
--input_checkpoint=CNN/cnn_quantized.ckpt \
--output_graph=CNN/cnn_quantized.pb \
--output_node_names="out" \
--input_binary=true
