import tensorflow as tf

# Saved TF model path
model_path = '/projects/livepeer/src/server/docs/examples/model_repository/livepeer_detector/1/model.savedmodel/saved_model.pb'
# Directory to export new model
target_dir = 'saved_model'

with tf.gfile.FastGFile(model_path, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

# Input tensor name
input_name = 'image'
# Output tensor name
output_name = 'Openpose/concat_stage7'


with tf.Session() as sess:
    model_input = tf.saved_model.utils.build_tensor_info(sess.graph.get_tensor_by_name(input_name + ':0'))
    model_output = tf.saved_model.utils.build_tensor_info(sess.graph.get_tensor_by_name(output_name + ':0'))

    signature_definition = tf.saved_model.signature_def_utils.build_signature_def(
        inputs={input_name: model_input},
        outputs={output_name: model_output},
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)

    builder = tf.saved_model.builder.SavedModelBuilder(target_dir)
    builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING],
                                         signature_def_map={
                                             tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature_definition
                                         }, clear_devices=True)
    builder.save()