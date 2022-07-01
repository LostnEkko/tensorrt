# import functools
# import gin
# import mesh_tensorflow.transformer.dataset as transformer_dataset
# import t5.data
# from t5.models import utils as model_utils
# import tensorflow.compat.v1 as tf
# import tensorflow_datasets as tfds

import json
import csv
from os.path import exists
# from t5.models import mesh_transformer
import tensorflow as tf

def json2tsv(path_to_json, output_tsv_path):
    '''
    convert json file to tsv file in format <input>/t<output>

    '''

    sfd = open(path_to_json, "r")
    ofd = open(output_tsv_path, "w+")
    tsv_writer = csv.writer(ofd, delimiter='\t')
    line = sfd.readline()
    while line:
        data = json.loads(line)
        tsv_writer.writerow(["url", "text"])
        tsv_writer.writerow([data["url"], data["text"]])
        line = sfd.readline()
    sfd.close()
    ofd.close()




def get_dataset_c4(path_to_json, sequence_length, vocabulary):
    overwrite_tsv_flag = True
    # Have tsv file created under the same dir and name as json file
    filename = '.'.join(path_to_json.split('.')[:-1]) + '.tsv'
    if overwrite_tsv_flag or not exists(filename):
        json2tsv(path_to_json, filename)

    # return mesh_transformer.tsv_dataset_fn(
    #     filename,
    #     sequence_length,
    #     "",
    #     vocabulary,
    #     shuffle_buffer_size=10000
    # )

    return tf.data.experimental.make_csv_dataset(
        filename,
        field_delim='\t',
        batch_size=1,
        header=True,
        shuffle=False,
        column_names= ["url", "text"]
    )


'''
def packed_parallel_tsv_dataset(dataset=gin.REQUIRED,
                                dataset_split=gin.REQUIRED,
                                batch_size=None,
                                sequence_length=gin.REQUIRED,
                                vocabulary=gin.REQUIRED,
                                append_eos=True,
                                eos_id=1,
                                max_encoded_len=0):
  """Reads parallel tab-separated text file. One example per line."""
  del batch_size
  del dataset_split

  def _parse_fn(record):  # pylint: disable=missing-docstring
    tokens = tf.decode_csv(
        record,
        record_defaults=[""] * 2,
        field_delim="\t",
        use_quote_delim=False)
    return {"inputs": tokens[0], "targets": tokens[1]}

  def _encode_fn(features):  # pylint: disable=missing-docstring
    inputs_vocabulary = vocabulary[0] if isinstance(vocabulary,
                                                    tuple) else vocabulary
    targets_vocabulary = vocabulary[1] if isinstance(vocabulary,
                                                     tuple) else vocabulary
    inputs_enc = inputs_vocabulary.encode_tf(features["inputs"])
    targets_enc = targets_vocabulary.encode_tf(features["targets"])
    if append_eos:
      inputs_enc = tf.concat([tf.cast(inputs_enc, tf.int64), [eos_id]], 0)
      targets_enc = tf.concat([tf.cast(targets_enc, tf.int64), [eos_id]], 0)
    return {"inputs": inputs_enc, "targets": targets_enc}

  dataset = dataset.map(
      _parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  dataset = dataset.map(
      _encode_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

  def _filter_fn(features):  # pylint: disable=missing-docstring
    return tf.less_equal(
        tf.reduce_max(
            tf.stack([tf.size(v) for v in features.values()], axis=0)),
        max_encoded_len)

  if max_encoded_len:
    tf.logging.info("Filtering encoded examples longer than %d" %
                    max_encoded_len)
    dataset = dataset.filter(_filter_fn)

  return pack_or_pad(dataset, sequence_length)
  '''