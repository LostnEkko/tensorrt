import tensorflow as tf


def get_dataset(sequence_length=128, batch_size=32, vocab_size=512):
    if False:
        dataset = None
    else:
        tf.random.set_seed(12345)
        attention_mask = tf.random.uniform(
            shape=(1, sequence_length),
            maxval=vocab_size,
            dtype=tf.int32
        )
        ds_attention_mask = tf.data.Dataset.from_tensor_slices(attention_mask)

        decoder_attention_mask = tf.random.uniform(
            shape=(1, sequence_length),
            maxval=vocab_size,
            dtype=tf.int32
        )
        ds_decoder_attention_mask = tf.data.Dataset.from_tensor_slices(decoder_attention_mask)

        decoder_input_ids = tf.random.uniform(
            shape=(1, sequence_length),
            maxval=vocab_size,
            dtype=tf.int32
        )
        ds_decoder_input_ids = tf.data.Dataset.from_tensor_slices(decoder_input_ids)

        input_ids = tf.random.uniform(
            shape=(1, sequence_length),
            maxval=vocab_size,
            dtype=tf.int32
        )
        ds_input_ids = tf.data.Dataset.from_tensor_slices(input_ids)

        dataset = tf.data.Dataset.zip((
            ds_attention_mask, 
            ds_decoder_attention_mask, 
            ds_decoder_input_ids, 
            ds_input_ids
        ))
        dataset = dataset.repeat()
        dataset = dataset.batch(batch_size)
        dataset = dataset.take(count=1)  # loop over 1 batch
        dataset = dataset.cache()
        dataset = dataset.repeat()
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

dataset = get_dataset()
ds_iter = iter(dataset)
import pprint
for batch in ds_iter:
    pprint.pprint(batch)
    break