############## Require T5 version 0.4.0 #############

import json
import csv
from os.path import exists

# from t5.models import mesh_transformer
import t5.data.preprocessors as prep
from t5.data.sentencepiece_vocabulary import SentencePieceVocabulary
import tensorflow as tf
from transformers import T5Tokenizer
import pprint
import tensorflow_io as tfio
import sys
import numpy as np

def get_dataset_c4(filenames, sequence_length=128, batch_size=32, vocab_size=512):
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    vocabulary = SentencePieceVocabulary(
        sentencepiece_model_file="spiece.model",
        extra_ids=0
    )
    noise_density = 0.15
    def json_filereader_fn(file_path):
        for line in open(file_path):
          data = json.loads(line)

          yield tokenizer(
              data["text"],
              return_tensors="tf",
              max_length=sequence_length,
              truncation=True,
              padding="max_length",
          ).input_ids

    if len(filenames) == 1:
          dataset = tf.data.Dataset.from_generator(
              json_filereader_fn,
              output_signature=tf.TensorSpec(
                  shape=(None, sequence_length),
                  dtype=tf.int32,
                  name=None
              ),
              args=filenames
          )
    else:
        raise RuntimeError()
    '''
    if len(filenames) == 1:
        dataset = tf.data.TextLineDataset(filenames[0])
        pass
    else:
        raise RuntimeError()

    def json2dict_fn(line):
        data = tfio.experimental.serialization.decode_json(
            line, 
            {'text': tf.TensorSpec(tf.TensorShape([]), tf.string, name="text")}
        )
        return data
    
    dataset = dataset.map(json2dict_fn)

    def dict2tokens_pyfn(data):
        # encoding = vocabulary.encode_tf(data['text']) #This works, nbut needs to truncate and pad myself
        data = data.numpy()
        if isinstance(data, (tuple, list, np.ndarray)):
            data = [bytes.decode(s) for s in data]
        else:
            data = bytes.decode(data)
        encoding = tokenizer(
            data, 
            return_tensors="tf", 
            max_length=sequence_length, 
            truncation=True,
            padding="max_length",
        ).input_ids
        return encoding
    
    def preprocess_inputs_fn(line):
        encoding = tf.py_function(dict2tokens_pyfn, [line['text']], [tf.int32])
        encoding = tf.squeeze(encoding)
        return {"targets": encoding}
    '''

    def preprocess_inputs_fn(line):
        return {"targets":line[0]}

    dataset = dataset.map(preprocess_inputs_fn)
    dataset = prep.denoise(
            dataset,
            vocabulary,
            noise_density=noise_density,
            noise_mask_fn=prep.random_spans_noise_mask,
            inputs_fn=prep.noise_token_to_sentinel,
            targets_fn=None
        )

    def transform_fn(features):
        pad_token_id = tokenizer.pad_token_id
        decoder_start_token_id = pad_token_id # by default those two the same. Can be read from json config provided

        # decoder_attention_mask, # Can be no masks at all, exclusive for flax
        # Attention mask 1 like for c4 if no padding
        # https://github.com/huggingface/transformers/blob/v4.20.1/src/transformers/models/t5/modeling_flax_t5.py#L1075
        # decoder_input_ids
        # replace padding token id's of the labels by -100 so it's ignored by the loss, if padding applied
        # labels are targets
        # labels = torch.tensor(labels)
        # labels[labels == tokenizer.pad_token_id] = -100
        # And shift labels later
        # Look Here: https://github.com/huggingface/transformers/blob/v4.20.1/src/transformers/models/t5/modeling_t5.py#L1624
        # https://github.com/huggingface/transformers/blob/d0acc9537829e7d067edbb791473bbceb2ecf056/src/transformers/models/t5/modeling_t5.py#L805-L830

        decoder_input_ids = tf.concat(
            [[decoder_start_token_id], 
            features["targets"][:-1]], 
            axis = 0
        )
        decoder_input_ids = tf.where(
            tf.equal(decoder_input_ids, -100), 
            tf.fill(tf.shape(decoder_input_ids), pad_token_id), 
            decoder_input_ids
        )
        return {
                "attention_mask": tf.ones_like(features["inputs"]), 
                "decoder_attention_mask": tf.ones_like(decoder_input_ids), 
                "decoder_input_ids": decoder_input_ids, 
                "input_ids": features["inputs"],
                "targets": features["targets"]
            }
    
    dataset = dataset.map(transform_fn)
    dataset = dataset.batch(batch_size)

    # Prefetch must be set to 1 to avoid deadlock: TF issue.
    dataset = dataset.prefetch(1) 
    return dataset