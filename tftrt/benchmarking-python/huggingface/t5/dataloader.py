############## Require T5 version 0.4.0 #############

import os
import glob

try:
    from prefetch_generator import background
except ModuleNotFoundError:
    print("[ERROR] Please install: `pip install --upgrade prefetch_generator`")
    raise

try:
    import orjson as json
except ModuleNotFoundError:
    print(
        "[WARNING] To process json data faster, please execute: "
        "`pip install --upgrade orjson`"
    )
    import json

import numpy as np
import tensorflow as tf

# from t5.models import mesh_transformer
import t5.data.preprocessors as prep
from t5.data.sentencepiece_vocabulary import SentencePieceVocabulary
from transformers import T5Tokenizer


def get_dataset_c4(
    data_dir,
    vocab_dir,
    tokenizer_dir=None,
    sequence_length=128,
    batch_size=32,
    vocab_size=512,
    noise_density=0.15
):

    # if False:
    #     fd = open(filenames[0], "r")
    #     line = fd.readline()
    #     while line:
    #         data = json.loads(line)
    #         encoding = tokenizer(
    #             data["text"],
    #             return_tensors="tf",
    #             max_length=sequence_length,
    #             truncation=True,
    #         )
    #         dataset = tf.data.Dataset.from_tensor_slices({"targets": encoding.input_ids})
    #         break
    #         line = fd.readline()
    #     fd.close()

    json_files = sorted(glob.glob(
        os.path.join(data_dir, "c4-validation.*.json")
    ))

    if tokenizer_dir is None:
        tokenizer = T5Tokenizer.from_pretrained("t5-small")
    else:
        tokenizer = T5Tokenizer.from_pretrained(tokenizer_dir)

    @background(max_prefetch=1)
    def jsonfile_parser(filename):

        for line in open(filename):
            data = json.loads(line)

            yield tokenizer(
                data["text"],
                return_tensors="tf",
                max_length=sequence_length,
                truncation=True,
                padding="max_length",
            ).input_ids

    def _get_ds_generator(_filename):
       return tf.data.Dataset.from_generator(
            lambda: jsonfile_parser(_filename),
            output_signature=tf.TensorSpec(
                shape=(None, sequence_length),
                dtype=tf.int32,
                # shape=(),
                # dtype=tf.string,
                name=None
            ),
        ).prefetch(buffer_size=tf.data.AUTOTUNE)

    dataset = tf.data.Dataset.sample_from_datasets(
        datasets=[_get_ds_generator(_f) for _f in json_files],
        seed=666,
        stop_on_empty_dataset=False
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
            tf.fill(decoder_input_ids.shape.as_list(), pad_token_id),
            decoder_input_ids
        )
        return {
                "attention_mask": tf.ones_like(features["inputs"]),
                "decoder_attention_mask": tf.ones_like(decoder_input_ids),
                "decoder_input_ids": decoder_input_ids,
                "input_ids": features["inputs"],
                "targets": features["targets"]
            }

    dataset = dataset.map(lambda line: {"targets":line[0]})
    vocabulary = SentencePieceVocabulary(
        sentencepiece_model_file=os.path.join(vocab_dir, "spiece.model"),
        extra_ids=0
    )
    dataset = prep.denoise(
            dataset,
            vocabulary,
            noise_density=noise_density,
            noise_mask_fn=prep.random_spans_noise_mask,
            inputs_fn=prep.noise_token_to_sentinel,
            targets_fn=None
        )
    dataset = dataset.map(transform_fn)

    # Prefetch an entire batch of data before batching
    dataset = dataset.prefetch(buffer_size=batch_size)

    # Then Batch
    dataset = dataset.batch(batch_size, drop_remainder=False)
    return dataset
