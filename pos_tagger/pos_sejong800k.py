# coding=utf-8

import os

import tensorflow as tf
from tensor2tensor.data_generators import (generator_utils, problem,
                                           text_encoder)
from tensor2tensor.data_generators.text_problems import (Text2TextProblem,
                                                         VocabType,
                                                         text2text_txt_tab_iterator)
from tensor2tensor.utils import metrics, registry


@registry.register_problem
class PosSejong800kSubword(Text2TextProblem):
    """ Problem spec for Sejong POS tagging. 

    This assigns parts of speech to each word (and other token).
    The data is stored in a file named `pos_sejong800k_subword.pairs`.
    This file is a UTF-8 text file where
    each line contains an input sequence and an output sequence,
    separated by a tab character.
    """

    @property
    def dataset_splits(self):
        """Splits of data to produce and number of output shards for each."""
        return [{
            "split": problem.DatasetSplit.TRAIN,
            "shards": 9,
        }, {
            "split": problem.DatasetSplit.EVAL,
            "shards": 1,
        }]

    @property
    def is_generate_per_split(self):
        """A single call to `generate_samples` generates for all `dataset_splits`.

        Set to True if you already have distinct subsets of data for each dataset
        split specified in `self.dataset_splits`. `self.generate_samples` will be
        called once for each split.

        Set to False if you have a unified dataset that you'd like to have split out
        into training and evaluation data automatically. `self.generate_samples`
        will be called only once and the data will be sharded across the dataset
        splits specified in `self.dataset_splits`.

        Returns:
            bool
        """
        return False

    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        """Generate samples of input text and target text pairs.

        Each yielded dict will be made into a single example. The values should be
        raw text. The Problem will generate a vocabulary and encode the raw text as
        integers as part of the data generation process.

        This method is typically called once per split in `self.dataset_splits`
        unless `self.is_generate_per_split=False`.

        Args:
            data_dir: final data directory. Typically only used in this method to copy
                over user-supplied vocab files (for example, if vocab_type ==
                VocabType.TOKEN).
            tmp_dir: temporary directory that you can use for downloading and scratch.
            dataset_split: problem.DatasetSplit, which data split to generate samples
                for (for example, training and evaluation).

        Yields:
            {"inputs": text, "targets": text}
        """
        dataset_filename = self.dataset_filename()
        data_path = os.path.join(tmp_dir, f'{dataset_filename}.pairs')
        return text2text_txt_tab_iterator(data_path)

    @property
    def vocab_type(self):
        """What kind of vocabulary to use.

        `VocabType`s:
            * `SUBWORD`: `SubwordTextEncoder`, an invertible wordpiece vocabulary.
                Must provide `self.approx_vocab_size`. Generates the vocabulary based on
                the training data. To limit the number of samples the vocab generation
                looks at, override `self.max_samples_for_vocab`. Recommended and
                default.
            * `CHARACTER`: `ByteTextEncoder`, encode raw bytes.
            * `TOKEN`: `TokenTextEncoder`, vocabulary based on a file. Must provide a
                vocabulary file yourself (`TokenTextEncoder.store_to_file`) because one
                will not be generated for you. The vocab file should be stored in
                `data_dir/` with the name specified by `self.vocab_filename`.

        Returns:
            VocabType constant
        """
        return VocabType.SUBWORD

    @property
    def approx_vocab_size(self):
        """Approximate vocab size to generate. Only for VocabType.SUBWORD."""
        return 2**15  # ~32k

    @property
    def oov_token(self):
        """Out of vocabulary token. Only for VocabType.TOKEN."""
        return None

    @property
    def has_inputs(self):
        return True

    def feature_encoders(self, data_dir):
        encoder = self.get_or_create_vocab(data_dir, None, force_get=True)
        encoders = {"targets": encoder}
        if self.has_inputs:
            encoders["inputs"] = encoder
        return encoders

    @property
    def vocab_filename(self):
        if self.vocab_type == VocabType.SUBWORD:
            return "vocab.%s.%d.%s" % (self.dataset_filename(),
                                       self.approx_vocab_size,
                                       VocabType.SUBWORD)
        else:
            return "vocab.%s.%s" % (self.dataset_filename(), VocabType.TOKEN)

    def get_or_create_vocab(self, data_dir, tmp_dir, force_get=False):
        if self.vocab_type == VocabType.CHARACTER:
            encoder = text_encoder.ByteTextEncoder()
        elif self.vocab_type == VocabType.SUBWORD:
            if force_get:
                vocab_filepath = os.path.join(data_dir, self.vocab_filename)
                encoder = text_encoder.SubwordTextEncoder(vocab_filepath)
            else:
                encoder = generator_utils.get_or_generate_vocab_inner(
                    data_dir, self.vocab_filename, self.approx_vocab_size,
                    self.generate_text_for_vocab(data_dir, tmp_dir),
                    max_subtoken_length=self.max_subtoken_length,
                    reserved_tokens=(
                        text_encoder.RESERVED_TOKENS + self.additional_reserved_tokens))
        elif self.vocab_type == VocabType.TOKEN:
            if force_get:
                vocab_filepath = os.path.join(data_dir, self.vocab_filename)
                encoder = text_encoder.TokenTextEncoder(vocab_filepath,
                                                        replace_oov=self.oov_token)
            else:
                encoder = get_or_generate_vocab_inner_token(
                    data_dir, self.vocab_filename,
                    self.generate_text_for_vocab(data_dir, tmp_dir),
                    self.oov_token)
        else:
            raise ValueError(
                "Unrecognized VocabType: %s" % str(self.vocab_type))
        return encoder

    @property
    def max_subtoken_length(self):
        """Maximum subtoken length when generating vocab.

        SubwordTextEncoder vocabulary building is quadratic-time wrt this variable,
        setting it to None uses the length of the longest token in the corpus.

        Returns:
        an integer or None
        """
        return 200

    def generate_encoded_samples(self, data_dir, tmp_dir, dataset_split):
        generator = self.generate_samples(data_dir, tmp_dir, dataset_split)
        encoder = self.get_or_create_vocab(data_dir, tmp_dir)
        return text2text_generate_encoded(generator, encoder,
                                          has_inputs=self.has_inputs)

    def hparams(self, defaults, unused_model_hparams):
        p = defaults
        p.stop_at_eos = int(True)
        # p.hidden_size = 1024

        source_vocab_size = self._encoders["inputs"].vocab_size
        p.input_modality = {
            "inputs": (registry.Modalities.SYMBOL, source_vocab_size)
            # "inputs": ("generic:elmo_modality", source_vocab_size)
            # "inputs": (registry.Modalities.GENERIC, source_vocab_size)
        }
        target_vocab_size = self._encoders["targets"].vocab_size
        p.target_modality = (registry.Modalities.SYMBOL, target_vocab_size)
        # p.target_modality = (
        #     registry.Modalities.CLASS_LABEL, target_vocab_size)

    def example_reading_spec(self):
        data_fields = {"targets": tf.VarLenFeature(tf.int64)}
        if self.has_inputs:
            data_fields["inputs"] = tf.VarLenFeature(tf.int64)

        data_items_to_decoders = None
        return (data_fields, data_items_to_decoders)

    def eval_metrics(self):
        return [
            metrics.Metrics.ACC, metrics.Metrics.ACC_TOP5,
            metrics.Metrics.ACC_PER_SEQ, metrics.Metrics.NEG_LOG_PERPLEXITY,
            metrics.Metrics.APPROX_BLEU, metrics.Metrics.ROUGE_2_F,
            metrics.Metrics.ROUGE_L_F
        ]


# PosSejong800K problem with TOKEN
@registry.register_problem
class PosSejong800kToken(PosSejong800kSubword):
    @property
    def vocab_type(self):
        return VocabType.TOKEN


def text2text_generate_encoded(sample_generator,
                               vocab,
                               targets_vocab=None,
                               has_inputs=True):
    """Encode Text2Text samples from the generator with the vocab."""
    targets_vocab = targets_vocab or vocab
    for sample in sample_generator:
        if has_inputs:
            sample["inputs"] = vocab.encode(sample["inputs"])
            sample["inputs"].append(text_encoder.EOS_ID)  # EOS 추가
        sample["targets"] = targets_vocab.encode(sample["targets"])
        sample["targets"].append(text_encoder.EOS_ID)  # EOS 추가
        yield sample


def build_token_encoder_from_generator(generator, oov_token=None):
    # TODO(jongseong): apply collections.defaultdict to vocab_list
    vocab_list = set()
    for sample in generator:
        vocab_list.update(sample.split())
    encoder = text_encoder.TokenTextEncoder(
        None, vocab_list=vocab_list, replace_oov=oov_token)
    return encoder


def get_or_generate_vocab_inner_token(data_dir, vocab_filename, generator, oov_token):
    """Inner implementation for vocab generators.

    Args:
        data_dir: The base directory where data and vocab files are stored. If None,
        then do not save the vocab even if it doesn't exist.
        vocab_filename: relative filename where vocab file is stored
        generator: a generator that produces tokens from the vocabulary

    Returns:
        A TokenTextEncoder vocabulary object.
    """
    if data_dir and vocab_filename:
        vocab_filepath = os.path.join(data_dir, vocab_filename)
        if tf.gfile.Exists(vocab_filepath):
            tf.logging.info("Found vocab file: %s", vocab_filepath)
            return text_encoder.TokenTextEncoder(vocab_filepath)
    else:
        vocab_filepath = None

    tf.logging.info("Generating vocab file: %s", vocab_filepath)
    vocab = build_token_encoder_from_generator(generator, oov_token)

    if vocab_filepath:
        tf.gfile.MakeDirs(data_dir)
        vocab.store_to_file(vocab_filepath)

    return vocab
