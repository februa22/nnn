# coding=utf-8

import os
from itertools import tee

import tensorflow as tf
from tensor2tensor.data_generators import problem, text_encoder, text_problems
from tensor2tensor.utils import registry
from text_encoder import ElmoEncoder


@registry.register_problem
class PosSejong800k(text_problems.Text2TextProblem):
    """ Problem spec for Sejong POS tagging. 

    This assigns parts of speech to each word (and other token).
    The data is stored in a file named `parsing_train.pairs`.
    This file is a UTF-8 text file where
    each line contains an input sequence and POS tags,
    separated by a tab character.
    """

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

    @property
    def dataset_splits(self):
        return [{
            "split": problem.DatasetSplit.TRAIN,
            "shards": 9,
        }, {
            "split": problem.DatasetSplit.EVAL,
            "shards": 1,
        }]

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
        return text_problems.text2text_txt_tab_iterator(data_path)

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
        return text_problems.VocabType.TOKEN

    @property
    def vocab_filename(self):
        return "vocab.%s.%s" % (self.dataset_filename(), self.vocab_type)

    @property
    def targets_vocab_filename(self):
        return "vocab_targets.%s.%s" % (self.dataset_filename(), self.vocab_type)

    def _generate_tabbed_vocabs(self, generator):
        vocab_list = set()
        targets_vocab_list = set()
        tf.logging.info("Generating vocabularies")
        for sample in generator:
            vocab_list.update(sample['inputs'].split())
            targets_vocab_list.update(sample['targets'].split())
        encoder = text_encoder.TokenTextEncoder(
            None, vocab_list=vocab_list, replace_oov=self.oov_token)
        targets_encoder = text_encoder.TokenTextEncoder(
            None, vocab_list=targets_vocab_list, replace_oov=self.oov_token)
        return encoder, targets_encoder

    def get_or_generate_tabbed_vocabs(self, generator, data_dir, tmp_dir):
        vocab_path = os.path.join(data_dir, self.vocab_filename)
        targets_vocab_path = os.path.join(
            data_dir, self.targets_vocab_filename)

        # Get
        if tf.gfile.Exists(vocab_path) and tf.gfile.Exists(targets_vocab_path):
            tf.logging.info(f'Getting vocab for inputs from {vocab_path}')
            encoder = text_encoder.TokenTextEncoder(vocab_path,
                                                    replace_oov=self.oov_token)
            tf.logging.info(
                f'Getting vocab for targets from {targets_vocab_path}')
            targets_encoder = text_encoder.TokenTextEncoder(targets_vocab_path,
                                                            replace_oov=self.oov_token)
            return encoder, targets_encoder

        # Generate
        encoder, targets_encoder = self._generate_tabbed_vocabs(generator)
        encoder.store_to_file(vocab_path)
        targets_encoder.store_to_file(targets_vocab_path)
        return encoder, targets_encoder

    def _generate_targets_vocab(self, generator):
        targets_vocab_list = set()
        tf.logging.info("Generating vocabulary for targets")
        for sample in generator:
            targets_vocab_list.update(sample['targets'].split())
        targets_encoder = text_encoder.TokenTextEncoder(
            None, vocab_list=targets_vocab_list, replace_oov=self.oov_token)
        return targets_encoder

    def get_or_generate_targets_vocab(self, generator, data_dir, tmp_dir):
        targets_vocab_path = os.path.join(
            data_dir, self.targets_vocab_filename)

        # Get
        if tf.gfile.Exists(targets_vocab_path):
            tf.logging.info(
                f'Getting vocab for targets from {targets_vocab_path}')
            targets_encoder = text_encoder.TokenTextEncoder(targets_vocab_path,
                                                            replace_oov=self.oov_token)
            return targets_encoder

        # Generate
        targets_encoder = self._generate_targets_vocab(generator)
        targets_encoder.store_to_file(targets_vocab_path)
        return targets_encoder

    def generate_encoded_samples(self, data_dir, tmp_dir, dataset_split):
        generator = self.generate_samples(data_dir, tmp_dir, dataset_split)

        generator, generator_for_vocab = tee(generator)
        vocab = ElmoEncoder(tmp_dir)
        targets_vocab = self.get_or_generate_targets_vocab(
            generator_for_vocab, data_dir, tmp_dir)

        return text_problems.text2text_generate_encoded(
            generator, vocab,
            targets_vocab,
            has_inputs=self.has_inputs)

    def hparams(self, defaults, unused_model_hparams):
        p = defaults
        p.stop_at_eos = int(False)

        source_vocab_size = self._encoders["inputs"].vocab_size
        p.input_modality = {
            # "inputs": (registry.Modalities.SYMBOL, source_vocab_size)
            # "inputs": ("symbol:elmo_modality", source_vocab_size)
            "inputs": (registry.Modalities.GENERIC, source_vocab_size)
        }
        target_vocab_size = self._encoders["targets"].vocab_size
        p.target_modality = (registry.Modalities.SYMBOL, target_vocab_size)
        # p.target_modality = (
        #     registry.Modalities.CLASS_LABEL, target_vocab_size)

        # TODO(jongseong): Is this even needed?
        if self.packed_length:
            identity = (registry.Modalities.GENERIC, None)
            if self.has_inputs:
                p.input_modality["inputs_segmentation"] = identity
                p.input_modality["inputs_position"] = identity
            p.input_modality["targets_segmentation"] = identity
            p.input_modality["targets_position"] = identity
