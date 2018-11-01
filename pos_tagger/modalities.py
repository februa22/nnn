# coding=utf-8

import os

import hgtk
import tensorflow as tf
from allennlp.commands.elmo import ElmoEmbedder
from tensor2tensor.layers import common_layers, modalities
from tensor2tensor.utils import modality, registry

from pos_tagger import preprocess

# Make sure to set n_characters=262
options_file = 'elmo/options.json'
weight_file = 'elmo/weights.hdf5'

tmp_dir = '/tmp/t2t_datagen'
options_path = os.path.join(tmp_dir, options_file)
weight_path = os.path.join(tmp_dir, weight_file)

# create your ELMo class based on weight and option file
elmo = ElmoEmbedder(options_path, weight_path)


@registry.register_generic_modality
class ElmoModality(modality.Modality):
    """SymbolModality for inputs with ELMo as embeddings."""

    def _get_elmo(self, x):
        # TODO(jongseong): Implement to get vector representations for inputs from ELMo.
        ret = []
        return ret

    def bottom_simple(self, x, name, reuse):
        with tf.variable_scope(name, reuse=reuse):
            # Ensure the inputs are 3-D
            if len(x.get_shape()) == 4:
                x = tf.squeeze(x, axis=3)
            while len(x.get_shape()) < 3:
                x = tf.expand_dims(x, axis=-1)

        # TODO(jongseong): Replace the embedding layer with ELMo.
        var = self._get_weights()
        # var = self._get_elmo(x)
        x = common_layers.dropout_no_scaling(
            x, 1.0 - self._model_hparams.symbol_dropout)
        ret = common_layers.gather(var, x)
        if self._model_hparams.multiply_embedding_mode == "sqrt_depth":
            ret *= self._body_input_depth**0.5
        ret *= tf.expand_dims(tf.to_float(tf.not_equal(x, 0)), -1)
        return ret

    def bottom(self, x):
        bilm_output = elmo.elmo_bilm(x)
        layer_activations = bilm_output['activations']
        mask_with_bos_eos = bilm_output['mask']
        tf.logging.info('===============')
        tf.logging.info(f'{layer_activations}')
        tf.logging.info(f'{mask_with_bos_eos}')
        return layer_activations
