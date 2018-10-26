# coding=utf-8

import tensorflow as tf
from tensor2tensor.layers import common_layers, modalities
from tensor2tensor.utils import registry


@registry.register_symbol_modality
class ElmoModality(modalities.SymbolModality):
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

        var = self._get_elmo(x)
        x = common_layers.dropout_no_scaling(
            x, 1.0 - self._model_hparams.symbol_dropout)
        ret = common_layers.gather(var, x)
        if self._model_hparams.multiply_embedding_mode == "sqrt_depth":
            ret *= self._body_input_depth**0.5
        ret *= tf.expand_dims(tf.to_float(tf.not_equal(x, 0)), -1)
        return ret
