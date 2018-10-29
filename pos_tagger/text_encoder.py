# coding=utf-8

import os

import hgtk
from allennlp.commands.elmo import ElmoEmbedder

from pos_tagger.elmo import preprocess


class ElmoEncoder(object):
    def __init__(self, tmp_dir):
        # Make sure to set n_characters=262
        options_file = 'elmo/options.json'
        weight_file = 'elmo/weights.hdf5'

        options_path = os.path.join(tmp_dir, options_file)
        weight_path = os.path.join(tmp_dir, weight_file)

        # create your ELMo class based on weight and option file
        self.elmo = ElmoEmbedder(options_path, weight_path)

    @property
    def vocab_size(self):
        return None

    def encode(self, s):
        """Transform a string into a float 2-D array.

        Args:
            s: space separated string.

        Returns:
            2-D array of float values.
        """
        preprocessed_sentence = preprocess.preprocess_and_tokenize(s)
        return self.elmo.embed_sentence(preprocessed_sentence)[0, :, :]


if __name__ == '__main__':
    tmp_dir = '/tmp/t2t_datagen'
    elmo = ElmoEncoder(tmp_dir)
    print(elmo.encode('밥을 먹자'))
