# coding=utf-8

import os

import hgtk
import numpy as np
from allennlp.commands.elmo import ElmoEmbedder, batch_to_ids

from pos_tagger import preprocess


class TextEncoder(object):
    @property
    def vocab_size(self):
        return None

    def encode(self, s):
        # return s[:]
        preprocessed_sentence = preprocess.preprocess_and_tokenize(s)
        print(f'{preprocessed_sentence}')
        character_ids = batch_to_ids([preprocessed_sentence])
        character_ids = np.array(character_ids)
        print(f'{character_ids}')
        flattened_character_ids = np.reshape(character_ids, [-1])
        flattened_character_ids = [int(character_id)
                                   for character_id in flattened_character_ids]
        print(f'{flattened_character_ids}')
        return flattened_character_ids


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
        """Transform a string into a float array.

        Args:
            s: space separated string.

        Returns:
            array of float values.
        """
        preprocessed_sentence = preprocess.preprocess_and_tokenize(s)
        embeddings = self.elmo.embed_sentence(preprocessed_sentence)[0, :, :]
        embeddings = np.reshape(embeddings, [-1])
        embeddings = [float(f) for f in embeddings]
        return embeddings


if __name__ == '__main__':
    tmp_dir = '/tmp/t2t_datagen'
    elmo = ElmoEncoder(tmp_dir)
    preprocessed_sentence = preprocess.preprocess_and_tokenize('밥을 먹자')
    print(preprocessed_sentence)
    # print(elmo.encode('밥을 먹자'))
    ids = batch_to_ids([preprocessed_sentence])
    print(ids)
