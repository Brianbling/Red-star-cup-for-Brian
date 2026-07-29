"""Wrapper for various CTC decoders in SWIG.

Greedy decoder: always available (compiled with minimal deps).
Beam search + Scorer: require kenlm + openfst (not compiled by default).
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import swig_decoders

_NOT_AVAILABLE = "BeamDecoder/Scorer not compiled (requires kenlm + openfst). Only ctc_greedy_decoder is available."


class Scorer:
    def __init__(self, alpha, beta, model_path, vocabulary):
        raise NotImplementedError(_NOT_AVAILABLE)


class BeamDecoder:
    def __init__(self, vocabulary, beam_size,
                 cutoff_prob=1.0,
                 cutoff_top_n=40,
                 ext_scorer=None):
        raise NotImplementedError(_NOT_AVAILABLE)


def ctc_greedy_decoder(probs_seq, vocabulary):
    """Wrapper for ctc best path decoder in swig.

    :param probs_seq: 2-D list of probability distributions over each time
                      step, with each element being a list of normalized
                      probabilities over vocabulary and blank.
    :type probs_seq: 2-D list
    :param vocabulary: Vocabulary list.
    :type vocabulary: list
    :return: Decoding result string.
    :rtype: basestring
    """
    probs_list = probs_seq.tolist() if hasattr(probs_seq, 'tolist') else probs_seq
    result = swig_decoders.ctc_greedy_decoder(probs_list, vocabulary)
    return result


def ctc_beam_search_decoder(probs_seq,
                            vocabulary,
                            beam_size,
                            cutoff_prob=1.0,
                            cutoff_top_n=40,
                            ext_scoring_func=None):
    raise NotImplementedError(_NOT_AVAILABLE)


def ctc_beam_search_decoder_batch(probs_split,
                                  vocabulary,
                                  beam_size,
                                  num_processes,
                                  cutoff_prob=1.0,
                                  cutoff_top_n=40,
                                  ext_scoring_func=None):
    raise NotImplementedError(_NOT_AVAILABLE)
