import copy
from typing import List

from ctcdecode import CTCBeamDecoder

from src.decoders.base_decoder import BaseDecoder


class CTCDecoder(BaseDecoder):

    def __call__(self, out, labels):
        labels = copy.copy(labels)
        # labels.('_')
        labels = ['_'] + labels
        self.decoder = CTCBeamDecoder(
            labels,
            model_path=None,
            alpha=0,
            beta=0,
            cutoff_top_n=40,
            cutoff_prob=1.0,
            beam_width=100,
            num_processes=4,
            blank_id=0,
            log_probs_input=False
        )
        timesteps_size, batch_size, n_labels = out.shape
        out_reshape = out.reshape((batch_size, timesteps_size, n_labels))
        beam_results, beam_scores, timesteps, out_lens = self.decoder.decode(out_reshape)
        return beam_results[:, 0, :].squeeze()
