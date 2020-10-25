from typing import List

from ctcdecode import CTCBeamDecoder
import torch

from src.decoders.base_decoder import BaseDecoder


class CTCDecoder(BaseDecoder):

    def __init__(self, letters: List[str], **kwargs):
        super(CTCDecoder, self).__init__(letters, **kwargs)
        self.decoder = CTCBeamDecoder(
            letters,
            model_path=kwargs.get("model_path"),
            alpha=kwargs.get("alpha", 0),
            beta=kwargs.get("beta", 0),
            cutoff_top_n=kwargs.get("cutoff_top_n", 40),
            cutoff_prob=kwargs.get("cutoff_prob", 1.0),
            beam_width=kwargs.get("beam_width", 20),
            num_processes=kwargs.get("num_processes", 4),
            blank_id=kwargs.get("blank_id", 0),
            log_probs_input=kwargs.get("log_probs_input", True)
        )

    def __call__(self, output):
        out_reshape = output.permute(1, 0, 2)
        # OUT_RESHAPE SHAPE: (N, T, C)
        beam_results, beam_scores, timesteps, out_lens = self.decoder.decode(out_reshape)
        return beam_results[:, 0, :].squeeze(dim=1), out_lens[:, 0]
