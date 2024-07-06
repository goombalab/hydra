# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

from transformers import BertConfig as TransformersBertConfig


class BertConfig(TransformersBertConfig):

    def __init__(
        self,
        attention_probs_dropout_prob: float = 0.0,
        use_position_embeddings: bool = False,
        matrix_mixer_type: str = 'hydra',
        is_prenorm: bool = False,
        d_conv: int = 7,
        expand: int = 2,
        chunk_size: int = 256,
        # Hydra Configurations
        d_state: int = 64,
        # Matrix Mixer Configurations
        is_data_dependent: bool = True,
        qk_dim: int = 16,
        headdim: int = 64,
        **kwargs,
    ):
        """Configuration class for MosaicBert.

        Args:
            alibi_starting_size (int): Use `alibi_starting_size` to determine how large of an alibi tensor to
                create when initializing the model. You should be able to ignore this parameter in most cases.
                Defaults to 512.
            attention_probs_dropout_prob (float): By default, turn off attention dropout in Mosaic BERT
                (otherwise, Flash Attention will be off by default). Defaults to 0.0.
        """
        super().__init__(
            attention_probs_dropout_prob=attention_probs_dropout_prob, **kwargs)
        self.use_position_embeddings = use_position_embeddings
        self.matrix_mixer_type = matrix_mixer_type
        self.is_prenorm = is_prenorm

        self.d_conv = d_conv
        self.expand = expand
        self.chunk_size = chunk_size

        # Hydra Configurations
        self.d_state = d_state

        # Matrix Mixer Configurations
        self.is_data_dependent = is_data_dependent  # Boolean flag for SAM
        self.qk_dim = qk_dim
        self.headdim = headdim
