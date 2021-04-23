# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# flake8: noqa
from .loss import BCELossConf
from .loss import BCEWithLogitsLossConf
from .loss import CosineEmbeddingLossConf
from .loss import CTCLossConf
from .loss import L1LossConf
from .loss import HingeEmbeddingLossConf
from .loss import KLDivLossConf
from .loss import MarginRankingLossConf
from .loss import MSELossConf
from .loss import MultiLabelMarginLossConf
from .loss import MultiLabelSoftMarginLossConf
from .loss import MultiMarginLossConf
from .loss import NLLLossConf
from .loss import NLLLoss2dConf
from .loss import PoissonNLLLossConf
from .loss import SmoothL1LossConf
from .loss import SoftMarginLossConf
from .loss import TripletMarginLossConf
from hydra.core.config_store import ConfigStoreWithProvider


def register():
    with ConfigStoreWithProvider("torch") as cs:
        cs.store(group="torch/nn/modules/loss", name="bceloss", node=BCELossConf)
        cs.store(
            group="torch/nn/modules/loss",
            name="bcewithlogitsloss",
            node=BCEWithLogitsLossConf,
        )
        cs.store(
            group="torch/nn/modules/loss",
            name="cosineembeddingloss",
            node=CosineEmbeddingLossConf,
        )
        cs.store(group="torch/nn/modules/loss", name="ctcloss", node=CTCLossConf)
        cs.store(group="torch/nn/modules/loss", name="l1loss", node=L1LossConf)
        cs.store(
            group="torch/nn/modules/loss",
            name="hingeembeddingloss",
            node=HingeEmbeddingLossConf,
        )
        cs.store(group="torch/nn/modules/loss", name="kldivloss", node=KLDivLossConf)
        cs.store(
            group="torch/nn/modules/loss",
            name="marginrankingloss",
            node=MarginRankingLossConf,
        )
        cs.store(group="torch/nn/modules/loss", name="mseloss", node=MSELossConf)
        cs.store(
            group="torch/nn/modules/loss",
            name="multilabelmarginloss",
            node=MultiLabelMarginLossConf,
        )
        cs.store(
            group="torch/nn/modules/loss",
            name="multilabelsoftmarginloss",
            node=MultiLabelSoftMarginLossConf,
        )
        cs.store(
            group="torch/nn/modules/loss",
            name="multimarginloss",
            node=MultiMarginLossConf,
        )
        cs.store(group="torch/nn/modules/loss", name="nllloss", node=NLLLossConf)
        cs.store(group="torch/nn/modules/loss", name="nllloss2d", node=NLLLoss2dConf)
        cs.store(
            group="torch/nn/modules/loss",
            name="poissonnllloss",
            node=PoissonNLLLossConf,
        )
        cs.store(
            group="torch/nn/modules/loss", name="smoothl1loss", node=SmoothL1LossConf
        )
        cs.store(
            group="torch/nn/modules/loss",
            name="softmarginloss",
            node=SoftMarginLossConf,
        )
        cs.store(
            group="torch/nn/modules/loss",
            name="tripletmarginloss",
            node=TripletMarginLossConf,
        )
