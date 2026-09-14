"""TrainingModule binding for FORWARD-extrap/turn-ADMLP ASAP-FF."""

import stp3.trainer_codex_seg_ASAP as base_trainer

from stp3.config_ff import get_cfg
from stp3.model_vlm.codex_pure_ASAP_hybrid_ff import VLM_STP3_Gen as VLM_STP3


base_trainer.get_cfg = get_cfg
base_trainer.VLM_STP3 = VLM_STP3


class TrainingModule(base_trainer.TrainingModule):
    pass
