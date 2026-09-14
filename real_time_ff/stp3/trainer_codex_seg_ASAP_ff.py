"""TrainingModule binding for the additive FF planner."""

import stp3.trainer_codex_seg_ASAP as base_trainer

from stp3.config_ff import get_cfg
from stp3.model_vlm.codex_pure_ASAP_ff import VLM_STP3_Gen as VLM_STP3


# The base trainer resolves these names at TrainingModule construction time.
# This follows the repository's existing additive trainer binding pattern.
base_trainer.get_cfg = get_cfg
base_trainer.VLM_STP3 = VLM_STP3


class TrainingModule(base_trainer.TrainingModule):
    pass
