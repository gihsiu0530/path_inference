"""Config loader for additive FF options without changing stp3/config.py."""

import os

import yaml

import stp3.config as base_config


get_parser = base_config.get_parser
CfgNode = base_config._CfgNode


def _merge_file_with_optional_base(cfg, filename):
    with open(filename, "r", encoding="utf-8") as handle:
        values = yaml.safe_load(handle) or {}
    base_filename = values.pop("BASE_CONFIG", None)
    if base_filename:
        if not os.path.isabs(base_filename):
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            base_filename = os.path.join(project_root, base_filename)
        cfg.merge_from_file(base_filename)
    cfg.set_new_allowed(True)
    cfg.merge_from_other_cfg(CfgNode(values, new_allowed=True))


def get_cfg(args=None, cfg_dict=None):
    cfg = base_config._C.clone()
    cfg.set_new_allowed(True)
    if cfg_dict is not None:
        values = CfgNode(cfg_dict, new_allowed=True)
        if "COST_FUNCTION" in values:
            for key in values.COST_FUNCTION:
                values.COST_FUNCTION.update({key: float(values.COST_FUNCTION.get(key))})
        cfg.merge_from_other_cfg(values)
    if args is not None:
        if args.config_file:
            _merge_file_with_optional_base(cfg, args.config_file)
        cfg.merge_from_list(args.opts)
    return cfg
