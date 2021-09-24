from collections import namedtuple
from pathlib import Path
from typing import List, Dict

import yaml

TutorialConfig = namedtuple("TutorialConfig", "name source")


def batch_config(config_path: Path) -> List[TutorialConfig]:
    parsed_configs = _parse_config(config_path).get("tutorials", [])
    if not parsed_configs:
        raise AttributeError("Provided configuration file has no specified tutorials to convert!")

    tutorial_configs = [
        TutorialConfig(
            name=tutorial_config.get("name", "no_name"),
            source=tutorial_config.get("source", None)
        )
        for tutorial_config in parsed_configs
    ]

    if any(list(tc.source is None for tc in tutorial_configs)):
        raise AttributeError(f"Not all configs specify source path! {tutorial_configs}")

    return tutorial_configs


def _parse_config(config_path: Path) -> List[Dict[str, str]]:
    with open(config_path) as f:
        return yaml.load(f)
