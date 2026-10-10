"""Serialized node interfaces must survive internal refactoring."""

import json
from pathlib import Path
from unittest.mock import patch

from comfyui_turing_utils.registration import NODE_CLASS_MAPPINGS


def normalize(value):
    if isinstance(value, (list, tuple)):
        return [normalize(item) for item in value]
    if isinstance(value, dict):
        return {
            key: normalize(item)
            for key, item in value.items()
            if key not in {"tooltip", "description"}
        }
    return value


def test_registered_node_interfaces_match_pre_refactor_workflows():
    expected = json.loads(
        (Path(__file__).parent / "fixtures/node_contracts.json").read_text()
    )
    actual = {}
    with patch("folder_paths.get_filename_list", return_value=["contract.safetensors"]):
        for name, node in NODE_CLASS_MAPPINGS.items():
            actual[name] = normalize(
                {
                    "inputs": node.INPUT_TYPES(),
                    "input_order": {
                        section: list(fields)
                        for section, fields in node.INPUT_TYPES().items()
                    },
                    "outputs": node.RETURN_TYPES,
                    "output_names": getattr(node, "RETURN_NAMES", None),
                    "function": node.FUNCTION,
                    "output_node": getattr(node, "OUTPUT_NODE", False),
                }
            )
    assert actual == expected
