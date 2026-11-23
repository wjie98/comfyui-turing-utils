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
        # The pre-refactor snapshot freezes existing workflows, not additions.
        assert expected.keys() <= NODE_CLASS_MAPPINGS.keys()
        for name in expected:
            node = NODE_CLASS_MAPPINGS[name]
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
    # CLIP types are inherited from ComfyUI. Retain coverage for every type
    # existing workflows can name while allowing upstream to add more.
    actual_types = actual["TuringUtilsConvRotCLIPLoader"]["inputs"]["required"]["type"][0]
    expected_types = expected["TuringUtilsConvRotCLIPLoader"]["inputs"]["required"]["type"][0]
    assert set(expected_types) <= set(actual_types)
    actual_types[:] = expected_types
    # Adding a strategy does not change the serialized inputs of any existing
    # mode. Check Spark separately and keep the old-mode snapshot frozen.
    attention = actual["TuringUtilsAttentionStrategy"]
    assert attention["input_order"].pop("optional") == ["vpack"]
    compression = attention["inputs"].pop("optional")["vpack"]
    assert compression[0] == "COMBO"
    assert compression[1]["options"] == ["5/5", "4/5", "3/5", "2/5"]
    assert compression[1]["default"] == "5/5"
    options = actual["TuringUtilsAttentionStrategy"]["inputs"]["required"]["strategy"][
        1
    ]["options"]
    spark = [option for option in options if option["key"] == "spark"]
    assert len(spark) == 1
    options.remove(spark[0])
    none = [option for option in options if option["key"] == "none"]
    assert len(none) == 1
    assert none[0]["inputs"] == {"required": {}}
    options.remove(none[0])
    # Veda explicitly adds the same protection policy as Sol/SLA. Verify the
    # addition before comparing all other existing serialized fields unchanged.
    veda = next(option for option in options if option["key"] == "veda")
    fields = veda["inputs"]["required"]
    prefix = fields.pop("prefix_policy")
    assert prefix[0] == "COMFY_DYNAMICCOMBO_V3"
    assert [option["key"] for option in prefix[1]["options"]] == [
        "auto",
        "none",
        "manual",
    ]
    assert actual == expected
