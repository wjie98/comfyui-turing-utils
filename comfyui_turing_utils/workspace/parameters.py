"""Serializable endpoint parameters and native DynamicCombo branch expansion."""

import copy
import math

DYNAMIC_COMBO = "COMFY_DYNAMICCOMBO_V3"
PARAMETER_TYPES = {"INT", "FLOAT", "BOOLEAN", "STRING", "COMBO", DYNAMIC_COMBO}
DYNAMIC_BINDINGS = "turing_dynamic_inputs"


def is_parameter(port):
    return port["kind"] == "parameter" or (
        port["kind"] == "value" and port["type"] == "STRING"
    )


def parameter_options(port):
    options = port.get("options") or {}
    return {"values": options} if isinstance(options, list) else options


def branch_parameters(option):
    result = {}
    for group in ("required", "optional"):
        for name, spec in option.get("inputs", {}).get(group, {}).items():
            type_, options = spec[0], spec[1] if len(spec) > 1 else {}
            if isinstance(type_, list):
                type_, options = "COMBO", {**options, "values": type_}
            elif type_ == "COMBO":
                options = {**options, "values": options.get("options", [])}
            result[name] = {"type": type_, "options": options}
            if "default" in options:
                result[name]["default"] = options["default"]
    return result


def validate_definition(port, depth=0):
    if depth > 16:
        raise ValueError("Dynamic parameter nesting is too deep")
    options = parameter_options(port)
    if not isinstance(options, dict):
        raise ValueError("Parameter options must be an object")
    if (
        not isinstance(port["type"], str)
        or port["type"] not in PARAMETER_TYPES
        or options.get("forceInput")
        or options.get("multiselect")
    ):
        raise ValueError("Dynamic branches only support editable parameters")
    if port["type"] == "COMBO":
        values = options.get("values")
        if not isinstance(values, list) or any(
            type(v) not in {str, int, float, bool}
            or isinstance(v, float) and not math.isfinite(v)
            for v in values
        ):
            raise ValueError("Combo parameters need serializable choices")
    elif port["type"] == DYNAMIC_COMBO:
        branches = options.get("options")
        if not isinstance(branches, list) or not branches:
            raise ValueError("DynamicCombo needs at least one branch")
        keys = set()
        for branch in branches:
            if (
                not isinstance(branch, dict)
                or not isinstance(branch.get("key"), str)
                or not branch["key"]
                or branch["key"] in keys
            ):
                raise ValueError("DynamicCombo branch keys must be unique strings")
            keys.add(branch["key"])
            inputs = branch.get("inputs", {})
            if not isinstance(inputs, dict) or set(inputs) - {"required", "optional"}:
                raise ValueError("Dynamic branches only support editable parameters")
            names = set()
            for group in inputs.values():
                if not isinstance(group, dict):
                    raise ValueError("Dynamic branch inputs must be an object")
                for name, spec in group.items():
                    if (
                        not isinstance(name, str)
                        or not name
                        or name in names
                        or not isinstance(spec, (list, tuple))
                        or len(spec) not in {1, 2}
                        or len(spec) == 2 and not isinstance(spec[1], dict)
                    ):
                        raise ValueError("Invalid dynamic branch input")
                    names.add(name)
            for child in branch_parameters(branch).values():
                validate_definition(child, depth + 1)
    if port["type"] != DYNAMIC_COMBO:
        validate_parameter(port, parameter_default(port))


def parameter_default(port):
    if "default" in port:
        return copy.deepcopy(port["default"])
    options = parameter_options(port)
    if port["type"] == DYNAMIC_COMBO:
        return {
            "selected": options["options"][0]["key"],
            "branches": {
                option["key"]: {
                    name: parameter_default(child)
                    for name, child in branch_parameters(option).items()
                }
                for option in options["options"]
            },
        }
    if port["type"] == "COMBO":
        return options["values"][0] if options["values"] else None
    return {"INT": 0, "FLOAT": 0.0, "BOOLEAN": False, "STRING": ""}.get(
        port["type"]
    )


def validate_parameter(port, value):
    type_ = port["type"]
    if type_ == DYNAMIC_COMBO:
        options = {o["key"]: o for o in parameter_options(port)["options"]}
        if (
            not isinstance(value, dict)
            or set(value) != {"selected", "branches"}
            or not isinstance(value["selected"], str)
            or value["selected"] not in options
            or not isinstance(value["branches"], dict)
            or set(value["branches"]) != set(options)
        ):
            raise ValueError("Invalid DynamicCombo parameter state")
        for key, branch in value["branches"].items():
            fields = branch_parameters(options[key])
            if not isinstance(branch, dict) or set(branch) != set(fields):
                raise ValueError("DynamicCombo branch parameters do not match its schema")
            for name, child in fields.items():
                validate_parameter(child, branch[name])
        return value
    valid = {
        "INT": type(value) is int,
        "FLOAT": type(value) is int or type(value) is float and math.isfinite(value),
        "BOOLEAN": type(value) is bool,
        "STRING": isinstance(value, str),
        "COMBO": value in parameter_options(port).get("values", [])
        or not parameter_options(port).get("values") and value is None,
    }.get(type_, False)
    if not valid:
        raise ValueError("Value does not match the endpoint parameter type")
    return value


def expand_dynamic(port, value, name):
    """Return the flat inputs expected by ComfyUI's native V3 parser."""
    validate_parameter(port, value)
    key = value["selected"]
    option = next(o for o in parameter_options(port)["options"] if o["key"] == key)
    result = {name: key}
    for child_name, child in branch_parameters(option).items():
        path = name + "." + child_name
        child_value = value["branches"][key][child_name]
        if child["type"] == DYNAMIC_COMBO:
            result.update(expand_dynamic(child, child_value, path))
        else:
            result[path] = child_value
    return result
