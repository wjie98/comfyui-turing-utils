"""Versioned semantic attention-layout contract and provider registry."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace


ATTENTION_LAYOUT_PROTOCOL_VERSION = 1
ATTENTION_LAYOUT_KEY = "turing_utils_attention_layout"
ATTENTION_LAYOUT_REQUIREMENT_KEY = "turing_utils_attention_layout_required"
ATTENTION_SEGMENT_ROLES = frozenset(
    {
        "text",
        "reference_image",
        "reference_video_anchor",
        "reference_video",
        "reference_audio",
        "target_audio",
        "target_video",
        "pose_video",
        "context_video",
    }
)


_ROLE_DEFAULTS = {
    "text": (False, False, True),
    "reference_image": (False, False, True),
    "reference_video_anchor": (False, False, True),
    "reference_video": (True, True, False),
    "reference_audio": (False, False, True),
    "target_audio": (False, False, True),
    "target_video": (True, True, False),
    "pose_video": (False, False, True),
    "context_video": (True, True, False),
}


@dataclass(frozen=True, slots=True)
class AttentionSegment:
    start: int
    stop: int
    role: str
    sparse_query_allowed: bool
    sparse_key_allowed: bool
    exact_kv: bool
    topology_id: str | None = None

    @classmethod
    def for_role(
        cls,
        start: int,
        stop: int,
        role: str,
        *,
        topology_id: str | None = None,
    ) -> "AttentionSegment":
        sparse_q, sparse_k, exact_kv = _ROLE_DEFAULTS.get(
            str(role), (False, False, True)
        )
        return cls(
            int(start),
            int(stop),
            str(role),
            sparse_q,
            sparse_k,
            exact_kv,
            topology_id,
        )

    def to_wire(self) -> dict:
        return {
            "start": self.start,
            "stop": self.stop,
            "role": self.role,
            "sparse_query_allowed": self.sparse_query_allowed,
            "sparse_key_allowed": self.sparse_key_allowed,
            "exact_kv": self.exact_kv,
            "topology_id": self.topology_id,
        }


@dataclass(frozen=True, slots=True)
class AttentionTopology:
    topology_id: str
    kind: str
    start: int
    stop: int
    tokens_per_frame: int
    spatial_tokens_height: int
    spatial_tokens_width: int
    axis: str = "both"

    def to_wire(self) -> dict:
        return {
            "topology_id": self.topology_id,
            "kind": self.kind,
            "start": self.start,
            "stop": self.stop,
            "tokens_per_frame": self.tokens_per_frame,
            "spatial_tokens_height": self.spatial_tokens_height,
            "spatial_tokens_width": self.spatial_tokens_width,
            "axis": self.axis,
        }


@dataclass(frozen=True, slots=True)
class AttentionSemanticLayout:
    provider: str
    query_segments: tuple[AttentionSegment, ...]
    key_segments: tuple[AttentionSegment, ...]
    topologies: tuple[AttentionTopology, ...]
    layer_index: int
    layer_count: int
    protocol_version: int = ATTENTION_LAYOUT_PROTOCOL_VERSION

    def validate(
        self,
        query_length: int | None = None,
        key_length: int | None = None,
    ) -> str | None:
        if self.protocol_version != ATTENTION_LAYOUT_PROTOCOL_VERSION:
            return f"unsupported semantic-layout protocol v{self.protocol_version}"
        if not self.provider:
            return "layout provider is empty"
        if self.layer_count <= 0 or not 0 <= self.layer_index < self.layer_count:
            return "layer metadata is invalid"
        for name, segments, length in (
            ("query", self.query_segments, query_length),
            ("key", self.key_segments, key_length),
        ):
            if not segments:
                return f"{name} segments are empty"
            cursor = 0
            for segment in segments:
                if (
                    segment.start != cursor
                    or segment.stop <= segment.start
                    or segment.role not in ATTENTION_SEGMENT_ROLES
                    or not isinstance(segment.sparse_query_allowed, bool)
                    or not isinstance(segment.sparse_key_allowed, bool)
                    or not isinstance(segment.exact_kv, bool)
                ):
                    return f"{name} segments are not a contiguous partition"
                cursor = segment.stop
            if length is not None and cursor != int(length):
                return f"{name} segments cover {cursor} tokens, expected {length}"
        topology_ids = set()
        query_limit = self.query_segments[-1].stop
        key_limit = self.key_segments[-1].stop
        for topology in self.topologies:
            query_links = tuple(
                segment
                for segment in self.query_segments
                if segment.topology_id == topology.topology_id
            )
            key_links = tuple(
                segment
                for segment in self.key_segments
                if segment.topology_id == topology.topology_id
            )
            required_links = (
                (query_links,)
                if topology.axis == "query"
                else (key_links,)
                if topology.axis == "key"
                else (query_links, key_links)
            )
            relevant_limit = (
                query_limit
                if topology.axis == "query"
                else key_limit
                if topology.axis == "key"
                else max(query_limit, key_limit)
            )
            if (
                not topology.topology_id
                or topology.topology_id in topology_ids
                or topology.kind != "video_grid"
                or topology.axis not in {"query", "key", "both"}
                or topology.start < 0
                or topology.stop <= topology.start
                or topology.stop > relevant_limit
                or topology.tokens_per_frame <= 0
                or topology.spatial_tokens_height <= 0
                or topology.spatial_tokens_width <= 0
                or topology.spatial_tokens_height * topology.spatial_tokens_width
                != topology.tokens_per_frame
                or (topology.stop - topology.start) % topology.tokens_per_frame
                or any(not links for links in required_links)
                or any(
                    min(segment.start for segment in links) != topology.start
                    or max(segment.stop for segment in links) != topology.stop
                    for links in required_links
                )
            ):
                return "video topology metadata is invalid"
            topology_ids.add(topology.topology_id)
        for segment in (*self.query_segments, *self.key_segments):
            if (
                segment.topology_id is not None
                and segment.topology_id not in topology_ids
            ):
                return f"segment references unknown topology {segment.topology_id!r}"
        return None

    def to_wire(self, *, extensions: Mapping | None = None) -> dict:
        wire = dict(extensions or {})
        wire.update(
            protocol_version=self.protocol_version,
            provider=self.provider,
            query_segments=tuple(segment.to_wire() for segment in self.query_segments),
            key_segments=tuple(segment.to_wire() for segment in self.key_segments),
            topologies=tuple(topology.to_wire() for topology in self.topologies),
            layer_index=self.layer_index,
            layer_count=self.layer_count,
        )
        # Keep the v0 fields during the transition so an older Python sparse
        # backend stays safe and dense instead of misreading the sequence.
        if self.query_segments == self.key_segments:
            wire["segments"] = tuple(
                (segment.start, segment.stop, segment.role)
                for segment in self.query_segments
            )
        primary = next(
            (
                topology
                for topology in self.topologies
                if any(
                    segment.role == "target_video"
                    and segment.topology_id == topology.topology_id
                    for segment in self.query_segments
                )
            ),
            self.topologies[0] if len(self.topologies) == 1 else None,
        )
        if primary is not None:
            wire.update(
                dense_prefix_tokens=primary.start,
                topology_start_tokens=primary.start,
                topology_tokens=primary.stop - primary.start,
                tokens_per_frame=primary.tokens_per_frame,
                spatial_tokens_height=primary.spatial_tokens_height,
                spatial_tokens_width=primary.spatial_tokens_width,
            )
        return wire


@dataclass(frozen=True)
class LayoutProviderStatus:
    """Result returned by a model adapter asked to publish attention layout."""

    model_kind: str | None
    installed: bool
    reason: str | None = None

    @property
    def required(self) -> bool:
        return self.model_kind is not None


LayoutProviderInstaller = Callable[[object], LayoutProviderStatus]
_PROVIDERS: list[LayoutProviderInstaller] = []


def register_attention_layout_provider(installer: LayoutProviderInstaller) -> None:
    if installer not in _PROVIDERS:
        _PROVIDERS.append(installer)


def ensure_attention_layout_provider(model) -> LayoutProviderStatus:
    for installer in tuple(_PROVIDERS):
        status = installer(model)
        if status.required:
            return status
    return LayoutProviderStatus(None, False, None)


def _segment_from_wire(value) -> AttentionSegment | None:
    if isinstance(value, AttentionSegment):
        return value
    if isinstance(value, Mapping):
        try:
            sparse_query_allowed = value["sparse_query_allowed"]
            sparse_key_allowed = value["sparse_key_allowed"]
            exact_kv = value["exact_kv"]
            if not all(
                isinstance(item, bool)
                for item in (sparse_query_allowed, sparse_key_allowed, exact_kv)
            ):
                return None
            return AttentionSegment(
                int(value["start"]),
                int(value["stop"]),
                str(value["role"]),
                sparse_query_allowed,
                sparse_key_allowed,
                exact_kv,
                value.get("topology_id"),
            )
        except (KeyError, TypeError, ValueError):
            return None
    if isinstance(value, tuple) and len(value) == 3:
        start, stop, role = value
        if (
            not isinstance(start, int)
            or isinstance(start, bool)
            or not isinstance(stop, int)
            or isinstance(stop, bool)
            or not isinstance(role, str)
        ):
            return None
        return AttentionSegment.for_role(start, stop, role)
    return None


def _topology_from_wire(value) -> AttentionTopology | None:
    if isinstance(value, AttentionTopology):
        return value
    if not isinstance(value, Mapping):
        return None
    try:
        return AttentionTopology(
            str(value["topology_id"]),
            str(value["kind"]),
            int(value["start"]),
            int(value["stop"]),
            int(value["tokens_per_frame"]),
            int(value["spatial_tokens_height"]),
            int(value["spatial_tokens_width"]),
            str(value.get("axis", "both")),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _parse_segments(values) -> tuple[AttentionSegment, ...] | None:
    if not isinstance(values, (tuple, list)) or not values:
        return None
    parsed = tuple(_segment_from_wire(value) for value in values)
    return None if any(value is None for value in parsed) else parsed


def _legacy_topology(layout: Mapping) -> tuple[AttentionTopology, ...] | None:
    names = (
        "topology_start_tokens",
        "topology_tokens",
        "tokens_per_frame",
        "spatial_tokens_height",
        "spatial_tokens_width",
    )
    values = tuple(layout.get(name) for name in names)
    if all(value is None for value in values):
        return ()
    if any(not isinstance(value, int) or isinstance(value, bool) for value in values):
        # Legacy modality segmentation did not require a spatial topology.
        # Ignore a partial v0 topology; protocol-v1 layouts remain strict.
        return ()
    start, tokens, frame_tokens, height, width = values
    return (
        AttentionTopology(
            "primary_video",
            "video_grid",
            start,
            start + tokens,
            frame_tokens,
            height,
            width,
        ),
    )


def parse_attention_semantic_layout(
    value,
    *,
    transformer_options: Mapping | None = None,
) -> AttentionSemanticLayout | None:
    if isinstance(value, AttentionSemanticLayout):
        if value.protocol_version != ATTENTION_LAYOUT_PROTOCOL_VERSION:
            return None
        layout = value
    elif isinstance(value, Mapping):
        provider = value.get("provider")
        protocol_version = value.get("protocol_version", 0)
        if not isinstance(provider, str) or not provider:
            # Versioned producers must identify themselves.  Provider-less v0
            # dictionaries were accepted by early experimental patch nodes and
            # remain readable as a bounded compatibility format.
            if protocol_version not in (0, None):
                return None
            provider = "legacy"
        if protocol_version == ATTENTION_LAYOUT_PROTOCOL_VERSION:
            query_segments = _parse_segments(value.get("query_segments"))
            key_segments = _parse_segments(value.get("key_segments"))
            topology_values = value.get("topologies", ())
            if not isinstance(topology_values, (tuple, list)):
                return None
            topologies = tuple(_topology_from_wire(item) for item in topology_values)
            if (
                query_segments is None
                or key_segments is None
                or any(item is None for item in topologies)
            ):
                return None
        elif protocol_version in (0, None):
            query_segments = _parse_segments(value.get("segments"))
            if query_segments is None:
                return None
            key_segments = query_segments
            topologies = _legacy_topology(value)
            if topologies is None:
                return None
            if len(topologies) == 1:
                topology = topologies[0]
                query_segments = tuple(
                    replace(segment, topology_id=topology.topology_id)
                    if segment.role in {"target_video", "reference_video"}
                    and segment.start >= topology.start
                    and segment.stop <= topology.stop
                    else segment
                    for segment in query_segments
                )
                key_segments = query_segments
        else:
            return None
        try:
            layout = AttentionSemanticLayout(
                provider,
                query_segments,
                key_segments,
                topologies,
                int(value.get("layer_index", -1)),
                int(value.get("layer_count", 0)),
            )
        except (TypeError, ValueError):
            return None
    else:
        return None

    if isinstance(transformer_options, Mapping):
        block_index = transformer_options.get("block_index")
        total_blocks = transformer_options.get("total_blocks")
        if (
            isinstance(block_index, int)
            and not isinstance(block_index, bool)
            and isinstance(total_blocks, int)
            and not isinstance(total_blocks, bool)
            and total_blocks > 0
            and 0 <= block_index < total_blocks
        ):
            layout = replace(
                layout,
                layer_index=block_index,
                layer_count=total_blocks,
            )
    return layout


def attention_semantic_layout(transformer_options) -> AttentionSemanticLayout | None:
    if not isinstance(transformer_options, Mapping):
        return None
    return parse_attention_semantic_layout(
        transformer_options.get(ATTENTION_LAYOUT_KEY),
        transformer_options=transformer_options,
    )


def has_complete_attention_layout(
    transformer_options,
    sequence_length: int | None = None,
    *,
    key_sequence_length: int | None = None,
    provider: str | None = None,
) -> bool:
    layout = attention_semantic_layout(transformer_options)
    if layout is None or (provider is not None and layout.provider != provider):
        return False
    if key_sequence_length is None:
        key_sequence_length = sequence_length
    return layout.validate(sequence_length, key_sequence_length) is None


__all__ = [
    "ATTENTION_LAYOUT_KEY",
    "ATTENTION_LAYOUT_PROTOCOL_VERSION",
    "ATTENTION_LAYOUT_REQUIREMENT_KEY",
    "ATTENTION_SEGMENT_ROLES",
    "AttentionSegment",
    "AttentionSemanticLayout",
    "AttentionTopology",
    "LayoutProviderStatus",
    "attention_semantic_layout",
    "ensure_attention_layout_provider",
    "has_complete_attention_layout",
    "parse_attention_semantic_layout",
    "register_attention_layout_provider",
]
