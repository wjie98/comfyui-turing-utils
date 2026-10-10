from comfyui_turing_utils.adapters.minimax.veda.selection import (
    choose_score_rows,
    compact_reduces_groups,
    estimate_workspace_bytes,
)


def test_veda_automatic_projection_requires_pressure_and_budget():
    from comfyui_turing_utils.adapters.minimax.veda.selection import (
        choose_projected_chunk,
    )

    common = dict(
        rows=100000,
        heads=14,
        dim=128,
        hidden=7168,
        element_size=2,
        workspace=400 * 1024**2,
    )
    full = common["workspace"] + 3 * 100000 * 14 * 128 * 2
    reserve = 64 * 1024**2
    assert choose_projected_chunk(**common, available=full + reserve) == 0
    pack = 3 * 14 * 128 * (7168 + 8)
    per_row = 7168 * 5 + 14 * 128 * 12 + 16
    for tiles in (64, 32, 16, 8, 4, 1):
        available = reserve + common["workspace"] + pack + tiles * 128 * per_row
        assert choose_projected_chunk(**common, available=available) == tiles
    assert choose_projected_chunk(**common, available=reserve) == 0


def test_veda_score_chunk_growth_preserves_head_group_budget():
    base = 10 * 1024**2
    extra = (241 - 128) * 241 * 64
    assert choose_score_rows(241, 5, base * 5, base) == 128
    assert choose_score_rows(241, 5, (base + extra) * 5 - 1, base) == 128
    assert choose_score_rows(241, 5, (base + extra) * 5, base) == 241
    assert choose_score_rows(17, 5, base * 5, base) == 17
    assert choose_score_rows(1000, 5, 1024**3, base) == 256
    assert choose_score_rows(241, 0, 1024**3, base) == 128


def test_veda_compact_requires_fewer_actual_groups():
    assert not compact_reduces_groups(
        14, 100, 10, 8
    )  # 10 -> 12 heads, still two groups
    assert compact_reduces_groups(14, 115, 10, 8)  # 11 -> 14 heads, one group saved
    assert not compact_reduces_groups(14, 200, 10, 8)  # both already fit
    assert compact_reduces_groups(1, 8, 10, 8)  # one full head cannot fit
    assert not compact_reduces_groups(1, 7, 10, 8)  # neither can fit
    assert not compact_reduces_groups(14, -1, 10, 8)


def test_veda_head_major_estimate_removes_value_copy_but_counts_fp32_casts():
    common = dict(
        slots=4096,
        video_tiles=31,
        heads=2,
        head_dim=128,
        projection_bytes_per_head=1024,
        score_rows=31,
        fused_prepare=True,
    )
    nhd = estimate_workspace_bytes(**common, element_size=2)
    hnd = estimate_workspace_bytes(**common, element_size=2, head_major_prepare=True)
    assert nhd - hnd == 4096 * 128 * 2 * 2
    fp32 = estimate_workspace_bytes(**common, element_size=4, head_major_prepare=True)
    assert fp32 - hnd == 4096 * 128 * 2 * (20 - 8)
