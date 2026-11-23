"""Selection must not import unrelated tests or hide changed files."""

from scripts import test_suite as suite
import subprocess


def test_explicit_file_and_node_id_do_not_add_the_whole_directory():
    assert suite.explicit_targets(["tests", "-q"]) == ["tests"]
    assert suite.explicit_targets(
        ["-q", "tests/test_sigmas.py::test_empty_schedule_round_trips"]
    ) == ["tests/test_sigmas.py::test_empty_schedule_round_trips"]
    assert suite.explicit_targets(["-k", "tests/keyword", "-m", "gpu"]) == []


def test_api_uses_progress_and_native_http_tests_without_gpu_matrix():
    selected = suite.select(["comfyui_turing_utils/job_api/progress.py"])
    assert selected.areas == ("api",)
    assert any(path.startswith("tests/test_job_api.py::") for path in selected.targets)
    assert "tests/test_job_api.py" not in selected.targets
    assert not any("gpu/" in path for path in selected.targets)


def test_quantization_retains_independent_boundary_smoke():
    selected = suite.select(["comfyui_turing_utils/quantization/dispatch.py"])
    assert "tests/gpu/test_nvfp4_bounds.py" in selected.targets


def test_extracted_owners_retain_their_execution_and_persistence_coverage():
    assert suite.areas_for(
        "comfyui_turing_utils/adapters/minimax/sampling_masks.py"
    ) == {"sampling"}
    selected = suite.select(["comfyui_turing_utils/runtime/shared_loaders.py"])
    assert set(selected.areas) == {"sampling", "api"}
    assert "tests/sampling/test_h3_sampling.py" in selected.targets
    assert any(path.startswith("tests/test_job_api.py::") for path in selected.targets)



def test_shared_contracts_and_unknown_runtime_code_expand_scope():
    selected = suite.select(["comfyui_turing_utils/registration.py"])
    assert set(selected.areas) == set(suite.AREAS)
    assert selected.full_required
    assert suite.areas_for("comfyui_turing_utils/new_domain/logic.py") == set(
        suite.AREAS
    )


def test_docs_and_runner_changes_use_only_small_baseline():
    assert suite.select(["README.md", "scripts/test_suite.py"]).targets == suite.BASE


def test_duplicate_areas_never_duplicate_targets():
    selected = suite.select(["kernel/csrc/bindings.cpp"], ["quantization"])
    assert len(selected.targets) == len(set(selected.targets))
    assert selected.full_required


def test_pytest_output_paths_are_not_mistaken_for_test_targets():
    assert (
        suite.explicit_targets(
            ["--basetemp", "/cache/scratch", "--junitxml", "/cache/report.xml", "-q"]
        )
        == []
    )


def test_changed_paths_preserve_both_rename_owners_and_untracked_files(tmp_path):
    def git(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    (tmp_path / "old.py").write_text("original")
    git("add", ".")
    git(
        "-c",
        "user.name=Tests",
        "-c",
        "user.email=tests@example.invalid",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "-qm",
        "fixture",
    )
    git("mv", "old.py", "new.py")
    (tmp_path / "untracked.py").write_text("new test")
    assert suite.changed_paths(tmp_path) == ["new.py", "old.py", "untracked.py"]
