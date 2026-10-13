"""Output protection and fitted-plan provenance checks."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts/mlp_followup"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("mlp_followup_verifier", SCRIPTS / "verify_followup.py")
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)
sys.path.pop(0)


def test_output_rejects_existing_directory_and_preserves_contents(tmp_path):
    kept = tmp_path / "kept"
    kept.mkdir()
    (kept / "sentinel").write_text("preserve")
    with pytest.raises(FileExistsError):
        verifier.fresh_output(kept, tmp_path / "repo")
    assert (kept / "sentinel").read_text() == "preserve"


def test_output_rejects_retained_paths_and_symlink_aliases(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(repo, target_is_directory=True)
    for output in [repo / "results/new", alias / "results/new", repo, tmp_path]:
        with pytest.raises((ValueError, FileExistsError)):
            verifier.fresh_output(output, repo)
    fresh = verifier.fresh_output(repo / "tmp/new", repo)
    assert fresh == repo / "tmp/new"


def test_output_rejects_overlap_with_inputs(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    with pytest.raises(ValueError, match="overlaps"):
        verifier.fresh_output(inputs / "new", tmp_path / "repo", inputs)


@pytest.fixture
def provenance(tmp_path):
    repo = tmp_path / "repo"
    active = repo / "scripts/mlp_followup/followup.py"
    active.parent.mkdir(parents=True)
    active.write_text("adapted implementation\n")
    historical_hash = hashlib.sha256(b'original prefit implementation\n').hexdigest()
    folds = repo / "configs/recording-folds.csv"
    folds.parent.mkdir()
    folds.write_text("video_name,validation_fold\nexample,0\n")
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    plan = dict(exploratory=True, source=dict(script_sha256=historical_hash, folds_sha256=digest(folds)))
    record = dict(
        fixed_inputs_sha256={folds.relative_to(repo).as_posix(): digest(folds)},
        prefit_script_sha256=historical_hash,
        active_fit_script_sha256=digest(active),
        implementation_sha256={active.relative_to(repo).as_posix(): digest(active)},
        result_sha256={},
    )
    (repo / "research").mkdir()
    (repo / "research/mlp-provenance.json").write_text(json.dumps(record))
    return repo, plan, active, folds


def test_documented_adaptation_preserves_prefit_hash(provenance):
    repo, plan, *_ = provenance
    checked = verifier.verify_implementation(repo, repo / "fresh-results", plan)
    assert checked["prefit_script_sha256"] == plan["source"]["script_sha256"]
    assert checked["active_script_sha256"] != checked["prefit_script_sha256"]


@pytest.mark.parametrize("changed", ["active", "folds", "plan"])
def test_unrecorded_implementation_or_prefit_changes_fail(provenance, changed):
    repo, plan, active, folds = provenance
    if changed == "plan":
        plan["source"]["script_sha256"] = "0" * 64
    else:
        {"active": active, "folds": folds}[changed].write_text("unrecorded change\n")
    with pytest.raises(AssertionError):
        verifier.verify_implementation(repo, repo / "fresh-results", plan)
