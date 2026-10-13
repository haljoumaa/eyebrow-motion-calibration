"""Strict source/protocol hashes and read-only verification outputs."""

import importlib.util
from pathlib import Path
import shutil
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import protocol_hashes as provenance
from verification_output import fresh_output


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


openface = load_script('verify_openface')
pilot = load_script('verify_pilot')
literature = load_script('reproduce_published_tables')
reproduction = load_script('reproduce')
sys.path.pop(0)


@pytest.fixture
def protocol(tmp_path):
    for relative in (provenance.PROTOCOL,):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    for relative in ('code.py', 'config.json'):
        (tmp_path / relative).write_text('fixed scientific input\n')
    hashes = {p: provenance.digest(tmp_path / p) for p in ('code.py', 'config.json')}
    hashes[provenance.PROTOCOL] = provenance.HISTORICAL
    return tmp_path, hashes


def test_exact_protocol_revision_and_replay_preserve_historical_hash(protocol):
    root, historical = protocol
    before = historical.copy()
    revisions = provenance.verify_source_hashes(root, historical)
    assert len(revisions) == 1 and revisions[0]['historical_sha256'] == provenance.HISTORICAL
    repeated = {p: provenance.digest(root / p) for p in historical}
    assert provenance.compare_source_hashes(root, historical, repeated) == revisions
    assert historical == before


@pytest.mark.parametrize('relative', [provenance.PROTOCOL, 'code.py', 'config.json'])
def test_unexpected_protocol_evidence_code_or_config_change_fails(protocol, relative):
    root, hashes = protocol
    with (root / relative).open('a') as stream:
        stream.write('unexpected change\n')
    with pytest.raises(ValueError, match='checksum mismatch'):
        provenance.verify_source_hashes(root, hashes)




def test_unknown_historical_hash_is_not_accepted(protocol):
    root, hashes = protocol
    hashes[provenance.PROTOCOL] = '0' * 64
    with pytest.raises(ValueError, match='Source checksum mismatch'):
        provenance.verify_source_hashes(root, hashes)


@pytest.mark.parametrize('change', ['missing', 'extra', 'forged', 'stale'])
def test_replay_hash_membership_and_actual_bytes_remain_strict(protocol, change):
    root, historical = protocol
    repeated = {p: provenance.digest(root / p) for p in historical}
    if change == 'missing':
        repeated.pop('code.py')
    elif change == 'extra':
        repeated['extra.py'] = '0' * 64
    elif change == 'forged':
        repeated['code.py'] = '0' * 64
    else:
        repeated[provenance.PROTOCOL] = provenance.HISTORICAL
    with pytest.raises(ValueError):
        provenance.compare_source_hashes(root, historical, repeated)


def test_default_output_does_not_create_directories(tmp_path):
    root = tmp_path / 'absent'
    assert fresh_output(None, root) is None
    assert not root.exists()


@pytest.mark.parametrize('kind', ['file', 'directory', 'dangling_symlink'])
def test_existing_output_is_rejected_without_changes(tmp_path, kind):
    path = tmp_path / 'kept'
    if kind == 'file':
        path.write_text('preserved')
    elif kind == 'directory':
        path.mkdir()
        (path / 'verification.json').write_text('preserved')
    else:
        path.symlink_to(tmp_path / 'missing')
    with pytest.raises(FileExistsError):
        fresh_output(path, tmp_path / 'repo')
    if kind == 'file':
        assert path.read_text() == 'preserved'
    elif kind == 'directory':
        assert (path / 'verification.json').read_text() == 'preserved'
    else:
        assert path.is_symlink() and not path.exists()


def test_retained_paths_and_aliases_are_protected(tmp_path):
    root = tmp_path / 'repo'
    root.mkdir()
    alias = tmp_path / 'alias'
    alias.symlink_to(root, target_is_directory=True)
    for path in (root / 'results/verification/new', alias / 'results/new'):
        with pytest.raises(ValueError, match='under tmp'):
            fresh_output(path, root)
    assert fresh_output(root / 'tmp/new', root) == root / 'tmp/new'
    assert fresh_output(tmp_path / 'external', root) == tmp_path / 'external'


@pytest.mark.parametrize('verifier', [openface.check, pilot.verify, literature.verify])
def test_all_original_verifiers_reject_overwrite_before_reading_inputs(tmp_path, verifier):
    destination = tmp_path / 'kept'
    destination.mkdir()
    sentinel = destination / 'verification.json'
    sentinel.write_text('historical bytes')
    with pytest.raises(FileExistsError):
        verifier(tmp_path / 'no_inputs', destination)
    assert sentinel.read_text() == 'historical bytes'


def test_combined_replay_rejects_changed_sources_before_fitting(tmp_path, monkeypatch):
    import json
    relative = 'code.py'
    (tmp_path / relative).write_text('changed code')
    metadata = tmp_path / 'results/openface/baselines/run_metadata.json'
    metadata.parent.mkdir(parents=True)
    metadata.write_text(json.dumps({'source_sha256': {relative: '0' * 64}}))
    def no_fit(*args, **kwargs):
        pytest.fail('Fitting must not start with an unexpected source hash')
    monkeypatch.setattr(reproduction, 'run_baselines', no_fit)
    with pytest.raises(ValueError, match='Source checksum mismatch'):
        reproduction.repeat(tmp_path / 'new-run', tmp_path)
    assert not (tmp_path / 'new-run').exists()
