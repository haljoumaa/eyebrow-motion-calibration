"""Strict source hashes, retired documentation provenance and read-only verification outputs."""

import importlib.util
from pathlib import Path
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


sys.path.pop(0)


RETIRED_PATH = 'docs/openface-protocol.md'
RETIRED_HASH = provenance.RETIRED_DOCUMENTATION[RETIRED_PATH][0]


@pytest.fixture
def protocol(tmp_path):
    for relative in ('code.py', 'config.json'):
        (tmp_path / relative).write_text('fixed scientific input\n')
    hashes = {p: provenance.digest(tmp_path / p) for p in ('code.py', 'config.json')}
    hashes[RETIRED_PATH] = RETIRED_HASH
    return tmp_path, hashes


def test_retired_documentation_and_replay_preserve_historical_hash(protocol):
    root, historical = protocol
    before = historical.copy()
    revisions = provenance.verify_source_hashes(root, historical)
    assert revisions == [dict(path=RETIRED_PATH, historical_sha256=RETIRED_HASH,
                              status='retired_documentation')]
    repeated = {p: provenance.digest(root / p) for p in ('code.py', 'config.json')}
    assert provenance.compare_source_hashes(root, historical, repeated) == revisions
    assert historical == before


@pytest.mark.parametrize('relative', ['code.py', 'config.json'])
def test_unexpected_code_or_config_change_fails(protocol, relative):
    root, hashes = protocol
    with (root / relative).open('a') as stream:
        stream.write('unexpected change\n')
    with pytest.raises(ValueError, match='checksum mismatch'):
        provenance.verify_source_hashes(root, hashes)


def test_retired_document_present_with_different_bytes_fails(protocol):
    root, hashes = protocol
    path = root / RETIRED_PATH
    path.parent.mkdir()
    path.write_text('unexpected document\n')
    with pytest.raises(ValueError, match='Source checksum mismatch'):
        provenance.verify_source_hashes(root, hashes)


def test_unknown_historical_hash_is_not_accepted(protocol):
    root, hashes = protocol
    hashes[RETIRED_PATH] = '0' * 64
    with pytest.raises(ValueError, match='Source checksum mismatch'):
        provenance.verify_source_hashes(root, hashes)


def test_unlisted_documentation_cannot_bypass_source_checks(protocol):
    root, hashes = protocol
    hashes['docs/other.md'] = RETIRED_HASH
    with pytest.raises(FileNotFoundError):
        provenance.verify_source_hashes(root, hashes)


@pytest.mark.parametrize('change', ['missing', 'extra', 'forged', 'retired'])
def test_replay_hash_membership_and_actual_bytes_remain_strict(protocol, change):
    root, historical = protocol
    repeated = {p: provenance.digest(root / p) for p in ('code.py', 'config.json')}
    if change == 'missing':
        repeated.pop('code.py')
    elif change == 'extra':
        repeated['extra.py'] = '0' * 64
    elif change == 'forged':
        repeated['code.py'] = '0' * 64
    else:
        repeated[RETIRED_PATH] = RETIRED_HASH
    with pytest.raises(ValueError):
        provenance.compare_source_hashes(root, historical, repeated)


def test_retired_documentation_does_not_allow_unsafe_paths(protocol):
    root, hashes = protocol
    hashes['../docs/openface-protocol.md'] = RETIRED_HASH
    with pytest.raises(ValueError, match='Unsafe source path'):
        provenance.verify_source_hashes(root, hashes)


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


@pytest.mark.parametrize('script_name', ['verify_openface', 'verify_pilot', 'reproduce_published_tables'])
def test_all_original_verifiers_reject_overwrite_before_reading_inputs(tmp_path, script_name):
    module = load_script(script_name)
    verifier = module.check if script_name == 'verify_openface' else module.verify
    destination = tmp_path / 'kept'
    destination.mkdir()
    sentinel = destination / 'verification.json'
    sentinel.write_text('historical bytes')
    with pytest.raises(FileExistsError):
        verifier(tmp_path / 'no_inputs', destination)
    assert sentinel.read_text() == 'historical bytes'


def test_combined_replay_rejects_changed_sources_before_fitting(tmp_path, monkeypatch):
    import json
    reproduction = load_script('reproduce')
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
