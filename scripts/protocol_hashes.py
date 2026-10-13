"""Verify computational sources and retain historical documentation provenance.

Retired documentation is recognized only by the exact path/hash pairs below.
Code, configurations, inputs and fitted artifacts remain checksum-verified.
"""
import hashlib
from pathlib import Path

RETIRED_DOCUMENTATION = {'docs/openface-protocol.md': ('e8b19cc223175b1b334c5d40ba1563f76170a4abfde67a1de327b39a69934408',
                               '2d98ee257ff1bec127e1a5ec431b1d3caeb2868ff8518e7405d8e3f489b279b8',
                               '27543c5f27b75cfff141c90f9142663b7d62d02537d050cf50c520f1f1bfd947',
                               'cc717dfe848bc798a837f94d4834a46ef1c69b16f4c418b2174c1279ed72a253'),
 'docs/face-landmarker-protocol.md': ('117d4d7461944beb5badb9ce2a29f78f3777f9c847588657cd8c87459cc43865',
                                      '31983a5fa055473b5fbc5b55e34ef2d993e989ee18cd31a0714b3f607a6b0f81',
                                      '27f3a5b1d7ceba10d19c43d40c1a85d2ccc67168392e1b74d953d7d60228115e',
                                      'cc717dfe848bc798a837f94d4834a46ef1c69b16f4c418b2174c1279ed72a253'),
 'docs/methodology.md': ('e8b19cc223175b1b334c5d40ba1563f76170a4abfde67a1de327b39a69934408',
                         '2d98ee257ff1bec127e1a5ec431b1d3caeb2868ff8518e7405d8e3f489b279b8',
                         '27543c5f27b75cfff141c90f9142663b7d62d02537d050cf50c520f1f1bfd947',
                         '117d4d7461944beb5badb9ce2a29f78f3777f9c847588657cd8c87459cc43865',
                         '31983a5fa055473b5fbc5b55e34ef2d993e989ee18cd31a0714b3f607a6b0f81',
                         '27f3a5b1d7ceba10d19c43d40c1a85d2ccc67168392e1b74d953d7d60228115e',
                         'cc717dfe848bc798a837f94d4834a46ef1c69b16f4c418b2174c1279ed72a253')}
SOURCE_REVISIONS = {'src/facial_motion/openface_baselines.py': {'recorded': ['9c7559d3e53651c92230b0cebadb95f73ff048c055f828b5cfba743bbbee689e',
                                                          'ca0780da39d84db2d1e87871e2c60223b320fc3b25c55e7cfce617843fdd9e28'],
                                             'current': '113ca4d9c0ce3e16c8ff17087c707d36c2f1aa5ce0f9d461d0b36052c13fa6cf'},
 'src/facial_motion/openface_reproduction.py': {'recorded': ['8c818d2f6209cca3e00e23eb53c3768899a2669170f96e2d99dd27975ce38a0b',
                                                             'a91e64b2a4c5deffb9ae9a6efb5022009234c8b1a6f8fb82bbb4105b8296d9d7'],
                                                'current': 'e44aeb87ee4a9a71bfa24f8477d0d2d47a05569191bb2997d5764b163d58164a'},
 'src/facial_motion/openface_recording.py': {'recorded': ['3708728a353c815608a25570cfbf2c0d69bd18c21d7fd0ea122da189d45d1da8'],
                                             'current': '3ad89402e09a5ac62ba421cff8b39eafdec57e0c7e01684022f14973fb1df0df'},
 'src/facial_motion/evaluate.py': {'recorded': ['76054c153fd03063c33a2fbe40ff72324512bd671fae189009191e8e4926b413',
                                                'df344aad034518489b07503cf85661e01cb0a301b637f634dc094a01c986a823'],
                                   'current': 'c84e7dec5c3a3cfbbc3bc6959aa1db1d1ec206f92a4a4fc18a0a6564dbbc3ad6'},
 'scripts/mlp_followup/verify_followup.py': {'recorded': ['c5d0cf47f20227c22afb4a36ae4fc120c90bbc04b4c71bb854ecc9687e5fe794'],
                                             'current': '659513f128047b6532480fcbd9e7e8539952f44f1d659f3d6d67061a8be4c03e'},
 'src/facial_motion/__init__.py': {'recorded': ['93875984349abc858d40cac369ad7fadccbf10fd125c85085f313355ae96a201'],
                                   'current': '289af802bdd6bdb5ee73ca987a966a2404e7a50e03dd53f738906142e41fde92'},
 'src/facial_motion/cli.py': {'recorded': ['69963b7e31ccc09c74878b0027a574079f1d2564eb2bdee314e9716eb56e1580'],
                              'current': '6055303c21b66437313f84884d4cc1b6e40c1bc7035bc81eefb63f3756840dcc'},
 'scripts/mlp_followup/summarize.py': {'recorded': ['3285c272a32021a2862b0a016702bbb83942ad5f94a2aac0afbfc658fd57f145'],
                                       'current': '816c72e48e18c0dc431c83a44a25b99542c615402aee11834615039b48ad71e5'},
 'tests/test_openface_baselines.py': {'recorded': ['d42413452385a4be7d79b6be3dbb2d6d78796c5626a6bfbf7144efff499d0376'],
                                          'current': '11a92f749050e3432271898a2ffa62eb4bf2a8c71ba2b2fb5a2f3fc94509e095'}}
# Publication-only revisions: local-input messages, explicit absent-input skips,
# and local pilot figure IDs supplied as arguments. Estimation, scientific
# assertions and tolerances are unchanged; only these exact bytes are accepted.
SOURCE_REVISIONS.update({'src/facial_motion/acquire.py': {'recorded': ['03d59f4a1be63447e83a68e80991f3a23f7919881f1c83163223ee5717515e21', '56746045e612eb5025e0dd2f36cf29bc1bde51f653e8bb62fcbe6b64fcc45152'], 'current': '56746045e612eb5025e0dd2f36cf29bc1bde51f653e8bb62fcbe6b64fcc45152'}, 'src/facial_motion/figures.py': {'recorded': ['1e5b57214f84cef39e548dccbd720902a0532d841256bb10ba559b9caa6a018c'], 'current': '36850256a08f714e75327f460dca090c2a26ef872bfa6e6dc3540e6cfe24e4b3'}, 'src/facial_motion/cli.py': {'recorded': ['69963b7e31ccc09c74878b0027a574079f1d2564eb2bdee314e9716eb56e1580', '6055303c21b66437313f84884d4cc1b6e40c1bc7035bc81eefb63f3756840dcc'], 'current': 'ab7b2842c828c75470e90c873b0ad96f1d8af3c4966f504a782ac47ffae491e8'}, 'tests/test_openface_baselines.py': {'recorded': ['d42413452385a4be7d79b6be3dbb2d6d78796c5626a6bfbf7144efff499d0376', '11a92f749050e3432271898a2ffa62eb4bf2a8c71ba2b2fb5a2f3fc94509e095', 'a0482316de3a49cf18eaacd7d64d97997d44b2802a9ceb99d921c6b71c41353a'], 'current': 'a0482316de3a49cf18eaacd7d64d97997d44b2802a9ceb99d921c6b71c41353a'}, 'tests/test_openface_recording.py': {'recorded': ['6d7be861e663a2a7c06528d386852a19e663ecfe096aea766f1101a877df320a', '33aabb89d4b60b68b18f0215356b3a688837164b65c89ad490797de966978ae7'], 'current': '33aabb89d4b60b68b18f0215356b3a688837164b65c89ad490797de966978ae7'}, 'tests/test_openface_reproduction.py': {'recorded': ['aa639d7ac25732cb4cd93a7ddaee536809c683a49eacc12431f3e5f98e21efc6', '2720e0a4300268c030663e852be781e257570cb74f21fe97a48be3572a5dd853'], 'current': '2720e0a4300268c030663e852be781e257570cb74f21fe97a48be3572a5dd853'}})

# Documentation removal and reader-facing labels; fitting and scoring are unchanged.
SOURCE_REVISIONS.update({
    'src/facial_motion/openface_baselines.py': {'recorded': ['9c7559d3e53651c92230b0cebadb95f73ff048c055f828b5cfba743bbbee689e', 'ca0780da39d84db2d1e87871e2c60223b320fc3b25c55e7cfce617843fdd9e28', '113ca4d9c0ce3e16c8ff17087c707d36c2f1aa5ce0f9d461d0b36052c13fa6cf'], 'current': 'e6e550435d80c3f3b8807478092819b129dd97eb084f8aa4d8bc9af9e5aad4fb'},
    'src/facial_motion/openface_reproduction.py': {'recorded': ['8c818d2f6209cca3e00e23eb53c3768899a2669170f96e2d99dd27975ce38a0b', 'a91e64b2a4c5deffb9ae9a6efb5022009234c8b1a6f8fb82bbb4105b8296d9d7', 'e44aeb87ee4a9a71bfa24f8477d0d2d47a05569191bb2997d5764b163d58164a'], 'current': '9dccc886c9a5846c17ee7f6bcf26204c7e5e25b6305308fe1e8209e1e6671de9'},
    'src/facial_motion/openface_recording.py': {'recorded': ['3708728a353c815608a25570cfbf2c0d69bd18c21d7fd0ea122da189d45d1da8', '3ad89402e09a5ac62ba421cff8b39eafdec57e0c7e01684022f14973fb1df0df'], 'current': 'a8c2fd9653ca004d16d38a358da279ed6e634ca19bdbc03be9196603dc9ffef3'},
    'src/facial_motion/evaluate.py': {'recorded': ['76054c153fd03063c33a2fbe40ff72324512bd671fae189009191e8e4926b413', 'df344aad034518489b07503cf85661e01cb0a301b637f634dc094a01c986a823', 'c84e7dec5c3a3cfbbc3bc6959aa1db1d1ec206f92a4a4fc18a0a6564dbbc3ad6'], 'current': 'a9a5054840aeae8548b84897d85ace2e0755a8d38b6683d77ab5c114c061c651'},
    'scripts/mlp_followup/summarize.py': {'recorded': ['3285c272a32021a2862b0a016702bbb83942ad5f94a2aac0afbfc658fd57f145', '816c72e48e18c0dc431c83a44a25b99542c615402aee11834615039b48ad71e5'], 'current': 'a58036450522bea7bc56022c26a03d2b5cd067b32d5e775ea84a21f14a997e94'},
})


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_path(root, relative):
    requested = Path(relative)
    path = root / requested
    if requested.is_absolute() or not path.resolve().is_relative_to(root):
        raise ValueError("Unsafe source path: " + relative)
    return path


def accepted_revision(relative, expected, actual):
    revision = SOURCE_REVISIONS.get(relative)
    return revision and expected in revision['recorded'] and actual == revision['current']


def verify_source_hashes(root, hashes):
    root = Path(root).resolve()
    revisions = []
    for relative, expected in hashes.items():
        path = source_path(root, relative)
        retired = RETIRED_DOCUMENTATION.get(relative)
        if retired is not None:
            if expected not in retired or (path.exists() and digest(path) != expected):
                raise ValueError("Source checksum mismatch: " + relative)
            revisions.append(dict(path=relative, historical_sha256=expected,
                                  status="retired_documentation"))
            continue
        actual = digest(path)
        if actual == expected:
            continue
        if accepted_revision(relative, expected, actual):
            revisions.append(dict(path=relative, current_path=str(path.relative_to(root)),
                                  historical_sha256=expected, current_sha256=actual))
        else:
            raise ValueError("Source checksum mismatch: " + relative)
    return revisions


def compare_source_hashes(root, historical, repeated):
    root = Path(root).resolve()
    def canonical(hashes):
        result = {}
        for relative, expected in hashes.items():
            target = str(source_path(root, relative).relative_to(root))
            if relative in RETIRED_DOCUMENTATION:
                continue
            if target in result and result[target] != expected:
                raise ValueError("Conflicting source hashes: " + target)
            result[target] = expected
        return result
    revisions = verify_source_hashes(root, historical)
    if any(relative in RETIRED_DOCUMENTATION for relative in repeated):
        raise ValueError("Replay includes retired documentation")
    if set(canonical(historical)) != set(canonical(repeated)):
        raise ValueError("Source hash membership mismatch")
    current = canonical(repeated)
    for relative, expected in repeated.items():
        actual = digest(source_path(root, relative))
        if actual != expected:
            raise ValueError("Replay source checksum mismatch: " + relative)
    for relative, expected in historical.items():
        if relative in RETIRED_DOCUMENTATION:
            continue
        actual = current[str(source_path(root, relative).relative_to(root))]
        if expected != actual and not accepted_revision(relative, expected, actual):
            raise ValueError("Replay source differs: " + relative)
    return revisions
