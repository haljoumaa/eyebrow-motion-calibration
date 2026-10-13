"""Repository defaults and fresh-output safeguards shared by the follow-up commands."""
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def fresh_output(output, repo, *inputs):
    output, repo = Path(output).resolve(), Path(repo).resolve()
    if output.exists():
        raise FileExistsError(f"Choose a fresh output directory: {output}")
    if repo.is_relative_to(output) or (
        output.is_relative_to(repo) and not output.is_relative_to(repo / "tmp")
    ):
        raise ValueError("Inside the repository, generated outputs must be under tmp/")
    for source in inputs:
        source = Path(source).resolve()
        if output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError(f"Output overlaps an input directory: {source}")
    output.mkdir(parents=True, exist_ok=False)
    return output
