"""Optional fresh directories for verification records; no retained-path writes."""

from pathlib import Path


def fresh_output(output, root):
    if output is None:
        return None
    requested = Path(output)
    if requested.exists() or requested.is_symlink():
        raise FileExistsError("Choose a fresh verification output directory: " + str(requested))
    output, root = requested.resolve(), Path(root).resolve()
    if root.is_relative_to(output) or (
        output.is_relative_to(root) and not output.is_relative_to(root / "tmp")
    ):
        raise ValueError("Inside the repository, verification outputs must be under tmp/")
    output.mkdir(parents=True, exist_ok=False)
    return output
