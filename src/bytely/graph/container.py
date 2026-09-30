"""Recognition of container configuration files."""

from pathlib import Path


def is_container_config_file(path: str | Path) -> bool:
    """Return whether a path names a Docker or Compose configuration file."""
    name = Path(path).name.lower()
    return (
        name == "dockerfile"
        or name.startswith("dockerfile.")
        or name
        in {
            "compose.yaml",
            "compose.yml",
            "docker-compose.yaml",
            "docker-compose.yml",
        }
    )
