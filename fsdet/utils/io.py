from pathlib import Path


def get_project_root() -> Path:
    """
    Return the root director of the project.
    """
    return Path(__file__).parent.parent.parent
