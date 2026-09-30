"""Include the canonical recipe/shell assets in wheels, including sdist builds."""

from pathlib import Path
from setuptools.command.build_py import build_py


class BuildPy(build_py):
    def run(self):
        super().run()
        for pattern in ("configs/*.json", "ops/train.sh", "ops/repo_env.sh"):
            for source in sorted(Path(".").glob(pattern)):
                destination = Path(self.build_lib) / "remax/_assets" / source
                destination.parent.mkdir(parents=True, exist_ok=True)
                self.copy_file(str(source), str(destination))
