import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "experiments" / "setup_specedge_locked_env.sh"


class SpecEdgeLockedEnvironmentSetupTests(unittest.TestCase):
    def test_script_is_explicit_locked_and_non_replacing(self):
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('DEFAULT_ENV_DIR="/home/hdd/zhangh/envs/specedge"', source)
        self.assertIn('--python must explicitly name a Python 3.14 executable.', source)
        self.assertIn('[[ "${PYTHON_BIN}" == /* ]]', source)
        self.assertIn('[[ "${python_version}" == "3.14" ]]', source)
        self.assertIn('assert_fresh_path "${DEFAULT_ENV_DIR}"', source)
        self.assertIn('"${UV_BIN}" lock --check', source)
        self.assertIn('UV_PROJECT_ENVIRONMENT="${DEFAULT_ENV_DIR}" "${UV_BIN}" sync --frozen --no-dev --python "${PYTHON_BIN}"', source)
        self.assertNotIn("rm -rf", source)
        self.assertLess(source.index('assert_fresh_path "${DEFAULT_ENV_DIR}"'), source.index('"${UV_BIN}" sync --frozen'))

    def test_official_pin_requires_python_314(self):
        pyproject = (REPO / "baselines" / "specedge" / "official" / "pyproject.toml").read_text(encoding="utf-8")
        lockfile = (REPO / "baselines" / "specedge" / "official" / "uv.lock").read_text(encoding="utf-8")
        self.assertIn('requires-python = "~=3.14.0"', pyproject)
        self.assertIn('requires-python = "==3.14.*"', lockfile)


if __name__ == "__main__":
    unittest.main()
