import tempfile
import unittest
from pathlib import Path

from livebridge_rl.cli import main
from livebridge_rl.data_contract import REQUIRED_FILES, validate_data_dir


def touch_files(root: Path, names: tuple[str, ...]) -> None:
    for name in names:
        (root / name).touch()


class DataContractTests(unittest.TestCase):
    def test_minimal_cdr_layout_is_valid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            touch_files(root, REQUIRED_FILES)

            report = validate_data_dir(root)

        self.assertTrue(report.valid_minimal_cdr)
        self.assertEqual(report.missing_required, ())
        self.assertEqual(set(report.present_required), set(REQUIRED_FILES))

    def test_missing_required_file_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            touch_files(root, REQUIRED_FILES[:-1])

            report = validate_data_dir(root)

        self.assertFalse(report.valid_minimal_cdr)
        self.assertEqual(report.missing_required, (REQUIRED_FILES[-1],))

    def test_cli_returns_nonzero_for_invalid_layout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(
                main(["validate-data", "--data-dir", directory]),
                1,
            )


if __name__ == "__main__":
    unittest.main()
