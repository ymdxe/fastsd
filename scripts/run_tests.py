"""Run FastSD CPU tests and check the pinned SpecEdge source and paper configs."""
import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--strict-official', action='store_true',
                        help='require Python 3.14 for the official source compilation')
    parser.add_argument('--test-python', default=sys.executable,
                        help='interpreter with the FastSD CPU test dependencies installed')
    args = parser.parse_args()
    official_python = sys.version_info[:2] == (3, 14)
    if args.strict_official and not official_python:
        parser.error('--strict-official requires Python 3.14')

    subprocess.run([sys.executable, 'baselines/specedge/repro.py', 'doctor'], cwd=ROOT, check=True)
    if official_python:
        subprocess.run([sys.executable, '-m', 'compileall', '-q', 'baselines/specedge/official/src'],
                       cwd=ROOT, check=True)
    else:
        print('Skipping official source compilation (requires Python 3.14).', flush=True)

    subprocess.run([args.test_python, '-m', 'compileall', '-q', 'baselines/specedge/integration',
                    'src', 'cloud', 'edge', 'benchmark', 'scripts'], cwd=ROOT, check=True)
    subprocess.run([args.test_python, '-m', 'unittest', 'discover', '-s', 'tests', '-v'],
                   cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
