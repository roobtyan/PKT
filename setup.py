# setup.py
from pathlib import Path

from setuptools import find_packages, setup


def read_requirements():
    req_file = Path(__file__).with_name("requirements.txt")
    if not req_file.exists():
        return []
    return [
        line.strip()
        for line in req_file.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]


setup(
    name="pkt",
    version="0.1.0",
    description="PKT visualizations and training utilities",
    packages=find_packages(exclude=("tests", "docs")),
    install_requires=read_requirements(),
    python_requires=">=3.8",
    include_package_data=True,
)
