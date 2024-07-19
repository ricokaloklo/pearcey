import setuptools
from pathlib import Path

setuptools.setup(
    name="pearcey",
    version="0.1.0",
    description="A python package to compute the Pearcey function/integral in catastrophe optics",
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    packages=[
        "pearcey",
    ],
    install_requires=[
        "numpy",
        "scipy",
    ],
    classifiers=[
        "License :: OSI Approved :: MIT License",
    ],
    python_requires='>=3.7',
)