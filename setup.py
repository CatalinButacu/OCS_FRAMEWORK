from setuptools import setup, find_packages

setup(
    name="TUIASI_OCS_TEAM11_2025",  # This is the name that will appear on PyPI
    version="0.1.0",
    author="Roxana Dobre, Radu Bălăiță, Cătălin Butacu",  # List all authors here
    author_email="roxana-elena.dobre@student.tuiasi.ro, radu-ionut.balaita@student.tuiasi.ro, ionel-catalin.butacu@student.tuiasi.ro",  # Add all emails (comma-separated)
    description="A Python package for optimization algorithms developed by Team 11 | TUIASI 2025",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/CatalinButacu/OCS_FRAMEWORK.git",  # Replace with your actual GitHub repo URL
    packages=find_packages(),  # Automatically finds the `framework` package
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "pandas>=1.3.0",
        "scikit-learn>=0.24.0",
        "pytest>=6.2.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)