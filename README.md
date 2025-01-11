# TUIASI_OCS_TEAM11_2025

A Python framework for optimization algorithms developed by Team 11.

---

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
  - [Running Examples](#running-examples)
  - [Developing with the Framework](#developing-with-the-framework)
- [Folder Structure](#folder-structure)
- [Contributing](#contributing)
- [License](#license)

---

## Installation

To set up the project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/CatalinButacu/OCS_FRAMEWORK.git
   cd TUIASI_OCS_TEAM11_2025
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment:

   On **Windows**:
   ```bash
   venv\Scripts\activate
   ```

   On **macOS/Linux**:
   ```bash
   source venv/bin/activate
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Install the framework in editable mode:
   ```bash
   pip install -e .
   ```

   This ensures that changes to the framework code are reflected immediately.

---

## Usage

### Running Examples

To run the example scripts located in the `examples` folder, use the following command:
```bash
python examples/example_canonical_ga.py
```

### Developing with the Framework

You can import the framework in your scripts like this:
```python
from framework.algorithms.canonical_ga import CGA, CGAAdaptiveV2, CGAGreedy
```

---

## Folder Structure

```
TUIASI_OCS_TEAM11_2025/
├── framework/               # Core framework code
│   ├── algorithms/          # Optimization algorithms
│   ├── utils/               # Utility functions
│   └── __init__.py          # Package initialization
├── examples/                # Example scripts
│   └── example_canonical_ga.py
├── requirements.txt         # Project dependencies
├── setup.py                 # Package installation script
└── README.md                # This file
```

---

## Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add your message here"
   ```
4. Push to the branch:
   ```bash
   git push origin feature/your-feature-name
   ```
5. Open a pull request.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
