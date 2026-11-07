# Contributing to scalegeom

Thank you for your interest. Bug reports, questions, examples from your own data, and pull requests are all welcome.

## Reporting a problem or asking a question

Open an issue on GitHub. For a bug, include the scalegeom version (`scalegeom.__version__`), your Python version and operating system, and a short script that reproduces the problem, ideally on simulated data from `scalegeom.gaussian_random_field`.

## Proposing a change

1. Open an issue first for anything larger than a small fix, so we can agree on the approach.
2. Fork the repository and create a branch.
3. Install in development mode: `pip install -e ".[test]"`.
4. Add or update tests in `tests/` and make sure `pytest` passes.
5. Describe the change in `CHANGELOG.md` under "Unreleased" and open a pull request.

New diagnostics, induced metrics, or family builders are especially welcome. Please keep the core free of dependencies beyond NumPy, SciPy and Matplotlib; optional features can use extras in `pyproject.toml`.

## Support and governance

scalegeom is maintained by Eric Vaz (Toronto Metropolitan University). I aim to respond to issues within two weeks. Design decisions follow the method in Vaz (2026), Geographical Analysis 58: e70061; changes that alter published numerical results are flagged in the changelog.

## Code of conduct

Be respectful and constructive. Harassment or discrimination of any kind is not tolerated. Concerns can be raised privately with the maintainer at evaz@torontomu.ca.
