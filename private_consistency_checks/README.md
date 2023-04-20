# Private consistency checks
This is a small Rust crate with a Python wrapper that implements private consistency checks. It checks whether a transaction corresponds to a bank's database (multiple transactions at the same time).

It uses https://www.maturin.rs/ for building. The build command is `maturin build`, after which the Python package can be imported as `from private_consistency_checks import ...`.
