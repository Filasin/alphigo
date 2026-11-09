"""The training package and its method subpackage are importable."""

def test_training_package_imports():
    import alphigo.training  # noqa: F401

def test_method_subpackage_imports():
    import alphigo.training.method  # noqa: F401
