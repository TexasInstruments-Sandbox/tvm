"""Pytest configuration for wheel packaging tests."""


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "quick: mark test as quick (compile + run under 5 minutes)",
    )
