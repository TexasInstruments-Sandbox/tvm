"""Pytest configuration for the accuracy harness tests."""


def pytest_configure(config):
    config.addinivalue_line("markers", "quick: fast test, included in pre-push regression")
