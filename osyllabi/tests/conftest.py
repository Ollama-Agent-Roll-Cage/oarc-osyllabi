"""
Pytest configuration for Osyllabi tests.
"""
import pytest

def pytest_addoption(parser):
    """Add custom command line options to pytest."""
    parser.addoption(
        "--live", 
        action="store_true",
        default=False,
        help="Run tests against a live Ollama instance"
    )

def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", 
        "live_only: mark test to run only when --live option is provided"
    )

def pytest_collection_modifyitems(config, items):
    """Skip tests marked as live_only unless --live is specified."""
    if not config.getoption("--live"):
        skip_live = pytest.mark.skip(reason="Live Ollama tests are disabled")
        for item in items:
            if "live_only" in item.keywords:
                item.add_marker(skip_live)
