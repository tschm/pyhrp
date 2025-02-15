import doctest
import os

# Print the current working directory
print("Current working directory:", os.getcwd())


def test_docstring():
    doctest_results = doctest.testfile("../../README.md")
    print("Current working directory:", os.getcwd())
    assert doctest_results.failed == 0
