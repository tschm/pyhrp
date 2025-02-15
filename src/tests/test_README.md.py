import doctest


def test_docstring():
    doctest_results = doctest.testfile("../../README.md")
    assert doctest_results.failed == 0
