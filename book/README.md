se# Project Book and Documentation

This directory contains the source and templates for generating the Rhiza companion book and API documentation.

## Structure

- `marimo/`: Interactive [Marimo](https://marimo.io/) notebooks that are included in the book.
- `minibook-templates/`: Jinja2 templates for the minibook generation.
- `pdoc-templates/`: Custom templates for [pdoc](https://pdoc.dev/) API documentation.
- `Makefile.book`: Specialized Makefile for building the book and documentation.

## Building the Book

You can build the complete documentation book using the main project Makefile:

```bash
make book
```

This process involves:
1. Exporting Marimo notebooks to HTML.
2. Generating API documentation from the source code.
3. Combining them into a cohesive "book" structure.
