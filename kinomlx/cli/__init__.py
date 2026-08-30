"""KinoMLX command-line interface.

The console script references :mod:`kinomlx.cli.main` directly. This package
deliberately re-exports nothing: a lazy ``main`` attribute collides with the
``kinomlx.cli.main`` submodule and can resolve to the module instead of the
callable when imported by a generated console-script wrapper.
"""
