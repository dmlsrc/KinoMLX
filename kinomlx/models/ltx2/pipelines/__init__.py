"""Generation pipelines for the LTX-2 model.

Each file in this package is a top-level pipeline composed from the
pure functions in ``../denoise.py`` / ``../encode.py`` / ``../decode.py``
/ ``../state.py``.  Pipelines slot in as siblings - adding a new one
doesn't touch the others.

Currently ships:

- :mod:`distilled` - distilled two-stage text/image-to-video (8 + 3 steps).
"""
