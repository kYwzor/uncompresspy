"""
uncompresspy is a package that is able to uncompress LZW files (.Z extension). It allows transparent access to the
uncompressed file contents and partial decompression.
"""
from .uncompresspy import LZWFile, open, extract, from_buffer
