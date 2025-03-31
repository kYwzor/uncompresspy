# uncompresspy

uncompresspy is a pure Python package for uncompressing LZW files (.Z), such as the ones created by Unix's shell tool
`compress`. It is available on PyPI for Python 3.10+.

### Key features

While there are other Python packages for uncompressing LZW files, uncompresspy distinguishes itself from others on the
following aspects:

* It is implemented purely in Python and requires no external dependencies (thus fully cross-platform).
* Full compatibility with all files created by `compress` (i.e., same output as `uncompress`).
* Does not require holding the entire input file in memory (the file is read as necessary).
* Faster than all other available pure Python implementations (compared to the next fastest you can expect a speedup of
  at least 1.5x in poorly compressed files, but upwards of 44x on extremely compressed files).
* Allows partial/on-demand decompression and does not require holding the full decompressed stream in memory (only
  package to do so).
* Provides a file-like interface (only package to do so).

### Caveats

* uncompresspy only provides an implementation for decompression, other packages may provide compression features.
* Although extremely optimized, uncompresspy still lags behind C/C++ based implementations when decompressing most
  files, due to the inherent speed disadvantages of the language.
* To speed up the decompression, uncompresspy stores dynamically-sized bytearrays in its dictionary, rather than the
  classic technique of building a stack and then reversing it. This means that in some scenarios (particularly highly
  compressed files) memory usage may be higher than in other packages (though this is usually offset by the other memory
  management techniques mentioned above).