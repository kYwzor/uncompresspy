import io
import os
import warnings
from bisect import bisect
from typing import BinaryIO

_INITIAL_CODE_WIDTH = 9
_INITIAL_MASK = 2 ** _INITIAL_CODE_WIDTH - 1
_CLEAR_CODE = 256
_MAGIC_BYTE0 = 0x1F
_MAGIC_BYTE1 = 0x9D
_BLOCK_MODE_FLAG = 0x80
_CODE_WIDTH_FLAG = 0x1F
_UNKNOWN_FLAGS = 0x60


class LZWFile(io.RawIOBase):
    """
    A file-like object that transparently decompresses an LZW-compressed file on the fly. Supports reading and seeking.
    It does not load the entire compressed file into memory, and supports incremental reads via the read() method.
    It only supports a binary file interface (i.e. data read is always returned as bytes).
    Context management is supported.

    Examples
    --------
    >>> with uncompresspy.LZWFile('example.txt.Z', 'rb') as f:
    ...     first_ten_bytes = f.read(10)
    ...     rest_of_file = f.read()
    """

    def __init__(self, filename: str | bytes | os.PathLike | BinaryIO, mode='rb', warn_truncation=True,
                 save_checkpoints=True):
        """
        Open an LZW-compressed file in binary mode.

        Parameters
        ----------
        filename : str, bytes, os.PathLike, or BinaryIO
            If filename is a str, bytes, or PathLike object, it represents the file path to be opened and read from.
            Otherwise, it must be a file-like object (supporting read and seek operations).
        mode : {'r', 'rb'}, optional
            File mode. Only reading modes are supported. Defaults to 'rb'.
        warn_truncation : bool, optional
            If True, warns about potential truncation of the file. Defaults to True.
        save_checkpoints : bool, optional
            If True, save checkpoint positions to improve seeking performance at a small memory hit. Defaults to True.

        Raises
        ------
        ValueError
            If an unsupported mode is provided or if the underlying file object is not readable or seekable.
        TypeError
            If the filename is not of an expected type.
        """
        self._file = None
        self._close_file: bool = False
        if mode not in ('', 'r', 'rb'):
            # We always operate in binary mode
            raise ValueError(f"Invalid mode: {mode!r} (only reading is supported)")

        if isinstance(filename, (str, bytes, os.PathLike)):
            # This is a path to a file, so we open the file
            self._file = io.open(filename, 'rb')
            self._close_file = True
        elif hasattr(filename, "read") and hasattr(filename, "seek"):
            if not filename.readable():
                raise ValueError("Underlying file object must be readable.")
            if not filename.seekable():
                raise ValueError("Underlying file object must be seekable.")
            self._file = filename
        else:
            raise TypeError("filename must be a str, bytes, PathLike or file object")

        self._warn_truncation = warn_truncation

        self._init_header()

        self._next_code = self._starting_code
        self._bit_buffer: int = 0
        self._bits_in_buffer: int = 0
        self._prev_entry: None | bytes = None
        self._code_width: int = _INITIAL_CODE_WIDTH
        self._current_mask: int = _INITIAL_MASK

        self._decomp_pos: int = 0

        # We keep checkpoints where the dictionary is reset to 0, to improve seeking performance
        # At byte 3 we have uncompressed 0 bytes
        self._save_checkpoints = save_checkpoints
        self._checkpoints_compressed: list[int] = [3]
        self._checkpoints_uncompressed: list[int] = [0]

        self._extra_buffer: bytearray = bytearray()

    def _init_header(self) -> None:
        """
        Initialize and validate the header of the compressed file.

        Reads the first three bytes of the file to validate magic bytes and set up configuration flags.

        Raises
        ------
        ValueError
            If the file is too short or if the magic bytes do not match expectations, or if the max code width is invalid.
        Warning
            Issues a warning if unknown header flags are encountered.
        """
        header = self._file.read(3)
        if len(header) < 3:
            raise ValueError("File too short, missing header.")
        if header[0] != _MAGIC_BYTE0 or header[1] != _MAGIC_BYTE1:
            raise ValueError(f"Invalid file header: Magic bytes do not match (expected {_MAGIC_BYTE0:02x} "
                             f"{_MAGIC_BYTE1:02x}, got {header[0]:02x} {header[1]:02x}).")

        flag_byte = header[2]

        self._max_width: int = flag_byte & _CODE_WIDTH_FLAG
        if self._max_width < _INITIAL_CODE_WIDTH:
            raise ValueError(f"Invalid file header: Max code width less than the minimum of {_INITIAL_CODE_WIDTH}.")

        if flag_byte & _UNKNOWN_FLAGS:
            warnings.warn("File header contains unknown flags, decompression may be incorrect.", RuntimeWarning)

        self._block_mode: bool = bool(flag_byte & _BLOCK_MODE_FLAG)

        self._dictionary: list[bytes] = [i.to_bytes() for i in range(256)]
        if self._block_mode:
            # In block mode, code 256 is reserved for CLEAR. Append empty bytes object just as a placeholder.
            self._dictionary.append(b'')
        self._starting_code: int = len(self._dictionary)

    def readable(self) -> bool:
        """
        Check if the file is readable.

        Returns
        -------
        bool
            Always True.
        """
        return True

    def read(self, size: int = -1) -> bytes:
        """
        Read up to ``size`` bytes of decompressed data.

        Parameters
        ----------
        size : int, optional
            The maximum number of decompressed bytes to read. If negative, read until EOF.
            Default is -1 (read until EOF).

        Returns
        -------
        bytes
            The decompressed data read from the LZW-compressed file.

        Raises
        ------
        ValueError
            If the file is closed.
        """
        if self.closed:
            raise ValueError("I/O operation on closed file.")
        return self._decode_bytes(size)

    def readinto(self, b) -> int:
        """
        Read decompressed bytes into a pre-allocated, writable bytes-like object.

        Parameters
        ----------
        b : writable buffer
            The buffer into which decompressed data is read.

        Returns
        -------
        int
            The number of bytes read into the buffer.
        """
        with memoryview(b) as view, view.cast("B") as byte_view:
            data = self.read(len(byte_view))
            byte_view[:len(data)] = data
        return len(data)

    def _decode_bytes(self, size: int = -1, get_bytes: bool = True) -> bytes | None:
        """
        Decode and return a specified number of decompressed bytes.

        Parameters
        ----------
        size : int, optional
            Number of bytes to decompress. If negative, decompress until EOF.
            Default is -1.
        get_bytes : bool, optional
            If True, return the decompressed bytes; if False, update internal buffers without returning data.
            Default is True.

        Returns
        -------
        bytes or None
            The decompressed data as bytes if get_bytes is True, otherwise None.
        """
        read_all = False
        if size < 0:
            read_all = True

        if not read_all and len(self._extra_buffer) >= size:
            # Early quit if we already have enough bytes decompressed, just serve those.
            aux = self._extra_buffer[:size]
            del self._extra_buffer[:size]
            self._decomp_pos += size
            if get_bytes:
                return bytes(aux)
            else:
                return None
        else:
            # Otherwise use the entire extra buffer as our decomp_buffer
            decomp_buffer = self._extra_buffer

        # Here we use local variables to cache the accesses to self
        # While this may seem like an odd thing to do, these variables are accessed very frequently inside the loop
        # Using local variables in this case results in a real speed up of around 2x
        bit_buffer = self._bit_buffer
        bits_in_buffer = self._bits_in_buffer
        code_width = self._code_width
        current_mask = self._current_mask
        next_code = self._next_code
        prev_entry = self._prev_entry

        dictionary = self._dictionary
        file = self._file
        max_width = self._max_width
        block_mode = self._block_mode
        starting_code = self._starting_code

        save_checkpoints = self._save_checkpoints
        checkpoints_compressed = self._checkpoints_compressed
        checkpoints_uncompressed = self._checkpoints_uncompressed
        decomp_pos = self._decomp_pos

        # Continue decompressing until we've reached the requested size or EOF.
        while read_all or len(decomp_buffer) < size:
            """
            For any given code_width, we need to read total_codes = 2 ** (code_width - 1)
            So we have total_bits = code_width * total_codes
            But we need to do total_bytes = total_bits // 8, which is the same as total_bits // 2 ** 3
            So we have total_bytes = code_width * 2 ** (code_width - 1) // 2 ** 3
            Or total_bytes = code_width * 2 ** (code_width - 4)          
            """
            cur_chunk = file.read(code_width * 2 ** (code_width - 4))

            if not cur_chunk:
                # This is EOF. There's nothing left to read, so we just quit.
                # We can clear out the dictionary to release memory.
                del dictionary[starting_code:]
                if self._warn_truncation and bits_in_buffer >= 8:
                    # This means we read at least one byte but then the code didn't finish (i.e. a partial code)
                    warnings.warn("Bitstream ended in a partial code, file may be truncated.", RuntimeWarning)
                break

            for i, cur_byte in enumerate(cur_chunk):
                bit_buffer += cur_byte << bits_in_buffer
                bits_in_buffer += 8

                if bits_in_buffer < code_width:
                    continue

                code = bit_buffer & current_mask
                bit_buffer >>= code_width
                bits_in_buffer -= code_width

                if block_mode and code == _CLEAR_CODE:
                    """
                    We have encountered a CLEAR, but we have already read further into this file, we need to rewind.
                    The bitstream is divided into blocks of codes that have the same code_width.
                    Each block is exactly code_width bytes wide (i.e. at code_width=9 each block has 9 bytes).
                    CLEAR code may be in the middle of a block, requiring realignment to the next code boundary.
                    We know how many bytes have been decoded since we started using the current code_width (i).
                    Then the modulo tells us how many bytes we have advanced into the current block.
                    If the modulo is 0, we're already at a boundary, nothing needs to be done.
                    But if we aren't, we need to advance to the end of the block.
                    That is one full block minus however many bytes we have already advanced into the current block.

                    E.g. if we have i=13, code_width=9:
                    13 % 9 = 4
                    13 + 9 - 4 = 18 -> new position 

                     0....2....4....6....8 | 9...11...13...15...17 | 18...20...22...
                    [  Block 0 (9 bytes)  ] [  Block 1 (9 bytes)  ] [  Block 2 
                                                      ^              ^
                                                      |              |
                                                   old pos        new pos
                    Given that our relative file position will be at len(cur_chunk), we need to go back that amount
                    minus the new position we've calculated. 
                    """

                    if advanced := i % code_width:
                        i += code_width - advanced

                    # We're rewinding relative to the current file position
                    file_pos = file.seek(i - len(cur_chunk), os.SEEK_CUR)

                    # If this is a new checkpoint save it
                    if save_checkpoints and bisect(checkpoints_compressed, file_pos) == len(checkpoints_compressed):
                        checkpoints_compressed.append(file_pos)
                        checkpoints_uncompressed.append(decomp_pos + len(decomp_buffer))

                    # Clear the dictionary except the starting codes
                    del dictionary[starting_code:]
                    next_code = starting_code

                    # Revert to initial code_width (will be incremented right after we break the loop)
                    code_width = _INITIAL_CODE_WIDTH - 1
                    prev_entry = None
                    break

                try:
                    entry = dictionary[code]
                except IndexError:
                    if code == next_code:
                        if prev_entry is None:
                            raise ValueError(
                                f"Invalid code {code} encountered in bitstream. Expected a literal character.")
                        # Special case: code not yet in the dictionary.
                        entry = prev_entry + prev_entry[:1]
                    else:
                        raise ValueError(f"Invalid code {code} encountered in bitstream.")

                decomp_buffer.extend(entry)

                if prev_entry is not None and next_code <= current_mask:
                    dictionary.append(prev_entry + entry[:1])
                    next_code += 1

                prev_entry = entry

            # Only increase code width if we won't surpass max.
            # Some files will stay at max_width even after the entire dictionary is filled
            if code_width < max_width:
                code_width += 1
                current_mask = 2 ** code_width - 1
                bit_buffer = 0
                bits_in_buffer = 0

        # The local variables may have been updated in the loop, so we need to update self
        self._bit_buffer = bit_buffer
        self._bits_in_buffer = bits_in_buffer
        self._code_width = code_width
        self._current_mask = current_mask
        self._next_code = next_code
        self._prev_entry = prev_entry

        # If more data was decompressed than requested, save the extra for later.
        if read_all:
            # Create a new extra buffer that is empty
            self._extra_buffer = bytearray()
        else:
            # Create a new extra buffer with the remaining data
            self._extra_buffer = decomp_buffer[size:]
            del decomp_buffer[size:]
        self._decomp_pos += len(decomp_buffer)
        if get_bytes:
            return bytes(decomp_buffer)
        else:
            return None

    def seekable(self) -> bool:
        """
        Check if the file supports random access.

        Returns
        -------
        bool
            Always True.
        """
        return True

    def tell(self) -> int:
        """
        Return the current position in the decompressed stream.

        Returns
        -------
        int
            The current decompressed byte position.
        """
        return self._decomp_pos

    def seek(self, offset: int, whence: int = 0):
        """
        Change the stream position to the given byte offset in the decompressed stream.

        Parameters
        ----------
        offset : int
            The byte offset to seek to.
        whence : int, optional
            The reference point for offset. It can be io.SEEK_SET (0, default), io.SEEK_CUR (1), or io.SEEK_END (2).
            io.SEEK_END is not supported.

        Returns
        -------
        int
            The new absolute position in the decompressed stream.

        Raises
        ------
        ValueError
            If offset is not an integer, if seeking to a negative stream position, or if the file is closed.
        io.UnsupportedOperation
            If seeking from the end is attempted.
        """
        if not isinstance(offset, int):
            raise ValueError(f"offset must be an integer.")
        if self.closed:
            raise ValueError("I/O operation on closed file.")
        if whence == io.SEEK_SET:
            new_pos = offset
            diff = offset - self._decomp_pos
        elif whence == io.SEEK_CUR:
            new_pos = self._decomp_pos + offset
            diff = offset
        elif whence == io.SEEK_END:
            raise io.UnsupportedOperation("Cannot seek from end in an LZW compressed file.")
        else:
            raise ValueError(f"Invalid whence: {whence}")
        if new_pos < 0:
            raise ValueError(f"Can't seek to a negative stream position.")

        if diff != 0:
            index = bisect(self._checkpoints_uncompressed, new_pos) - 1
            checkpoint = self._checkpoints_uncompressed[index]

            # We have to go to a checkpoint whenever we seek backwards
            # We also want to go to a checkpoint if that allows us to skip some decompression blocks
            if diff < 0 or checkpoint > self._decomp_pos:
                self._file.seek(self._checkpoints_compressed[index])
                del self._dictionary[self._starting_code:]
                self._next_code = self._starting_code
                self._bit_buffer = 0
                self._bits_in_buffer = 0
                self._prev_entry = None
                self._code_width = _INITIAL_CODE_WIDTH
                self._current_mask = _INITIAL_MASK
                self._decomp_pos = checkpoint
                self._extra_buffer = bytearray()

            self._decode_bytes(new_pos - self._decomp_pos, get_bytes=False)
        return new_pos

    def writable(self) -> bool:
        """
        Check if the file is writable.

        Returns
        -------
        bool
            Always False since uncompresspy does not support writing.
        """
        return False

    def write(self, data) -> int:
        """
        Writing is not supported.

        Parameters
        ----------
        data : Any
            Data that would be written (unsupported).

        Raises
        ------
        io.UnsupportedOperation
            Always raised because writing is not permitted.
        """
        raise io.UnsupportedOperation('write')

    def fileno(self) -> int:
        """
        Return the file descriptor for the underlying file.

        Returns
        -------
        int
            The file descriptor associated with the underlying file.

        Raises
        ------
        ValueError
            If an I/O operation is attempted on a closed file.
        """
        if self.closed:
            raise ValueError("I/O operation on closed file")
        return self._file.fileno()

    def close(self):
        """
        Close the LZWFile and the underlying file if necessary.
        """
        if self._close_file and self._file is not None:
            self._file.close()
            self._file = None
        self._extra_buffer = None
        self._dictionary = None
        super().close()


# Convenience function for opening LZW-files.
def open(filename: str | bytes | os.PathLike | BinaryIO, mode: str = 'rb', warn_truncation: bool = True,
         save_checkpoints: bool = True, encoding: str = None, errors: str = None,
         newline: str = None) -> LZWFile | io.TextIOWrapper:
    """
    Open an LZW-compressed file in binary or text mode and return a file-like object that decompresses data on the fly.

    Parameters
    ----------
    filename : str, bytes, os.PathLike, or BinaryIO
        If filename is a str, bytes, or PathLike object, it represents the file path to be opened and read from.
        Otherwise, it must be a file-like object (supporting read and seek operations).
    mode : str, optional
        The mode in which to open the file. Supported modes are 'r' or 'rb' (default) for binary mode and 'rt' for text mode.
    warn_truncation : bool, optional
        If True, warns about potential truncation of the file. Defaults to True.
    save_checkpoints : bool, optional
        If True, save checkpoint positions to improve seeking performance at a small memory hit. Defaults to True.
    encoding : str, optional
        Text encoding to use in text mode. Must not be provided for binary mode.
    errors : str, optional
        Error handling scheme to use in text mode. Must not be provided for binary mode.
    newline : str, optional
        Specifies how universal newlines mode works in text mode. Must not be provided for binary mode.

    Returns
    -------
    LZWFile or io.TextIOWrapper
        A file-like object that yields decompressed data. Wrapped in an io.TextIOWrapper if text mode is used.

    Raises
    ------
    ValueError
        If an invalid mode is specified or if text mode is incorrectly combined with binary arguments.

    Examples
    --------
    >>> with uncompresspy.open('example.txt.Z', 'rt') as f:
    ...     line = f.readline()
    """
    if "t" in mode:
        if "b" in mode:
            raise ValueError(f"Invalid mode: {mode!r}")
    else:
        if encoding is not None:
            raise ValueError("Argument 'encoding' not supported in binary mode")
        if errors is not None:
            raise ValueError("Argument 'errors' not supported in binary mode")
        if newline is not None:
            raise ValueError("Argument 'newline' not supported in binary mode")

    binary_file = LZWFile(filename, mode.replace("t", ""), warn_truncation, save_checkpoints)

    if "t" in mode:
        encoding = io.text_encoding(encoding)
        return io.TextIOWrapper(binary_file, encoding, errors, newline)
    else:
        return binary_file


# Convenience function for extracting
def extract(input_filename: str | bytes | os.PathLike | BinaryIO, output_filename: str | bytes | os.PathLike,
            overwrite: bool = False, chunk_size: int = io.DEFAULT_BUFFER_SIZE) -> None:
    """
    Extract an LZW-compressed input file into an uncompressed output file.

    Parameters
    ----------
    input_filename : str, bytes, os.PathLike, or BinaryIO
        The filename or file-like object of the LZW-compressed input file.
    output_filename : str, bytes, or os.PathLike
        The filename for the uncompressed output file.
    overwrite : bool, optional
        If False and the output file already exists, raise a FileExistsError.
        Defaults to False.
    chunk_size : int, optional
        The size of chunks (in bytes) to use when reading from the input file. Defaults to io.DEFAULT_BUFFER_SIZE.
        Bigger chunks may provide a performance benefit at the cost of memory usage during extraction.

    Returns
    -------
    None

    Raises
    ------
    FileExistsError
        If overwrite is False and the output file already exists.

    Examples
    --------
    >>> uncompresspy.extract('example.txt.Z', 'example.txt')
    """
    if not overwrite:
        if os.path.exists(output_filename):
            raise FileExistsError(f'File {output_filename!r} already exists. If you mean to replace it, use the '
                                  f'argument "overwrite=True".')
    with LZWFile(input_filename, 'rb', save_checkpoints=False) as input_file:
        with io.open(output_filename, 'wb') as output_file:
            while chunk := input_file.read(chunk_size):
                output_file.write(chunk)
