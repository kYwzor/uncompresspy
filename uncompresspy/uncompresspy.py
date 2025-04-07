import io
import os
import warnings
from typing import BinaryIO

INITIAL_CODE_WIDTH = 9
INITIAL_MASK = 2 ** INITIAL_CODE_WIDTH - 1
CLEAR_CODE = 256
MAGIC_BYTE0 = 0x1F
MAGIC_BYTE1 = 0x9D
BLOCK_MODE_FLAG = 0x80
CODE_WIDTH_FLAG = 0x1F
UNKNOWN_FLAGS = 0x60


class LZWFile(io.RawIOBase):
    """
    A file-like object that transparently decompresses an LZW-compressed file on the fly.
    It does not load the entire compressed file into memory, and supports incremental reads via the read() method.
    It only supports a binary file interface (data read is returned as bytes). Context management is supported.
    """

    def __init__(self, filename: str | bytes | os.PathLike | BinaryIO, mode = 'rb', keep_buffer: bool = False):
        """Open an LZW-compressed file in binary mode.

        If filename is a str, bytes, or PathLike object, it gives the name of the file to be opened. Otherwise, it
        must be a file object, which will be used to read the compressed data.

        Only reading modes are supported ('r' and 'rb').
        """
        self._file = None
        self._close_file = False
        if mode in ('', 'r', 'rb'):
            # We always operate in binary mode
            mode = 'rb'
        else:
            raise ValueError(f"Invalid mode: {mode!r} (only reading is supported)")

        if isinstance(filename, (str, bytes, os.PathLike)):
            # This is a path to a file, so we open the file
            self._file = io.open(filename, mode)
            self._close_file = True
        elif hasattr(filename, "read") and hasattr(filename, "seek"):
            if not filename.readable():
                raise ValueError("Underlying file object must be readable.")
            if not filename.seekable():
                raise ValueError("Underlying file object must be seekable.")
            self._file = filename
        else:
            raise TypeError("filename must be a str, bytes, PathLike or file object")

        self._init_header()

        self._next_code = self._starting_code
        self._bit_buffer = 0
        self._bits_in_buffer = 0
        self._prev_entry = None
        self._code_width = INITIAL_CODE_WIDTH
        self._current_mask = INITIAL_MASK

        self._decomp_pos = 0

        self._extra_buffer = bytearray()
        self._keep_buffer = keep_buffer
        if self._keep_buffer:
            self._total_buffer = bytearray()

    def _init_header(self) -> None:
        header = self._file.read(3)
        if len(header) < 3:
            raise ValueError("File too short, missing header.")
        if header[0] != MAGIC_BYTE0 or header[1] != MAGIC_BYTE1:
            raise ValueError(f"Invalid file header: Magic bytes do not match (expected {MAGIC_BYTE0:02x} "
                             f"{MAGIC_BYTE1:02x}, got {header[0]:02x} {header[1]:02x}).")

        flag_byte = header[2]

        self._max_width = flag_byte & CODE_WIDTH_FLAG
        if self._max_width < INITIAL_CODE_WIDTH:
            raise ValueError(f"Invalid file header: Max code width less than the minimum of {INITIAL_CODE_WIDTH}.")

        if flag_byte & UNKNOWN_FLAGS:
            warnings.warn("File header contains unknown flags, decompression may be incorrect.")

        self._block_mode = bool(flag_byte & BLOCK_MODE_FLAG)

        self._dictionary = [i.to_bytes() for i in range(256)]
        if self._block_mode:
            # In block mode, code 256 is reserved for CLEAR.
            self._dictionary.append(b'')
        self._starting_code = len(self._dictionary)

    def readable(self):
        return True

    def read(self, size=-1):
        """
        Read up to 'size' bytes of decompressed data.
        If size is negative, read until the end of the compressed stream.
        """
        if self.closed:
            raise ValueError("I/O operation on closed file.")
        return self._decode_bytes(size)

    def readinto(self, b):
        with memoryview(b) as view, view.cast("B") as byte_view:
            data = self.read(len(byte_view))
            byte_view[:len(data)] = data
        return len(data)

    def _decode_bytes(self, size=-1, get_bytes=True) -> bytes | None:
        read_all = False
        if size < 0:
            read_all = True

        if not read_all and len(self._extra_buffer) >= size:
            # Early quit if we already have enough bytes decompressed, just serve those.
            aux = self._extra_buffer[:size]
            del self._extra_buffer[:size]
            self._decomp_pos += size
            if self._keep_buffer:
                self._total_buffer += aux
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
                break

            for i, cur_byte in enumerate(cur_chunk):
                bit_buffer += cur_byte << bits_in_buffer
                bits_in_buffer += 8

                if bits_in_buffer < code_width:
                    continue

                code = bit_buffer & current_mask
                bit_buffer >>= code_width
                bits_in_buffer -= code_width

                if block_mode and code == CLEAR_CODE:
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
                    file.seek(i - len(cur_chunk), os.SEEK_CUR)

                    # Clear the dictionary except the starting codes
                    del dictionary[starting_code:]
                    next_code = starting_code

                    # Revert to initial code_width (will be incremented right after we break the loop)
                    code_width = INITIAL_CODE_WIDTH - 1
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
        if self._keep_buffer:
            self._total_buffer += decomp_buffer
        if get_bytes:
            return bytes(decomp_buffer)
        else:
            return None

    def seekable(self):
        return True

    def tell(self):
        return self._decomp_pos

    def seek(self, offset, whence=0):
        if self.closed:
            raise ValueError("I/O operation on closed file.")
        if whence == io.SEEK_SET:
            new_pos = offset
            diff = offset - self._decomp_pos
        elif whence == io.SEEK_CUR:
            new_pos = self._decomp_pos + offset
            if new_pos < 0:
                raise ValueError(f"Can't seek to a negative position.")
            diff = offset
        elif whence == io.SEEK_END:
            raise io.UnsupportedOperation("Cannot seek from end in an LZW compressed file.")
        else:
            raise ValueError(f"Invalid whence: {whence}")
        if diff > 0:
            # We have to advance, decode bytes but don't request them
            self._decode_bytes(diff, get_bytes=False)
        elif diff < 0:
            if self._keep_buffer:
                self._extra_buffer = self._total_buffer[new_pos:] + self._extra_buffer
                del self._total_buffer[new_pos:]
                self._decomp_pos = new_pos
            else:
                warnings.warn(f"Seeking backwards is extremely inefficient without the 'keep_buffer' option, as it "
                              f"requires restarting the decompression from the beginning of the file. Consider using"
                              f"the 'keep_buffer' option if seeking backwards is a common operation for your use-case.")
                self._file.seek(0)

                self._init_header()

                self._next_code = self._starting_code
                self._bit_buffer = 0
                self._bits_in_buffer = 0
                self._prev_entry = None
                self._code_width = INITIAL_CODE_WIDTH
                self._current_mask = INITIAL_MASK

                self._decomp_pos = 0

                self._extra_buffer = bytearray()

                self.read(new_pos)
        return new_pos

    def writable(self):
        return False

    def write(self, data):
        raise io.UnsupportedOperation('write')

    def fileno(self):
        """Return the file descriptor for the underlying file."""
        if self.closed:
            raise ValueError("I/O operation on closed file")
        return self._file.fileno()

    def close(self):
        # Mimic file buffer behavior
        if self._close_file and self._file is not None:
            self._file.close()
            self._file = None
        self._extra_buffer = None
        self._total_buffer = None
        self._dictionary = None
        super().close()


# Convenience function for opening LZW-files.
def open(filename: str | bytes | os.PathLike | BinaryIO, mode: str = 'rb', encoding=None, errors=None, newline=None,
         **kwargs) -> LZWFile | io.TextIOWrapper:
    """
    Open an LZW-compressed file in binary or text mode and return a file-like object that decompresses data on the fly.
    filename can be either an actual file name (given as a str, bytes, or PathLike object), in which case the named file
    is opened, or it can be an existing file object to read from.

    The mode argument can be "r", "rb" (default) for binary mode or "rt" for text mode.

    For text mode, an LZWFile object is created, and wrapped in an io.TextIOWrapper instance with the specified
    encoding, error handling behavior, and line ending(s).

    Usage:
      with uncompresspy.open('example.txt.Z', 'rt') as f:
          data = f.readline()
          # Will read one line of decompressed bytes from 'example.txt.Z'
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

    binary_file = LZWFile(filename, mode.replace("t", ""), **kwargs)

    if "t" in mode:
        encoding = io.text_encoding(encoding)
        return io.TextIOWrapper(binary_file, encoding, errors, newline)
    else:
        return binary_file


# Convenience function for extracting
def extract(input_filename: str | bytes | os.PathLike | BinaryIO, output_filename: str | bytes | os.PathLike,
            overwrite=False, chunk_size=io.DEFAULT_BUFFER_SIZE) -> None:
    """
    Extract an LZW-compressed input file into an uncompressed output file.

    Usage:
      uncompresspy.extract('example.txt.Z', 'example.txt')
      # Will write the uncompressed data from 'example.txt.Z' to 'example.txt'.
    """
    if not overwrite:
        if os.path.exists(output_filename):
            raise FileExistsError(f'File {output_filename!r} already exists. If you mean to replace it, use the '
                                  f'argument "overwrite=True".')
    with LZWFile(input_filename, 'rb') as input_file:
        with io.open(output_filename, 'wb') as output_file:
            while chunk := input_file.read(chunk_size):
                output_file.write(chunk)
