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
    A file-like object that transparently decompresses an LZW-compressed .Z file on the fly.
    It does not load the entire compressed file into memory, and supports incremental reads via the read() method.
    This class supports context management so you can use it with a with-statement.
    """

    def __init__(self, buffer_obj: BinaryIO, keep_buffer: bool = False, chunk_size: int = io.DEFAULT_BUFFER_SIZE):
        if not buffer_obj.readable():
            raise ValueError("Underlying buffer object must be readable.")
        self.file = buffer_obj
        self.chunk_size = chunk_size
        self._init_header()

        self.next_code = self.starting_code
        self.bit_buffer = 0
        self.bits_in_buffer = 0
        self.prev_entry = None
        self.code_width = INITIAL_CODE_WIDTH
        self.current_mask = INITIAL_MASK

        self.decomp_pos = 0

        self.extra_buffer = bytearray()
        self.keep_buffer = keep_buffer
        if self.keep_buffer:
            self.total_buffer = bytearray()

    def _init_header(self):
        header = self.file.read(3)
        if len(header) < 3:
            raise ValueError("File too short, missing header.")
        if header[0] != MAGIC_BYTE0 or header[1] != MAGIC_BYTE1:
            raise ValueError(f"Invalid file header: Magic bytes do not match (expected {MAGIC_BYTE0:02x} "
                             f"{MAGIC_BYTE1:02x}, got {header[0]:02x} {header[1]:02x}).")

        flag_byte = header[2]

        self.max_width = flag_byte & CODE_WIDTH_FLAG
        if self.max_width < INITIAL_CODE_WIDTH:
            raise ValueError(f"Invalid file header: Max code width less than the minimum of {INITIAL_CODE_WIDTH}.")

        if flag_byte & UNKNOWN_FLAGS:
            warnings.warn("File header contains unknown flags, decompression may be incorrect.")

        self.block_mode = bool(flag_byte & BLOCK_MODE_FLAG)

        self.dictionary = [i.to_bytes() for i in range(256)]
        if self.block_mode:
            # In block mode, code 256 is reserved for CLEAR.
            self.dictionary.append(b'')
        self.starting_code = len(self.dictionary)

    def readable(self):
        return True

    def read(self, size=-1):
        """
        Read up to 'size' bytes of decompressed data.
        If size is negative, read until the end of the compressed stream.
        """
        read_all = False
        if size < 0:
            read_all = True

        if not read_all and len(self.extra_buffer) >= size:
            # Early quit if we already have enough bytes decompressed, just serve those.
            aux = self.extra_buffer[:size]
            del self.extra_buffer[:size]
            self.decomp_pos += size
            if self.keep_buffer:
                self.total_buffer += aux
            return bytes(aux)
        else:
            # Otherwise use the entire extra buffer as our decomp_buffer
            decomp_buffer = self.extra_buffer

        # Here we use local variables to cache the accesses to self
        # While this may seem like an odd thing to do, these variables are accessed very frequently inside the loop
        # Using local variables in this case results in a real speed up of around 2x
        bit_buffer = self.bit_buffer
        bits_in_buffer = self.bits_in_buffer
        code_width = self.code_width
        current_mask = self.current_mask
        next_code = self.next_code
        prev_entry = self.prev_entry

        dictionary = self.dictionary
        file = self.file
        max_width = self.max_width
        block_mode = self.block_mode
        starting_code = self.starting_code

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
                # If there's nothing left to read, just quit
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
        self.bit_buffer = bit_buffer
        self.bits_in_buffer = bits_in_buffer
        self.code_width = code_width
        self.current_mask = current_mask
        self.next_code = next_code
        self.prev_entry = prev_entry

        # If more data was decompressed than requested, save the extra for later.
        if read_all:
            # Create a new extra buffer that is empty
            self.extra_buffer = bytearray()
        else:
            # Create a new extra buffer with the remaining data
            self.extra_buffer = decomp_buffer[size:]
            del decomp_buffer[size:]
        self.decomp_pos += len(decomp_buffer)
        if self.keep_buffer:
            self.total_buffer += decomp_buffer
        return bytes(decomp_buffer)

    def seekable(self):
        return True

    def tell(self):
        return self.decomp_pos

    def seek(self, offset, whence=0):
        if whence == io.SEEK_SET:
            new_pos = offset
            diff = offset - self.decomp_pos
        elif whence == io.SEEK_CUR:
            new_pos = self.decomp_pos + offset
            if new_pos < 0:
                raise ValueError(f"Can't seek to a negative position.")
            diff = offset
        elif whence == io.SEEK_END:
            raise io.UnsupportedOperation("Cannot seek from end in a LZW compressed file.")
        else:
            raise ValueError(f"Invalid whence: {whence}")
        if diff > 0:
            # We have to advance, just read and ignore the output
            self.read(diff)
        elif diff < 0:
            if self.keep_buffer:
                self.extra_buffer = self.total_buffer[new_pos:] + self.extra_buffer
                del self.total_buffer[new_pos:]
                self.decomp_pos = new_pos
            else:
                warnings.warn(f"Seeking backwards is extremely inefficient without the 'keep_buffer' option, as it "
                              f"requires restarting the decompression from the beginning of the file. Consider using"
                              f"the 'keep_buffer' option if seeking backwards is a common operation for your use-case.")
                self.file.seek(0)

                self._init_header()

                self.next_code = self.starting_code
                self.bit_buffer = 0
                self.bits_in_buffer = 0
                self.prev_entry = None
                self.code_width = INITIAL_CODE_WIDTH
                self.current_mask = INITIAL_MASK

                self.decomp_pos = 0

                self.extra_buffer = bytearray()

                self.read(new_pos)
        return new_pos

    def close(self):
        # Mimic file buffer behavior
        if not self.file.closed:
            self.file.close()
        super().close()

    def __enter__(self):
        # Allow usage of context managers
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Allow usage of context managers
        self.close()

    def __iter__(self):
        # Allow iteration of the object
        return self

    def __next__(self):
        # Allow iteration of the object
        chunk = self.read(self.chunk_size)
        if chunk == b"":
            # EOF
            raise StopIteration
        return chunk


# Convenience function for opening .Z files.
def open(filepath: str | os.PathLike, mode: str = 'rb', **kwargs):
    """
    Open a .Z (LZW compressed) file in binary mode and return a file-like object that decompresses data on the fly.

    Usage:
      with uncompresspy.open('example.txt.Z') as f:
          data = f.read(100)
          # Will read 100 decompressed bytes from 'example.txt.Z'
    """
    if mode != 'rb':
        raise ValueError("Only binary mode ('rb') is supported.")
    file_obj = io.open(filepath, 'rb')
    return LZWFile(file_obj, **kwargs)


# Convenience function for extracting
def extract(input_filepath: str | os.PathLike, output_filepath: str | os.PathLike, **kwargs):
    """
    Extract a .Z (LZW compressed) input file into an uncompressed output file.

    Usage:
      uncompresspy.extract('example.txt.Z', 'example.txt')
      # Will write the uncompressed data from 'example.txt.Z' to 'example.txt'.
    """
    with open(input_filepath, 'rb', **kwargs) as input_file:
        with io.open(output_filepath, 'wb') as output_file:
            for chunk in input_file:
                output_file.write(chunk)


def from_buffer(input_buffer: BinaryIO, **kwargs):
    """
    Return a file-like object that decompresses data in the input buffer on the fly.

    Usage:
      with open('example.txt.Z', 'rb') as f1:
          with uncompresspy.from_buffer(f1) as f2:
              data = f2.read(100)
              # Will read 100 decompressed bytes from 'example.txt.Z'
    """
    return LZWFile(input_buffer, **kwargs)
