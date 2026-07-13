import dace

sdfg = dace.SDFG.from_file('axpy_stream.sdfg')

# Inspect all arrays and streams
for name, arr in sdfg.arrays.items():
    arr_type = type(arr).__name__
    print(f"{name:20s}  type={arr_type:15s}  "
          f"transient={arr.transient}")
    if isinstance(arr, dace.data.Stream):
        print(f"  buffer_size={arr.buffer_size}  "
              f"dtype={arr.dtype}")