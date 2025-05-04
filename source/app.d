import std.complex;
import std.numeric;
import std.stdio;
import std.typecons;
import std.exception;
import mir.ndslice;

//
// mir + normal kernel
//
extern (C) void incr(double* ptr, int sz);

alias cudaError = int;

extern (C) @nogc nothrow
{
  cudaError cudaMalloc(void** ptr, ulong size);
  cudaError cudaFree(void* ptr);
  cudaError cudaMallocHost(void** ptr, ulong size);
  cudaError cudaFreeHost(void* ptr);
  cudaError cudaDeviceSynchronize();
}

void exercise_mir_kern()
{
  double* ptr;
  auto res = cudaMallocHost(cast(void**)&ptr, double.sizeof * 1024);
  enforce(res == 0, "Failed to allocate.");

  // use ndslice to fill array
  auto slice = ptr.sliced(1024);
  slice[] = 3.14;

  // run kernel
  incr(ptr, 1024);

  res = cudaDeviceSynchronize();
  enforce(res == 0, "Failed to sync.");
  writeln(slice);

  res = cudaFreeHost(ptr);
  enforce(res == 0, "Failed to deallocate.");
}

//
//  CUFFT
//
alias cufftHandle = uint;
alias cufftResult = int;

enum CufftType
{
  R2C = 0x2a, // Real to Complex (single precision)
  C2R = 0x2c, // Complex to Real (single precision)
  C2C = 0x29, // Complex to Complex (single precision)
  D2Z = 0x6a, // Real to Complex (double precision)
  Z2D = 0x6c, // Complex to Real (double precision)
  Z2Z = 0x69 // Complex to Complex (double precision)
}

struct CuComplex
{
  float x;
  float y;
}

enum CufftDir
{
  FWD = -1,
  REV = 1
}

extern (C) @nogc nothrow
{
  cufftResult cufftPlan1d(cufftHandle* plan, int nx, CufftType type, int batch);
  cufftResult cufftDestroy(cufftHandle plan);
  cufftResult cufftExecC2C(cufftHandle plan, CuComplex* idata, CuComplex* odata, CufftDir dir);
}

void exercise_cufft()
{
  int count = 64;
  cufftHandle handle;
  CuComplex* data;

  auto res = cudaMallocHost(cast(void**)&data, CuComplex.sizeof * count);
  enforce(res == 0, "Failed to allocate.");

  // fill test data and do a cpu based fft for comparison
  foreach (i; 0 .. count)
  {
    data[i].x = i / cast(float) count;
    data[i].y = i / cast(float) count;
  }
  auto cpu = fft((cast(Complex!float*) data).sliced(count));
  writeln(cpu);

  // create an execute the cufft
  auto hres = cufftPlan1d(&handle, count, CufftType.C2C, 1);
  enforce(hres == 0, "Failed to plan.");

  hres = cufftExecC2C(handle, data, data, CufftDir.FWD);
  enforce(hres == 0, "Failed to C2C.");

  // print the cufft for comparison
  res = cudaDeviceSynchronize();
  enforce(res == 0, "Failed to sync.");
  writeln((cast(Complex!float*) data).sliced(count));

  // cleanup
  hres = cufftDestroy(handle);
  enforce(hres == 0, "Failed to destroy.");

  res = cudaFreeHost(data);
  enforce(res == 0, "Failed to deallocate.");
}

///  main
void main()
{
  exercise_mir_kern();
  exercise_cufft();
}
