// RUN: byteir-opt %s -nvvm-codegen | FileCheck %s

// CHECK-LABEL: gpu.module @unified

module attributes {byre.container_module, gpu.container_module} {
  gpu.module @unified {
    gpu.func @Unknown92(%arg0: memref<512xf32>, %arg1: memref<512xf32>, %arg2: memref<512xf32>) kernel {
      %cst = arith.constant 1.000000e-01 : f32
      %cst_0 = arith.constant 0.899999976 : f32
      %c512 = arith.constant 512 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = gpu.grid_dim  x
      %6 = arith.muli %1, %5 : index
      scf.for %arg3 = %4 to %c512 step %6 {
        %7 = memref.load %arg1[%arg3] : memref<512xf32>
        %8 = memref.load %arg0[%arg3] : memref<512xf32>
        %9 = arith.mulf %7, %cst_0 : f32
        %10 = arith.mulf %8, %cst : f32
        %11 = arith.addf %10, %9 : f32
        memref.store %11, %arg2[%arg3] : memref<512xf32>
      }
      gpu.return
    }
    gpu.func @Unknown82(%arg0: memref<256xf32>, %arg1: memref<256xf32>, %arg2: memref<256xf32>) kernel {
      %cst = arith.constant 1.000000e-01 : f32
      %cst_0 = arith.constant 0.899999976 : f32
      %c256 = arith.constant 256 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = gpu.grid_dim  x
      %6 = arith.muli %1, %5 : index
      scf.for %arg3 = %4 to %c256 step %6 {
        %7 = memref.load %arg1[%arg3] : memref<256xf32>
        %8 = memref.load %arg0[%arg3] : memref<256xf32>
        %9 = arith.mulf %7, %cst_0 : f32
        %10 = arith.mulf %8, %cst : f32
        %11 = arith.addf %10, %9 : f32
        memref.store %11, %arg2[%arg3] : memref<256xf32>
      }
      gpu.return
    }
    gpu.func @Unknown72(%arg0: memref<128xf32>, %arg1: memref<128xf32>, %arg2: memref<128xf32>) kernel {
      %cst = arith.constant 1.000000e-01 : f32
      %cst_0 = arith.constant 0.899999976 : f32
      %c128 = arith.constant 128 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = gpu.grid_dim  x
      %6 = arith.muli %1, %5 : index
      scf.for %arg3 = %4 to %c128 step %6 {
        %7 = memref.load %arg1[%arg3] : memref<128xf32>
        %8 = memref.load %arg0[%arg3] : memref<128xf32>
        %9 = arith.mulf %7, %cst_0 : f32
        %10 = arith.mulf %8, %cst : f32
        %11 = arith.addf %10, %9 : f32
        memref.store %11, %arg2[%arg3] : memref<128xf32>
      }
      gpu.return
    }
    gpu.func @Unknown62(%arg0: memref<64xf32>, %arg1: memref<64xf32>, %arg2: memref<64xf32>) kernel {
      %cst = arith.constant 1.000000e-01 : f32
      %cst_0 = arith.constant 0.899999976 : f32
      %c64 = arith.constant 64 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = gpu.grid_dim  x
      %6 = arith.muli %1, %5 : index
      scf.for %arg3 = %4 to %c64 step %6 {
        %7 = memref.load %arg1[%arg3] : memref<64xf32>
        %8 = memref.load %arg0[%arg3] : memref<64xf32>
        %9 = arith.mulf %7, %cst_0 : f32
        %10 = arith.mulf %8, %cst : f32
        %11 = arith.addf %10, %9 : f32
        memref.store %11, %arg2[%arg3] : memref<64xf32>
      }
      gpu.return
    }
    gpu.func @Unknown61(%arg0: memref<1000xf32>, %arg1: memref<1x1000xf16>, %arg2: memref<1x1000xf16>) kernel {
      %c0 = arith.constant 0 : index
      %c1000 = arith.constant 1000 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = gpu.grid_dim  x
      %6 = arith.muli %1, %5 : index
      scf.for %arg3 = %4 to %c1000 step %6 {
        %7 = memref.load %arg0[%arg3] : memref<1000xf32>
        %8 = memref.load %arg1[%c0, %arg3] : memref<1x1000xf16>
        %9 = arith.truncf %7 : f32 to f16
        %10 = arith.addf %8, %9 : f16
        memref.store %10, %arg2[%c0, %arg3] : memref<1x1000xf16>
      }
      gpu.return
    }
    gpu.func @Unknown60(%arg0: memref<1000x512xf32>, %arg1: memref<1000x512xf16>) kernel {
      %c512000 = arith.constant 512000 : index
      %c512 = arith.constant 512 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = gpu.grid_dim  x
      %6 = arith.muli %1, %5 : index
      scf.for %arg2 = %4 to %c512000 step %6 {
        %7 = arith.remsi %arg2, %c512 : index
        %8 = arith.divsi %arg2, %c512 : index
        %9 = memref.load %arg0[%8, %7] : memref<1000x512xf32>
        %10 = arith.truncf %9 : f32 to f16
        memref.store %10, %arg1[%8, %7] : memref<1000x512xf16>
      }
      gpu.return
    }
    gpu.func @Unknown59(%arg0: memref<1x512xf16>, %arg1: memref<1x512xf16>) kernel {
      %cst = arith.constant 2.040100e-02 : f16
      %c0 = arith.constant 0 : index
      %c512 = arith.constant 512 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = gpu.grid_dim  x
      %6 = arith.muli %1, %5 : index
      scf.for %arg2 = %4 to %c512 step %6 {
        %7 = memref.load %arg0[%c0, %arg2] : memref<1x512xf16>
        %8 = arith.mulf %7, %cst : f16
        memref.store %8, %arg1[%c0, %arg2] : memref<1x512xf16>
      }
      gpu.return
    }
    gpu.func @Unknown51(%arg0: memref<1x512x7x7xf16>, %arg1: memref<1x512x7x7xf16>, %arg2: memref<1x512x7x7xf16>) kernel {
      %c25088 = arith.constant 25088 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c7 = arith.constant 7 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = gpu.grid_dim  x
      %6 = arith.muli %1, %5 : index
      scf.for %arg3 = %4 to %c25088 step %6 {
        %7 = arith.remsi %arg3, %c7 : index
        %8 = arith.divsi %arg3, %c7 : index
        %9 = arith.remsi %8, %c7 : index
        %10 = arith.divsi %8, %c7 : index
        %11 = memref.load %arg0[%c0, %10, %9, %7] : memref<1x512x7x7xf16>
        %12 = memref.load %arg1[%c0, %10, %9, %7] : memref<1x512x7x7xf16>
        %13 = arith.addf %11, %12 : f16
        %14 = arith.maximumf %13, %cst : f16
        memref.store %14, %arg2[%c0, %10, %9, %7] : memref<1x512x7x7xf16>
      }
      gpu.return
    }
    gpu.func @Unknown49(%arg0: memref<512x512x3x3xf32>, %arg1: memref<512x512x3x3xf16>) kernel {
      %c2359296 = arith.constant 2359296 : index
      %c512 = arith.constant 512 : index
      %c3 = arith.constant 3 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = gpu.grid_dim  x
      %6 = arith.muli %1, %5 : index
      scf.for %arg2 = %4 to %c2359296 step %6 {
        %7 = arith.remsi %arg2, %c3 : index
        %8 = arith.divsi %arg2, %c3 : index
        %9 = arith.remsi %8, %c3 : index
        %10 = arith.divsi %8, %c3 : index
        %11 = arith.remsi %10, %c512 : index
        %12 = arith.divsi %10, %c512 : index
        %13 = memref.load %arg0[%12, %11, %9, %7] : memref<512x512x3x3xf32>
        %14 = arith.truncf %13 : f32 to f16
        memref.store %14, %arg1[%12, %11, %9, %7] : memref<512x512x3x3xf16>
      }
      gpu.return
    }
    gpu.func @Unknown48(%arg0: memref<1x512x7x7xf16>, %arg1: memref<1x512x7x7xf16>) kernel {
      %c25088 = arith.constant 25088 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c7 = arith.constant 7 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = gpu.grid_dim  x
      %6 = arith.muli %1, %5 : index
      scf.for %arg2 = %4 to %c25088 step %6 {
        %7 = arith.remsi %arg2, %c7 : index
        %8 = arith.divsi %arg2, %c7 : index
        %9 = arith.remsi %8, %c7 : index
        %10 = arith.divsi %8, %c7 : index
        %11 = memref.load %arg0[%c0, %10, %9, %7] : memref<1x512x7x7xf16>
        %12 = arith.maximumf %11, %cst : f16
        memref.store %12, %arg1[%c0, %10, %9, %7] : memref<1x512x7x7xf16>
      }
      gpu.return
    }
    gpu.func @Unknown46(%arg0: memref<512x256x3x3xf32>, %arg1: memref<512x256x3x3xf16>) kernel {
      %c1179648 = arith.constant 1179648 : index
      %c256 = arith.constant 256 : index
      %c3 = arith.constant 3 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = gpu.grid_dim  x
      %6 = arith.muli %1, %5 : index
      scf.for %arg2 = %4 to %c1179648 step %6 {
        %7 = arith.remsi %arg2, %c3 : index
        %8 = arith.divsi %arg2, %c3 : index
        %9 = arith.remsi %8, %c3 : index
        %10 = arith.divsi %8, %c3 : index
        %11 = arith.remsi %10, %c256 : index
        %12 = arith.divsi %10, %c256 : index
        %13 = memref.load %arg0[%12, %11, %9, %7] : memref<512x256x3x3xf32>
        %14 = arith.truncf %13 : f32 to f16
        memref.store %14, %arg1[%12, %11, %9, %7] : memref<512x256x3x3xf16>
      }
      gpu.return
    }
    gpu.func @Unknown44(%arg0: memref<512x256x1x1xf32>, %arg1: memref<512x256x1x1xf16>) kernel {
      %c131072 = arith.constant 131072 : index
      %c0 = arith.constant 0 : index
      %c256 = arith.constant 256 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = gpu.grid_dim  x
      %6 = arith.muli %1, %5 : index
      scf.for %arg2 = %4 to %c131072 step %6 {
        %7 = arith.remsi %arg2, %c256 : index
        %8 = arith.divsi %arg2, %c256 : index
        %9 = memref.load %arg0[%8, %7, %c0, %c0] : memref<512x256x1x1xf32>
        %10 = arith.truncf %9 : f32 to f16
        memref.store %10, %arg1[%8, %7, %c0, %c0] : memref<512x256x1x1xf16>
      }
      gpu.return
    }
    gpu.func @Unknown37(%arg0: memref<1x256x14x14xf16>, %arg1: memref<1x256x14x14xf16>, %arg2: memref<1x256x14x14xf16>) kernel {
      %c50176 = arith.constant 50176 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c14 = arith.constant 14 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = gpu.grid_dim  x
      %6 = arith.muli %1, %5 : index
      scf.for %arg3 = %4 to %c50176 step %6 {
        %7 = arith.remsi %arg3, %c14 : index
        %8 = arith.divsi %arg3, %c14 : index
        %9 = arith.remsi %8, %c14 : index
        %10 = arith.divsi %8, %c14 : index
        %11 = memref.load %arg0[%c0, %10, %9, %7] : memref<1x256x14x14xf16>
        %12 = memref.load %arg1[%c0, %10, %9, %7] : memref<1x256x14x14xf16>
        %13 = arith.addf %11, %12 : f16
        %14 = arith.maximumf %13, %cst : f16
        memref.store %14, %arg2[%c0, %10, %9, %7] : memref<1x256x14x14xf16>
      }
      gpu.return
    }
    gpu.func @Unknown35(%arg0: memref<256x256x3x3xf32>, %arg1: memref<256x256x3x3xf16>) kernel {
      %c589824 = arith.constant 589824 : index
      %c256 = arith.constant 256 : index
      %c3 = arith.constant 3 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = gpu.grid_dim  x
      %6 = arith.muli %1, %5 : index
      scf.for %arg2 = %4 to %c589824 step %6 {
        %7 = arith.remsi %arg2, %c3 : index
        %8 = arith.divsi %arg2, %c3 : index
        %9 = arith.remsi %8, %c3 : index
        %10 = arith.divsi %8, %c3 : index
        %11 = arith.remsi %10, %c256 : index
        %12 = arith.divsi %10, %c256 : index
        %13 = memref.load %arg0[%12, %11, %9, %7] : memref<256x256x3x3xf32>
        %14 = arith.truncf %13 : f32 to f16
        memref.store %14, %arg1[%12, %11, %9, %7] : memref<256x256x3x3xf16>
      }
      gpu.return
    }
    gpu.func @Unknown34(%arg0: memref<1x256x14x14xf16>, %arg1: memref<1x256x14x14xf16>) kernel {
      %c50176 = arith.constant 50176 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c14 = arith.constant 14 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = gpu.grid_dim  x
      %6 = arith.muli %1, %5 : index
      scf.for %arg2 = %4 to %c50176 step %6 {
        %7 = arith.remsi %arg2, %c14 : index
        %8 = arith.divsi %arg2, %c14 : index
        %9 = arith.remsi %8, %c14 : index
        %10 = arith.divsi %8, %c14 : index
        %11 = memref.load %arg0[%c0, %10, %9, %7] : memref<1x256x14x14xf16>
        %12 = arith.maximumf %11, %cst : f16
        memref.store %12, %arg1[%c0, %10, %9, %7] : memref<1x256x14x14xf16>
      }
      gpu.return
    }
    gpu.func @Unknown32(%arg0: memref<256x128x3x3xf32>, %arg1: memref<256x128x3x3xf16>) kernel {
      %c294912 = arith.constant 294912 : index
      %c128 = arith.constant 128 : index
      %c3 = arith.constant 3 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = gpu.grid_dim  x
      %6 = arith.muli %1, %5 : index
      scf.for %arg2 = %4 to %c294912 step %6 {
        %7 = arith.remsi %arg2, %c3 : index
        %8 = arith.divsi %arg2, %c3 : index
        %9 = arith.remsi %8, %c3 : index
        %10 = arith.divsi %8, %c3 : index
        %11 = arith.remsi %10, %c128 : index
        %12 = arith.divsi %10, %c128 : index
        %13 = memref.load %arg0[%12, %11, %9, %7] : memref<256x128x3x3xf32>
        %14 = arith.truncf %13 : f32 to f16
        memref.store %14, %arg1[%12, %11, %9, %7] : memref<256x128x3x3xf16>
      }
      gpu.return
    }
    gpu.func @Unknown30(%arg0: memref<256x128x1x1xf32>, %arg1: memref<256x128x1x1xf16>) kernel {
      %c32768 = arith.constant 32768 : index
      %c0 = arith.constant 0 : index
      %c128 = arith.constant 128 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = gpu.grid_dim  x
      %6 = arith.muli %1, %5 : index
      scf.for %arg2 = %4 to %c32768 step %6 {
        %7 = arith.remsi %arg2, %c128 : index
        %8 = arith.divsi %arg2, %c128 : index
        %9 = memref.load %arg0[%8, %7, %c0, %c0] : memref<256x128x1x1xf32>
        %10 = arith.truncf %9 : f32 to f16
        memref.store %10, %arg1[%8, %7, %c0, %c0] : memref<256x128x1x1xf16>
      }
      gpu.return
    }
    gpu.func @Unknown23(%arg0: memref<1x128x28x28xf16>, %arg1: memref<1x128x28x28xf16>, %arg2: memref<1x128x28x28xf16>) kernel {
      %c100352 = arith.constant 100352 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c28 = arith.constant 28 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = gpu.grid_dim  x
      %6 = arith.muli %1, %5 : index
      scf.for %arg3 = %4 to %c100352 step %6 {
        %7 = arith.remsi %arg3, %c28 : index
        %8 = arith.divsi %arg3, %c28 : index
        %9 = arith.remsi %8, %c28 : index
        %10 = arith.divsi %8, %c28 : index
        %11 = memref.load %arg0[%c0, %10, %9, %7] : memref<1x128x28x28xf16>
        %12 = memref.load %arg1[%c0, %10, %9, %7] : memref<1x128x28x28xf16>
        %13 = arith.addf %11, %12 : f16
        %14 = arith.maximumf %13, %cst : f16
        memref.store %14, %arg2[%c0, %10, %9, %7] : memref<1x128x28x28xf16>
      }
      gpu.return
    }
    gpu.func @Unknown21(%arg0: memref<128x128x3x3xf32>, %arg1: memref<128x128x3x3xf16>) kernel {
      %c147456 = arith.constant 147456 : index
      %c128 = arith.constant 128 : index
      %c3 = arith.constant 3 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = gpu.grid_dim  x
      %6 = arith.muli %1, %5 : index
      scf.for %arg2 = %4 to %c147456 step %6 {
        %7 = arith.remsi %arg2, %c3 : index
        %8 = arith.divsi %arg2, %c3 : index
        %9 = arith.remsi %8, %c3 : index
        %10 = arith.divsi %8, %c3 : index
        %11 = arith.remsi %10, %c128 : index
        %12 = arith.divsi %10, %c128 : index
        %13 = memref.load %arg0[%12, %11, %9, %7] : memref<128x128x3x3xf32>
        %14 = arith.truncf %13 : f32 to f16
        memref.store %14, %arg1[%12, %11, %9, %7] : memref<128x128x3x3xf16>
      }
      gpu.return
    }
    gpu.func @Unknown20(%arg0: memref<1x128x28x28xf16>, %arg1: memref<1x128x28x28xf16>) kernel {
      %c100352 = arith.constant 100352 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c28 = arith.constant 28 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = gpu.grid_dim  x
      %6 = arith.muli %1, %5 : index
      scf.for %arg2 = %4 to %c100352 step %6 {
        %7 = arith.remsi %arg2, %c28 : index
        %8 = arith.divsi %arg2, %c28 : index
        %9 = arith.remsi %8, %c28 : index
        %10 = arith.divsi %8, %c28 : index
        %11 = memref.load %arg0[%c0, %10, %9, %7] : memref<1x128x28x28xf16>
        %12 = arith.maximumf %11, %cst : f16
        memref.store %12, %arg1[%c0, %10, %9, %7] : memref<1x128x28x28xf16>
      }
      gpu.return
    }
    gpu.func @Unknown18(%arg0: memref<128x64x3x3xf32>, %arg1: memref<128x64x3x3xf16>) kernel {
      %c73728 = arith.constant 73728 : index
      %c64 = arith.constant 64 : index
      %c3 = arith.constant 3 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = gpu.grid_dim  x
      %6 = arith.muli %1, %5 : index
      scf.for %arg2 = %4 to %c73728 step %6 {
        %7 = arith.remsi %arg2, %c3 : index
        %8 = arith.divsi %arg2, %c3 : index
        %9 = arith.remsi %8, %c3 : index
        %10 = arith.divsi %8, %c3 : index
        %11 = arith.remsi %10, %c64 : index
        %12 = arith.divsi %10, %c64 : index
        %13 = memref.load %arg0[%12, %11, %9, %7] : memref<128x64x3x3xf32>
        %14 = arith.truncf %13 : f32 to f16
        memref.store %14, %arg1[%12, %11, %9, %7] : memref<128x64x3x3xf16>
      }
      gpu.return
    }
    gpu.func @Unknown16(%arg0: memref<128x64x1x1xf32>, %arg1: memref<128x64x1x1xf16>) kernel {
      %c8192 = arith.constant 8192 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = gpu.grid_dim  x
      %6 = arith.muli %1, %5 : index
      scf.for %arg2 = %4 to %c8192 step %6 {
        %7 = arith.remsi %arg2, %c64 : index
        %8 = arith.divsi %arg2, %c64 : index
        %9 = memref.load %arg0[%8, %7, %c0, %c0] : memref<128x64x1x1xf32>
        %10 = arith.truncf %9 : f32 to f16
        memref.store %10, %arg1[%8, %7, %c0, %c0] : memref<128x64x1x1xf16>
      }
      gpu.return
    }
    gpu.func @Unknown9(%arg0: memref<1x64x56x56xf16>, %arg1: memref<1x64x56x56xf16>, %arg2: memref<1x64x56x56xf16>) kernel {
      %c200704 = arith.constant 200704 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c56 = arith.constant 56 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = gpu.grid_dim  x
      %6 = arith.muli %1, %5 : index
      scf.for %arg3 = %4 to %c200704 step %6 {
        %7 = arith.remsi %arg3, %c56 : index
        %8 = arith.divsi %arg3, %c56 : index
        %9 = arith.remsi %8, %c56 : index
        %10 = arith.divsi %8, %c56 : index
        %11 = memref.load %arg0[%c0, %10, %9, %7] : memref<1x64x56x56xf16>
        %12 = memref.load %arg1[%c0, %10, %9, %7] : memref<1x64x56x56xf16>
        %13 = arith.addf %11, %12 : f16
        %14 = arith.maximumf %13, %cst : f16
        memref.store %14, %arg2[%c0, %10, %9, %7] : memref<1x64x56x56xf16>
      }
      gpu.return
    }
    gpu.func @Unknown6(%arg0: memref<1x64x56x56xf16>, %arg1: memref<1x64x56x56xf16>) kernel {
      %c200704 = arith.constant 200704 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c56 = arith.constant 56 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = gpu.grid_dim  x
      %6 = arith.muli %1, %5 : index
      scf.for %arg2 = %4 to %c200704 step %6 {
        %7 = arith.remsi %arg2, %c56 : index
        %8 = arith.divsi %arg2, %c56 : index
        %9 = arith.remsi %8, %c56 : index
        %10 = arith.divsi %8, %c56 : index
        %11 = memref.load %arg0[%c0, %10, %9, %7] : memref<1x64x56x56xf16>
        %12 = arith.maximumf %11, %cst : f16
        memref.store %12, %arg1[%c0, %10, %9, %7] : memref<1x64x56x56xf16>
      }
      gpu.return
    }
    gpu.func @Unknown4(%arg0: memref<64x64x3x3xf32>, %arg1: memref<64x64x3x3xf16>) kernel {
      %c36864 = arith.constant 36864 : index
      %c64 = arith.constant 64 : index
      %c3 = arith.constant 3 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = gpu.grid_dim  x
      %6 = arith.muli %1, %5 : index
      scf.for %arg2 = %4 to %c36864 step %6 {
        %7 = arith.remsi %arg2, %c3 : index
        %8 = arith.divsi %arg2, %c3 : index
        %9 = arith.remsi %8, %c3 : index
        %10 = arith.divsi %8, %c3 : index
        %11 = arith.remsi %10, %c64 : index
        %12 = arith.divsi %10, %c64 : index
        %13 = memref.load %arg0[%12, %11, %9, %7] : memref<64x64x3x3xf32>
        %14 = arith.truncf %13 : f32 to f16
        memref.store %14, %arg1[%12, %11, %9, %7] : memref<64x64x3x3xf16>
      }
      gpu.return
    }
    gpu.func @Unknown3(%arg0: memref<1x64x112x112xf16>, %arg1: memref<1x64x112x112xf16>) kernel {
      %c802816 = arith.constant 802816 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c112 = arith.constant 112 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = gpu.grid_dim  x
      %6 = arith.muli %1, %5 : index
      scf.for %arg2 = %4 to %c802816 step %6 {
        %7 = arith.remsi %arg2, %c112 : index
        %8 = arith.divsi %arg2, %c112 : index
        %9 = arith.remsi %8, %c112 : index
        %10 = arith.divsi %8, %c112 : index
        %11 = memref.load %arg0[%c0, %10, %9, %7] : memref<1x64x112x112xf16>
        %12 = arith.maximumf %11, %cst : f16
        memref.store %12, %arg1[%c0, %10, %9, %7] : memref<1x64x112x112xf16>
      }
      gpu.return
    }
    gpu.func @Unknown1(%arg0: memref<64x3x7x7xf32>, %arg1: memref<64x3x7x7xf16>) kernel {
      %c9408 = arith.constant 9408 : index
      %c3 = arith.constant 3 : index
      %c7 = arith.constant 7 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = gpu.grid_dim  x
      %6 = arith.muli %1, %5 : index
      scf.for %arg2 = %4 to %c9408 step %6 {
        %7 = arith.remsi %arg2, %c7 : index
        %8 = arith.divsi %arg2, %c7 : index
        %9 = arith.remsi %8, %c7 : index
        %10 = arith.divsi %8, %c7 : index
        %11 = arith.remsi %10, %c3 : index
        %12 = arith.divsi %10, %c3 : index
        %13 = memref.load %arg0[%12, %11, %9, %7] : memref<64x3x7x7xf32>
        %14 = arith.truncf %13 : f32 to f16
        memref.store %14, %arg1[%12, %11, %9, %7] : memref<64x3x7x7xf16>
      }
      gpu.return
    }
    gpu.func @Unknown0(%arg0: memref<1x3x224x224xf32>, %arg1: memref<1x3x224x224xf16>) kernel {
      %c150528 = arith.constant 150528 : index
      %c0 = arith.constant 0 : index
      %c224 = arith.constant 224 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = gpu.grid_dim  x
      %6 = arith.muli %1, %5 : index
      scf.for %arg2 = %4 to %c150528 step %6 {
        %7 = arith.remsi %arg2, %c224 : index
        %8 = arith.divsi %arg2, %c224 : index
        %9 = arith.remsi %8, %c224 : index
        %10 = arith.divsi %8, %c224 : index
        %11 = memref.load %arg0[%c0, %10, %9, %7] : memref<1x3x224x224xf32>
        %12 = arith.truncf %11 : f32 to f16
        memref.store %12, %arg1[%c0, %10, %9, %7] : memref<1x3x224x224xf16>
      }
      gpu.return
    }
    gpu.func @Unknown58_kernel(%arg0: memref<512x49xf16>, %arg1: memref<512xf16>) kernel attributes {gpu.known_block_size = array<i32: 64, 1, 1>, gpu.known_grid_size = array<i32: 512, 1, 1>} {
      %c4 = arith.constant 4 : index
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %c2 = arith.constant 2 : index
      %c32 = arith.constant 32 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c1 = arith.constant 1 : index
      %c49 = arith.constant 49 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %0 = gpu.block_id  x
      %subview = memref.subview %arg0[%0, 0] [1, 49] [1, 1] : memref<512x49xf16> to memref<49xf16, strided<[1], offset: ?>>
      %expand_shape = memref.expand_shape %subview [[0, 1]] : memref<49xf16, strided<[1], offset: ?>> into memref<1x49xf16, strided<[49, 1], offset: ?>>
      %alloca = memref.alloca() : memref<64xf16, #gpu.address_space<workgroup>>
      %1 = gpu.thread_id  x
      %2 = arith.remsi %1, %c64 : index
      %3 = arith.cmpi slt, %2, %c0 : index
      %4 = arith.addi %2, %c64 : index
      %5 = arith.select %3, %4, %2 : index
      %6 = arith.cmpi slt, %5, %c49 : index
      %7 = arith.select %6, %5, %c49 : index
      %8 = arith.addi %5, %c1 : index
      %9 = arith.cmpi slt, %8, %c49 : index
      %10 = arith.select %9, %8, %c49 : index
      %11 = arith.subi %10, %7 : index
      %subview_0 = memref.subview %expand_shape[0, %7] [1, %11] [1, 1] : memref<1x49xf16, strided<[49, 1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
      %expand_shape_1 = memref.expand_shape %subview_0 [[0, 1]] : memref<?xf16, strided<[1], offset: ?>> into memref<1x?xf16, strided<[?, 1], offset: ?>>
      %12 = arith.cmpi ugt, %11, %c0 : index
      %13 = scf.if %12 -> (f16) {
        %21 = memref.load %expand_shape_1[%c0, %c0] : memref<1x?xf16, strided<[?, 1], offset: ?>>
        scf.yield %21 : f16
      } else {
        scf.yield %cst : f16
      }
      %14 = arith.addf %13, %cst : f16
      memref.store %14, %alloca[%1] : memref<64xf16, #gpu.address_space<workgroup>>
      gpu.barrier
      %alloca_2 = memref.alloca() : memref<32xf16, #gpu.address_space<workgroup>>
      %15 = arith.cmpi ult, %1, %c32 : index
      scf.if %15 {
        %21 = arith.muli %1, %c2 : index
        %22 = memref.load %alloca[%21] : memref<64xf16, #gpu.address_space<workgroup>>
        %23 = arith.addf %22, %cst : f16
        %24 = arith.addi %21, %c1 : index
        %25 = memref.load %alloca[%24] : memref<64xf16, #gpu.address_space<workgroup>>
        %26 = arith.addf %25, %23 : f16
        memref.store %26, %alloca_2[%1] : memref<32xf16, #gpu.address_space<workgroup>>
      }
      gpu.barrier
      %alloca_3 = memref.alloca() : memref<16xf16, #gpu.address_space<workgroup>>
      %16 = arith.cmpi ult, %1, %c16 : index
      scf.if %16 {
        %21 = arith.muli %1, %c2 : index
        %22 = memref.load %alloca_2[%21] : memref<32xf16, #gpu.address_space<workgroup>>
        %23 = arith.addf %22, %cst : f16
        %24 = arith.addi %21, %c1 : index
        %25 = memref.load %alloca_2[%24] : memref<32xf16, #gpu.address_space<workgroup>>
        %26 = arith.addf %25, %23 : f16
        memref.store %26, %alloca_3[%1] : memref<16xf16, #gpu.address_space<workgroup>>
      }
      gpu.barrier
      %alloca_4 = memref.alloca() : memref<8xf16, #gpu.address_space<workgroup>>
      %17 = arith.cmpi ult, %1, %c8 : index
      scf.if %17 {
        %21 = arith.muli %1, %c2 : index
        %22 = memref.load %alloca_3[%21] : memref<16xf16, #gpu.address_space<workgroup>>
        %23 = arith.addf %22, %cst : f16
        %24 = arith.addi %21, %c1 : index
        %25 = memref.load %alloca_3[%24] : memref<16xf16, #gpu.address_space<workgroup>>
        %26 = arith.addf %25, %23 : f16
        memref.store %26, %alloca_4[%1] : memref<8xf16, #gpu.address_space<workgroup>>
      }
      gpu.barrier
      %alloca_5 = memref.alloca() : memref<4xf16, #gpu.address_space<workgroup>>
      %18 = arith.cmpi ult, %1, %c4 : index
      scf.if %18 {
        %21 = arith.muli %1, %c2 : index
        %22 = memref.load %alloca_4[%21] : memref<8xf16, #gpu.address_space<workgroup>>
        %23 = arith.addf %22, %cst : f16
        %24 = arith.addi %21, %c1 : index
        %25 = memref.load %alloca_4[%24] : memref<8xf16, #gpu.address_space<workgroup>>
        %26 = arith.addf %25, %23 : f16
        memref.store %26, %alloca_5[%1] : memref<4xf16, #gpu.address_space<workgroup>>
      }
      gpu.barrier
      %alloca_6 = memref.alloca() : memref<2xf16, #gpu.address_space<workgroup>>
      %19 = arith.cmpi ult, %1, %c2 : index
      scf.if %19 {
        %21 = arith.muli %1, %c2 : index
        %22 = memref.load %alloca_5[%21] : memref<4xf16, #gpu.address_space<workgroup>>
        %23 = arith.addf %22, %cst : f16
        %24 = arith.addi %21, %c1 : index
        %25 = memref.load %alloca_5[%24] : memref<4xf16, #gpu.address_space<workgroup>>
        %26 = arith.addf %25, %23 : f16
        memref.store %26, %alloca_6[%1] : memref<2xf16, #gpu.address_space<workgroup>>
      }
      gpu.barrier
      %20 = arith.cmpi ult, %1, %c1 : index
      scf.if %20 {
        %21 = arith.muli %1, %c2 : index
        %22 = memref.load %alloca_6[%21] : memref<2xf16, #gpu.address_space<workgroup>>
        %23 = arith.addf %22, %cst : f16
        %24 = arith.addi %21, %c1 : index
        %25 = memref.load %alloca_6[%24] : memref<2xf16, #gpu.address_space<workgroup>>
        %26 = arith.addf %25, %23 : f16
        memref.store %26, %arg1[%0] : memref<512xf16>
      }
      gpu.barrier
      gpu.return
    }
  }
  func.func @main(%arg0: memref<64xf32, "cuda"> {byre.argname = "Input0", byre.argtype = 1 : i32}, %arg1: memref<64xf32, "cuda"> {byre.argname = "Input1", byre.argtype = 1 : i32}, %arg2: memref<64x3x7x7xf32, "cuda"> {byre.argname = "Input2", byre.argtype = 1 : i32}, %arg3: memref<1000xf32, "cuda"> {byre.argname = "Input3", byre.argtype = 1 : i32}, %arg4: memref<1000x512xf32, "cuda"> {byre.argname = "Input4", byre.argtype = 1 : i32}, %arg5: memref<64xf32, "cuda"> {byre.argname = "Input5", byre.argtype = 1 : i32}, %arg6: memref<64xf32, "cuda"> {byre.argname = "Input6", byre.argtype = 1 : i32}, %arg7: memref<64xf32, "cuda"> {byre.argname = "Input7", byre.argtype = 1 : i32}, %arg8: memref<64xf32, "cuda"> {byre.argname = "Input8", byre.argtype = 1 : i32}, %arg9: memref<64x64x3x3xf32, "cuda"> {byre.argname = "Input9", byre.argtype = 1 : i32}, %arg10: memref<64x64x3x3xf32, "cuda"> {byre.argname = "Input10", byre.argtype = 1 : i32}, %arg11: memref<64xf32, "cuda"> {byre.argname = "Input11", byre.argtype = 1 : i32}, %arg12: memref<64xf32, "cuda"> {byre.argname = "Input12", byre.argtype = 1 : i32}, %arg13: memref<64xf32, "cuda"> {byre.argname = "Input13", byre.argtype = 1 : i32}, %arg14: memref<64xf32, "cuda"> {byre.argname = "Input14", byre.argtype = 1 : i32}, %arg15: memref<64x64x3x3xf32, "cuda"> {byre.argname = "Input15", byre.argtype = 1 : i32}, %arg16: memref<64x64x3x3xf32, "cuda"> {byre.argname = "Input16", byre.argtype = 1 : i32}, %arg17: memref<128xf32, "cuda"> {byre.argname = "Input17", byre.argtype = 1 : i32}, %arg18: memref<128xf32, "cuda"> {byre.argname = "Input18", byre.argtype = 1 : i32}, %arg19: memref<128xf32, "cuda"> {byre.argname = "Input19", byre.argtype = 1 : i32}, %arg20: memref<128xf32, "cuda"> {byre.argname = "Input20", byre.argtype = 1 : i32}, %arg21: memref<128x64x3x3xf32, "cuda"> {byre.argname = "Input21", byre.argtype = 1 : i32}, %arg22: memref<128x128x3x3xf32, "cuda"> {byre.argname = "Input22", byre.argtype = 1 : i32}, %arg23: memref<128x64x1x1xf32, "cuda"> {byre.argname = "Input23", byre.argtype = 1 : i32}, %arg24: memref<128xf32, "cuda"> {byre.argname = "Input24", byre.argtype = 1 : i32}, %arg25: memref<128xf32, "cuda"> {byre.argname = "Input25", byre.argtype = 1 : i32}, %arg26: memref<128xf32, "cuda"> {byre.argname = "Input26", byre.argtype = 1 : i32}, %arg27: memref<128xf32, "cuda"> {byre.argname = "Input27", byre.argtype = 1 : i32}, %arg28: memref<128xf32, "cuda"> {byre.argname = "Input28", byre.argtype = 1 : i32}, %arg29: memref<128xf32, "cuda"> {byre.argname = "Input29", byre.argtype = 1 : i32}, %arg30: memref<128x128x3x3xf32, "cuda"> {byre.argname = "Input30", byre.argtype = 1 : i32}, %arg31: memref<128x128x3x3xf32, "cuda"> {byre.argname = "Input31", byre.argtype = 1 : i32}, %arg32: memref<256xf32, "cuda"> {byre.argname = "Input32", byre.argtype = 1 : i32}, %arg33: memref<256xf32, "cuda"> {byre.argname = "Input33", byre.argtype = 1 : i32}, %arg34: memref<256xf32, "cuda"> {byre.argname = "Input34", byre.argtype = 1 : i32}, %arg35: memref<256xf32, "cuda"> {byre.argname = "Input35", byre.argtype = 1 : i32}, %arg36: memref<256x128x3x3xf32, "cuda"> {byre.argname = "Input36", byre.argtype = 1 : i32}, %arg37: memref<256x256x3x3xf32, "cuda"> {byre.argname = "Input37", byre.argtype = 1 : i32}, %arg38: memref<256x128x1x1xf32, "cuda"> {byre.argname = "Input38", byre.argtype = 1 : i32}, %arg39: memref<256xf32, "cuda"> {byre.argname = "Input39", byre.argtype = 1 : i32}, %arg40: memref<256xf32, "cuda"> {byre.argname = "Input40", byre.argtype = 1 : i32}, %arg41: memref<256xf32, "cuda"> {byre.argname = "Input41", byre.argtype = 1 : i32}, %arg42: memref<256xf32, "cuda"> {byre.argname = "Input42", byre.argtype = 1 : i32}, %arg43: memref<256xf32, "cuda"> {byre.argname = "Input43", byre.argtype = 1 : i32}, %arg44: memref<256xf32, "cuda"> {byre.argname = "Input44", byre.argtype = 1 : i32}, %arg45: memref<256x256x3x3xf32, "cuda"> {byre.argname = "Input45", byre.argtype = 1 : i32}, %arg46: memref<256x256x3x3xf32, "cuda"> {byre.argname = "Input46", byre.argtype = 1 : i32}, %arg47: memref<512xf32, "cuda"> {byre.argname = "Input47", byre.argtype = 1 : i32}, %arg48: memref<512xf32, "cuda"> {byre.argname = "Input48", byre.argtype = 1 : i32}, %arg49: memref<512xf32, "cuda"> {byre.argname = "Input49", byre.argtype = 1 : i32}, %arg50: memref<512xf32, "cuda"> {byre.argname = "Input50", byre.argtype = 1 : i32}, %arg51: memref<512x256x3x3xf32, "cuda"> {byre.argname = "Input51", byre.argtype = 1 : i32}, %arg52: memref<512x512x3x3xf32, "cuda"> {byre.argname = "Input52", byre.argtype = 1 : i32}, %arg53: memref<512x256x1x1xf32, "cuda"> {byre.argname = "Input53", byre.argtype = 1 : i32}, %arg54: memref<512xf32, "cuda"> {byre.argname = "Input54", byre.argtype = 1 : i32}, %arg55: memref<512xf32, "cuda"> {byre.argname = "Input55", byre.argtype = 1 : i32}, %arg56: memref<512xf32, "cuda"> {byre.argname = "Input56", byre.argtype = 1 : i32}, %arg57: memref<512xf32, "cuda"> {byre.argname = "Input57", byre.argtype = 1 : i32}, %arg58: memref<512xf32, "cuda"> {byre.argname = "Input58", byre.argtype = 1 : i32}, %arg59: memref<512xf32, "cuda"> {byre.argname = "Input59", byre.argtype = 1 : i32}, %arg60: memref<512x512x3x3xf32, "cuda"> {byre.argname = "Input60", byre.argtype = 1 : i32}, %arg61: memref<512x512x3x3xf32, "cuda"> {byre.argname = "Input61", byre.argtype = 1 : i32}, %arg62: memref<i64, "cuda"> {byre.argname = "Input62", byre.argtype = 1 : i32}, %arg63: memref<64xf32, "cuda"> {byre.argname = "Input63", byre.argtype = 1 : i32}, %arg64: memref<64xf32, "cuda"> {byre.argname = "Input64", byre.argtype = 1 : i32}, %arg65: memref<i64, "cuda"> {byre.argname = "Input65", byre.argtype = 1 : i32}, %arg66: memref<64xf32, "cuda"> {byre.argname = "Input66", byre.argtype = 1 : i32}, %arg67: memref<64xf32, "cuda"> {byre.argname = "Input67", byre.argtype = 1 : i32}, %arg68: memref<i64, "cuda"> {byre.argname = "Input68", byre.argtype = 1 : i32}, %arg69: memref<64xf32, "cuda"> {byre.argname = "Input69", byre.argtype = 1 : i32}, %arg70: memref<64xf32, "cuda"> {byre.argname = "Input70", byre.argtype = 1 : i32}, %arg71: memref<i64, "cuda"> {byre.argname = "Input71", byre.argtype = 1 : i32}, %arg72: memref<64xf32, "cuda"> {byre.argname = "Input72", byre.argtype = 1 : i32}, %arg73: memref<64xf32, "cuda"> {byre.argname = "Input73", byre.argtype = 1 : i32}, %arg74: memref<i64, "cuda"> {byre.argname = "Input74", byre.argtype = 1 : i32}, %arg75: memref<64xf32, "cuda"> {byre.argname = "Input75", byre.argtype = 1 : i32}, %arg76: memref<64xf32, "cuda"> {byre.argname = "Input76", byre.argtype = 1 : i32}, %arg77: memref<i64, "cuda"> {byre.argname = "Input77", byre.argtype = 1 : i32}, %arg78: memref<128xf32, "cuda"> {byre.argname = "Input78", byre.argtype = 1 : i32}, %arg79: memref<128xf32, "cuda"> {byre.argname = "Input79", byre.argtype = 1 : i32}, %arg80: memref<i64, "cuda"> {byre.argname = "Input80", byre.argtype = 1 : i32}, %arg81: memref<128xf32, "cuda"> {byre.argname = "Input81", byre.argtype = 1 : i32}, %arg82: memref<128xf32, "cuda"> {byre.argname = "Input82", byre.argtype = 1 : i32}, %arg83: memref<i64, "cuda"> {byre.argname = "Input83", byre.argtype = 1 : i32}, %arg84: memref<128xf32, "cuda"> {byre.argname = "Input84", byre.argtype = 1 : i32}, %arg85: memref<128xf32, "cuda"> {byre.argname = "Input85", byre.argtype = 1 : i32}, %arg86: memref<i64, "cuda"> {byre.argname = "Input86", byre.argtype = 1 : i32}, %arg87: memref<128xf32, "cuda"> {byre.argname = "Input87", byre.argtype = 1 : i32}, %arg88: memref<128xf32, "cuda"> {byre.argname = "Input88", byre.argtype = 1 : i32}, %arg89: memref<i64, "cuda"> {byre.argname = "Input89", byre.argtype = 1 : i32}, %arg90: memref<128xf32, "cuda"> {byre.argname = "Input90", byre.argtype = 1 : i32}, %arg91: memref<128xf32, "cuda"> {byre.argname = "Input91", byre.argtype = 1 : i32}, %arg92: memref<i64, "cuda"> {byre.argname = "Input92", byre.argtype = 1 : i32}, %arg93: memref<256xf32, "cuda"> {byre.argname = "Input93", byre.argtype = 1 : i32}, %arg94: memref<256xf32, "cuda"> {byre.argname = "Input94", byre.argtype = 1 : i32}, %arg95: memref<i64, "cuda"> {byre.argname = "Input95", byre.argtype = 1 : i32}, %arg96: memref<256xf32, "cuda"> {byre.argname = "Input96", byre.argtype = 1 : i32}, %arg97: memref<256xf32, "cuda"> {byre.argname = "Input97", byre.argtype = 1 : i32}, %arg98: memref<i64, "cuda"> {byre.argname = "Input98", byre.argtype = 1 : i32}, %arg99: memref<256xf32, "cuda"> {byre.argname = "Input99", byre.argtype = 1 : i32}, %arg100: memref<256xf32, "cuda"> {byre.argname = "Input100", byre.argtype = 1 : i32}, %arg101: memref<i64, "cuda"> {byre.argname = "Input101", byre.argtype = 1 : i32}, %arg102: memref<256xf32, "cuda"> {byre.argname = "Input102", byre.argtype = 1 : i32}, %arg103: memref<256xf32, "cuda"> {byre.argname = "Input103", byre.argtype = 1 : i32}, %arg104: memref<i64, "cuda"> {byre.argname = "Input104", byre.argtype = 1 : i32}, %arg105: memref<256xf32, "cuda"> {byre.argname = "Input105", byre.argtype = 1 : i32}, %arg106: memref<256xf32, "cuda"> {byre.argname = "Input106", byre.argtype = 1 : i32}, %arg107: memref<i64, "cuda"> {byre.argname = "Input107", byre.argtype = 1 : i32}, %arg108: memref<512xf32, "cuda"> {byre.argname = "Input108", byre.argtype = 1 : i32}, %arg109: memref<512xf32, "cuda"> {byre.argname = "Input109", byre.argtype = 1 : i32}, %arg110: memref<i64, "cuda"> {byre.argname = "Input110", byre.argtype = 1 : i32}, %arg111: memref<512xf32, "cuda"> {byre.argname = "Input111", byre.argtype = 1 : i32}, %arg112: memref<512xf32, "cuda"> {byre.argname = "Input112", byre.argtype = 1 : i32}, %arg113: memref<i64, "cuda"> {byre.argname = "Input113", byre.argtype = 1 : i32}, %arg114: memref<512xf32, "cuda"> {byre.argname = "Input114", byre.argtype = 1 : i32}, %arg115: memref<512xf32, "cuda"> {byre.argname = "Input115", byre.argtype = 1 : i32}, %arg116: memref<i64, "cuda"> {byre.argname = "Input116", byre.argtype = 1 : i32}, %arg117: memref<512xf32, "cuda"> {byre.argname = "Input117", byre.argtype = 1 : i32}, %arg118: memref<512xf32, "cuda"> {byre.argname = "Input118", byre.argtype = 1 : i32}, %arg119: memref<i64, "cuda"> {byre.argname = "Input119", byre.argtype = 1 : i32}, %arg120: memref<512xf32, "cuda"> {byre.argname = "Input120", byre.argtype = 1 : i32}, %arg121: memref<512xf32, "cuda"> {byre.argname = "Input121", byre.argtype = 1 : i32}, %arg122: memref<1x3x224x224xf32, "cuda"> {byre.argname = "Input122", byre.argtype = 1 : i32}, %arg123: memref<1x1000xf16, "cuda"> {byre.argname = "Output0", byre.argtype = 2 : i32}, %arg124: memref<64xf32, "cuda"> {byre.arg_alias_index = 0 : i64, byre.argname = "Output1", byre.argtype = 2 : i32}, %arg125: memref<64xf32, "cuda"> {byre.arg_alias_index = 1 : i64, byre.argname = "Output2", byre.argtype = 2 : i32}, %arg126: memref<64xf32, "cuda"> {byre.arg_alias_index = 5 : i64, byre.argname = "Output3", byre.argtype = 2 : i32}, %arg127: memref<64xf32, "cuda"> {byre.arg_alias_index = 6 : i64, byre.argname = "Output4", byre.argtype = 2 : i32}, %arg128: memref<64xf32, "cuda"> {byre.arg_alias_index = 7 : i64, byre.argname = "Output5", byre.argtype = 2 : i32}, %arg129: memref<64xf32, "cuda"> {byre.arg_alias_index = 8 : i64, byre.argname = "Output6", byre.argtype = 2 : i32}, %arg130: memref<64xf32, "cuda"> {byre.arg_alias_index = 11 : i64, byre.argname = "Output7", byre.argtype = 2 : i32}, %arg131: memref<64xf32, "cuda"> {byre.arg_alias_index = 12 : i64, byre.argname = "Output8", byre.argtype = 2 : i32}, %arg132: memref<64xf32, "cuda"> {byre.arg_alias_index = 13 : i64, byre.argname = "Output9", byre.argtype = 2 : i32}, %arg133: memref<64xf32, "cuda"> {byre.arg_alias_index = 14 : i64, byre.argname = "Output10", byre.argtype = 2 : i32}, %arg134: memref<128xf32, "cuda"> {byre.arg_alias_index = 17 : i64, byre.argname = "Output11", byre.argtype = 2 : i32}, %arg135: memref<128xf32, "cuda"> {byre.arg_alias_index = 18 : i64, byre.argname = "Output12", byre.argtype = 2 : i32}, %arg136: memref<128xf32, "cuda"> {byre.arg_alias_index = 19 : i64, byre.argname = "Output13", byre.argtype = 2 : i32}, %arg137: memref<128xf32, "cuda"> {byre.arg_alias_index = 20 : i64, byre.argname = "Output14", byre.argtype = 2 : i32}, %arg138: memref<128xf32, "cuda"> {byre.arg_alias_index = 24 : i64, byre.argname = "Output15", byre.argtype = 2 : i32}, %arg139: memref<128xf32, "cuda"> {byre.arg_alias_index = 25 : i64, byre.argname = "Output16", byre.argtype = 2 : i32}, %arg140: memref<128xf32, "cuda"> {byre.arg_alias_index = 26 : i64, byre.argname = "Output17", byre.argtype = 2 : i32}, %arg141: memref<128xf32, "cuda"> {byre.arg_alias_index = 27 : i64, byre.argname = "Output18", byre.argtype = 2 : i32}, %arg142: memref<128xf32, "cuda"> {byre.arg_alias_index = 28 : i64, byre.argname = "Output19", byre.argtype = 2 : i32}, %arg143: memref<128xf32, "cuda"> {byre.arg_alias_index = 29 : i64, byre.argname = "Output20", byre.argtype = 2 : i32}, %arg144: memref<256xf32, "cuda"> {byre.arg_alias_index = 32 : i64, byre.argname = "Output21", byre.argtype = 2 : i32}, %arg145: memref<256xf32, "cuda"> {byre.arg_alias_index = 33 : i64, byre.argname = "Output22", byre.argtype = 2 : i32}, %arg146: memref<256xf32, "cuda"> {byre.arg_alias_index = 34 : i64, byre.argname = "Output23", byre.argtype = 2 : i32}, %arg147: memref<256xf32, "cuda"> {byre.arg_alias_index = 35 : i64, byre.argname = "Output24", byre.argtype = 2 : i32}, %arg148: memref<256xf32, "cuda"> {byre.arg_alias_index = 39 : i64, byre.argname = "Output25", byre.argtype = 2 : i32}, %arg149: memref<256xf32, "cuda"> {byre.arg_alias_index = 40 : i64, byre.argname = "Output26", byre.argtype = 2 : i32}, %arg150: memref<256xf32, "cuda"> {byre.arg_alias_index = 41 : i64, byre.argname = "Output27", byre.argtype = 2 : i32}, %arg151: memref<256xf32, "cuda"> {byre.arg_alias_index = 42 : i64, byre.argname = "Output28", byre.argtype = 2 : i32}, %arg152: memref<256xf32, "cuda"> {byre.arg_alias_index = 43 : i64, byre.argname = "Output29", byre.argtype = 2 : i32}, %arg153: memref<256xf32, "cuda"> {byre.arg_alias_index = 44 : i64, byre.argname = "Output30", byre.argtype = 2 : i32}, %arg154: memref<512xf32, "cuda"> {byre.arg_alias_index = 47 : i64, byre.argname = "Output31", byre.argtype = 2 : i32}, %arg155: memref<512xf32, "cuda"> {byre.arg_alias_index = 48 : i64, byre.argname = "Output32", byre.argtype = 2 : i32}, %arg156: memref<512xf32, "cuda"> {byre.arg_alias_index = 49 : i64, byre.argname = "Output33", byre.argtype = 2 : i32}, %arg157: memref<512xf32, "cuda"> {byre.arg_alias_index = 50 : i64, byre.argname = "Output34", byre.argtype = 2 : i32}, %arg158: memref<512xf32, "cuda"> {byre.arg_alias_index = 54 : i64, byre.argname = "Output35", byre.argtype = 2 : i32}, %arg159: memref<512xf32, "cuda"> {byre.arg_alias_index = 55 : i64, byre.argname = "Output36", byre.argtype = 2 : i32}, %arg160: memref<512xf32, "cuda"> {byre.arg_alias_index = 56 : i64, byre.argname = "Output37", byre.argtype = 2 : i32}, %arg161: memref<512xf32, "cuda"> {byre.arg_alias_index = 57 : i64, byre.argname = "Output38", byre.argtype = 2 : i32}, %arg162: memref<512xf32, "cuda"> {byre.arg_alias_index = 58 : i64, byre.argname = "Output39", byre.argtype = 2 : i32}, %arg163: memref<512xf32, "cuda"> {byre.arg_alias_index = 59 : i64, byre.argname = "Output40", byre.argtype = 2 : i32}, %arg164: memref<64xf32, "cuda"> {byre.argname = "Output41", byre.argtype = 2 : i32}, %arg165: memref<64xf32, "cuda"> {byre.argname = "Output42", byre.argtype = 2 : i32}, %arg166: memref<64xf32, "cuda"> {byre.argname = "Output43", byre.argtype = 2 : i32}, %arg167: memref<64xf32, "cuda"> {byre.argname = "Output44", byre.argtype = 2 : i32}, %arg168: memref<64xf32, "cuda"> {byre.argname = "Output45", byre.argtype = 2 : i32}, %arg169: memref<64xf32, "cuda"> {byre.argname = "Output46", byre.argtype = 2 : i32}, %arg170: memref<64xf32, "cuda"> {byre.argname = "Output47", byre.argtype = 2 : i32}, %arg171: memref<64xf32, "cuda"> {byre.argname = "Output48", byre.argtype = 2 : i32}, %arg172: memref<64xf32, "cuda"> {byre.argname = "Output49", byre.argtype = 2 : i32}, %arg173: memref<64xf32, "cuda"> {byre.argname = "Output50", byre.argtype = 2 : i32}, %arg174: memref<128xf32, "cuda"> {byre.argname = "Output51", byre.argtype = 2 : i32}, %arg175: memref<128xf32, "cuda"> {byre.argname = "Output52", byre.argtype = 2 : i32}, %arg176: memref<128xf32, "cuda"> {byre.argname = "Output53", byre.argtype = 2 : i32}, %arg177: memref<128xf32, "cuda"> {byre.argname = "Output54", byre.argtype = 2 : i32}, %arg178: memref<128xf32, "cuda"> {byre.argname = "Output55", byre.argtype = 2 : i32}, %arg179: memref<128xf32, "cuda"> {byre.argname = "Output56", byre.argtype = 2 : i32}, %arg180: memref<128xf32, "cuda"> {byre.argname = "Output57", byre.argtype = 2 : i32}, %arg181: memref<128xf32, "cuda"> {byre.argname = "Output58", byre.argtype = 2 : i32}, %arg182: memref<128xf32, "cuda"> {byre.argname = "Output59", byre.argtype = 2 : i32}, %arg183: memref<128xf32, "cuda"> {byre.argname = "Output60", byre.argtype = 2 : i32}, %arg184: memref<256xf32, "cuda"> {byre.argname = "Output61", byre.argtype = 2 : i32}, %arg185: memref<256xf32, "cuda"> {byre.argname = "Output62", byre.argtype = 2 : i32}, %arg186: memref<256xf32, "cuda"> {byre.argname = "Output63", byre.argtype = 2 : i32}, %arg187: memref<256xf32, "cuda"> {byre.argname = "Output64", byre.argtype = 2 : i32}, %arg188: memref<256xf32, "cuda"> {byre.argname = "Output65", byre.argtype = 2 : i32}, %arg189: memref<256xf32, "cuda"> {byre.argname = "Output66", byre.argtype = 2 : i32}, %arg190: memref<256xf32, "cuda"> {byre.argname = "Output67", byre.argtype = 2 : i32}, %arg191: memref<256xf32, "cuda"> {byre.argname = "Output68", byre.argtype = 2 : i32}, %arg192: memref<256xf32, "cuda"> {byre.argname = "Output69", byre.argtype = 2 : i32}, %arg193: memref<256xf32, "cuda"> {byre.argname = "Output70", byre.argtype = 2 : i32}, %arg194: memref<512xf32, "cuda"> {byre.argname = "Output71", byre.argtype = 2 : i32}, %arg195: memref<512xf32, "cuda"> {byre.argname = "Output72", byre.argtype = 2 : i32}, %arg196: memref<512xf32, "cuda"> {byre.argname = "Output73", byre.argtype = 2 : i32}, %arg197: memref<512xf32, "cuda"> {byre.argname = "Output74", byre.argtype = 2 : i32}, %arg198: memref<512xf32, "cuda"> {byre.argname = "Output75", byre.argtype = 2 : i32}, %arg199: memref<512xf32, "cuda"> {byre.argname = "Output76", byre.argtype = 2 : i32}, %arg200: memref<512xf32, "cuda"> {byre.argname = "Output77", byre.argtype = 2 : i32}, %arg201: memref<512xf32, "cuda"> {byre.argname = "Output78", byre.argtype = 2 : i32}, %arg202: memref<512xf32, "cuda"> {byre.argname = "Output79", byre.argtype = 2 : i32}, %arg203: memref<512xf32, "cuda"> {byre.argname = "Output80", byre.argtype = 2 : i32}, %arg204: memref<64x3x7x7xf16, "cuda"> {byre.argname = "Output81", byre.argtype = 2 : i32}, %arg205: memref<1x3x224x224xf16, "cuda"> {byre.argname = "Output82", byre.argtype = 2 : i32}, %arg206: memref<1x64x112x112xf16, "cuda"> {byre.argname = "Output83", byre.argtype = 2 : i32}, %arg207: memref<1x64x112x112xf16, "cuda"> {byre.argname = "Output84", byre.argtype = 2 : i32}, %arg208: memref<1x64x56x56xf16, "cuda"> {byre.argname = "Output85", byre.argtype = 2 : i32}, %arg209: memref<64x64x3x3xf16, "cuda"> {byre.argname = "Output86", byre.argtype = 2 : i32}, %arg210: memref<1x64x56x56xf16, "cuda"> {byre.argname = "Output87", byre.argtype = 2 : i32}, %arg211: memref<1x64x56x56xf16, "cuda"> {byre.argname = "Output88", byre.argtype = 2 : i32}, %arg212: memref<64x64x3x3xf16, "cuda"> {byre.argname = "Output89", byre.argtype = 2 : i32}, %arg213: memref<1x64x56x56xf16, "cuda"> {byre.argname = "Output90", byre.argtype = 2 : i32}, %arg214: memref<1x64x56x56xf16, "cuda"> {byre.argname = "Output91", byre.argtype = 2 : i32}, %arg215: memref<64x64x3x3xf16, "cuda"> {byre.argname = "Output92", byre.argtype = 2 : i32}, %arg216: memref<1x64x56x56xf16, "cuda"> {byre.argname = "Output93", byre.argtype = 2 : i32}, %arg217: memref<1x64x56x56xf16, "cuda"> {byre.argname = "Output94", byre.argtype = 2 : i32}, %arg218: memref<64x64x3x3xf16, "cuda"> {byre.argname = "Output95", byre.argtype = 2 : i32}, %arg219: memref<1x64x56x56xf16, "cuda"> {byre.argname = "Output96", byre.argtype = 2 : i32}, %arg220: memref<1x64x56x56xf16, "cuda"> {byre.argname = "Output97", byre.argtype = 2 : i32}, %arg221: memref<128x64x3x3xf16, "cuda"> {byre.argname = "Output98", byre.argtype = 2 : i32}, %arg222: memref<1x128x28x28xf16, "cuda"> {byre.argname = "Output99", byre.argtype = 2 : i32}, %arg223: memref<1x128x28x28xf16, "cuda"> {byre.argname = "Output100", byre.argtype = 2 : i32}, %arg224: memref<128x128x3x3xf16, "cuda"> {byre.argname = "Output101", byre.argtype = 2 : i32}, %arg225: memref<1x128x28x28xf16, "cuda"> {byre.argname = "Output102", byre.argtype = 2 : i32}, %arg226: memref<128x64x1x1xf16, "cuda"> {byre.argname = "Output103", byre.argtype = 2 : i32}, %arg227: memref<1x128x28x28xf16, "cuda"> {byre.argname = "Output104", byre.argtype = 2 : i32}, %arg228: memref<1x128x28x28xf16, "cuda"> {byre.argname = "Output105", byre.argtype = 2 : i32}, %arg229: memref<128x128x3x3xf16, "cuda"> {byre.argname = "Output106", byre.argtype = 2 : i32}, %arg230: memref<1x128x28x28xf16, "cuda"> {byre.argname = "Output107", byre.argtype = 2 : i32}, %arg231: memref<1x128x28x28xf16, "cuda"> {byre.argname = "Output108", byre.argtype = 2 : i32}, %arg232: memref<128x128x3x3xf16, "cuda"> {byre.argname = "Output109", byre.argtype = 2 : i32}, %arg233: memref<1x128x28x28xf16, "cuda"> {byre.argname = "Output110", byre.argtype = 2 : i32}, %arg234: memref<1x128x28x28xf16, "cuda"> {byre.argname = "Output111", byre.argtype = 2 : i32}, %arg235: memref<256x128x3x3xf16, "cuda"> {byre.argname = "Output112", byre.argtype = 2 : i32}, %arg236: memref<1x256x14x14xf16, "cuda"> {byre.argname = "Output113", byre.argtype = 2 : i32}, %arg237: memref<1x256x14x14xf16, "cuda"> {byre.argname = "Output114", byre.argtype = 2 : i32}, %arg238: memref<256x256x3x3xf16, "cuda"> {byre.argname = "Output115", byre.argtype = 2 : i32}, %arg239: memref<1x256x14x14xf16, "cuda"> {byre.argname = "Output116", byre.argtype = 2 : i32}, %arg240: memref<256x128x1x1xf16, "cuda"> {byre.argname = "Output117", byre.argtype = 2 : i32}, %arg241: memref<1x256x14x14xf16, "cuda"> {byre.argname = "Output118", byre.argtype = 2 : i32}, %arg242: memref<1x256x14x14xf16, "cuda"> {byre.argname = "Output119", byre.argtype = 2 : i32}, %arg243: memref<256x256x3x3xf16, "cuda"> {byre.argname = "Output120", byre.argtype = 2 : i32}, %arg244: memref<1x256x14x14xf16, "cuda"> {byre.argname = "Output121", byre.argtype = 2 : i32}, %arg245: memref<1x256x14x14xf16, "cuda"> {byre.argname = "Output122", byre.argtype = 2 : i32}, %arg246: memref<256x256x3x3xf16, "cuda"> {byre.argname = "Output123", byre.argtype = 2 : i32}, %arg247: memref<1x256x14x14xf16, "cuda"> {byre.argname = "Output124", byre.argtype = 2 : i32}, %arg248: memref<1x256x14x14xf16, "cuda"> {byre.argname = "Output125", byre.argtype = 2 : i32}, %arg249: memref<512x256x3x3xf16, "cuda"> {byre.argname = "Output126", byre.argtype = 2 : i32}, %arg250: memref<1x512x7x7xf16, "cuda"> {byre.argname = "Output127", byre.argtype = 2 : i32}, %arg251: memref<1x512x7x7xf16, "cuda"> {byre.argname = "Output128", byre.argtype = 2 : i32}, %arg252: memref<512x512x3x3xf16, "cuda"> {byre.argname = "Output129", byre.argtype = 2 : i32}, %arg253: memref<1x512x7x7xf16, "cuda"> {byre.argname = "Output130", byre.argtype = 2 : i32}, %arg254: memref<512x256x1x1xf16, "cuda"> {byre.argname = "Output131", byre.argtype = 2 : i32}, %arg255: memref<1x512x7x7xf16, "cuda"> {byre.argname = "Output132", byre.argtype = 2 : i32}, %arg256: memref<1x512x7x7xf16, "cuda"> {byre.argname = "Output133", byre.argtype = 2 : i32}, %arg257: memref<512x512x3x3xf16, "cuda"> {byre.argname = "Output134", byre.argtype = 2 : i32}, %arg258: memref<1x512x7x7xf16, "cuda"> {byre.argname = "Output135", byre.argtype = 2 : i32}, %arg259: memref<1x512x7x7xf16, "cuda"> {byre.argname = "Output136", byre.argtype = 2 : i32}, %arg260: memref<512x512x3x3xf16, "cuda"> {byre.argname = "Output137", byre.argtype = 2 : i32}, %arg261: memref<1x512x7x7xf16, "cuda"> {byre.argname = "Output138", byre.argtype = 2 : i32}, %arg262: memref<1x512x7x7xf16, "cuda"> {byre.argname = "Output139", byre.argtype = 2 : i32}, %arg263: memref<1x512xf16, "cuda"> {byre.argname = "Output140", byre.argtype = 2 : i32}, %arg264: memref<512x1000xf16, "cuda"> {byre.argname = "Output141", byre.argtype = 2 : i32}) attributes {byre.entry_point} {
    %alloc = memref.alloc() : memref<1838592xi8, "cuda">
    byre.compute @PTXOp(%arg122, %arg205) {BlockSize.x = 256 : i32, GridSize.x = 147 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown0", memory_effects = [1 : i32, 2 : i32]} : memref<1x3x224x224xf32, "cuda">, memref<1x3x224x224xf16, "cuda">
    byre.compute @PTXOp(%arg2, %arg204) {BlockSize.x = 256 : i32, GridSize.x = 10 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown1", memory_effects = [1 : i32, 2 : i32]} : memref<64x3x7x7xf32, "cuda">, memref<64x3x7x7xf16, "cuda">
    byre.compute @ConvOp_f16f16_f16(%arg205, %arg204, %arg206) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<3> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<1x3x224x224xf16, "cuda">, memref<64x3x7x7xf16, "cuda">, memref<1x64x112x112xf16, "cuda">
    %0 = "byre.alias"(%alloc) <{offset = 224768 : i64}> : (memref<1838592xi8, "cuda">) -> memref<1x64x112x112xf16, "cuda">
    %1 = "byre.alias"(%alloc) <{offset = 7424 : i64}> : (memref<1838592xi8, "cuda">) -> memref<64xf32, "cuda">
    %2 = "byre.alias"(%alloc) <{offset = 7168 : i64}> : (memref<1838592xi8, "cuda">) -> memref<64xf32, "cuda">
    byre.compute @BatchNormTrainingOp_f16f32f32_f16f32f32(%arg206, %arg1, %arg0, %0, %1, %2) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x64x112x112xf16, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<1x64x112x112xf16, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">
    byre.compute @PTXOp(%0, %arg207) {BlockSize.x = 256 : i32, GridSize.x = 784 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown3", memory_effects = [1 : i32, 2 : i32]} : memref<1x64x112x112xf16, "cuda">, memref<1x64x112x112xf16, "cuda">
    byre.compute @PoolMaxOp_f16_f16(%arg207, %arg208) {base_dilations = dense<1> : tensor<4xi64>, device = "cuda", memory_effects = [1 : i32, 2 : i32], padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dilations = dense<1> : tensor<4xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} : memref<1x64x112x112xf16, "cuda">, memref<1x64x56x56xf16, "cuda">
    byre.compute @PTXOp(%arg9, %arg209) {BlockSize.x = 256 : i32, GridSize.x = 36 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown4", memory_effects = [1 : i32, 2 : i32]} : memref<64x64x3x3xf32, "cuda">, memref<64x64x3x3xf16, "cuda">
    byre.compute @ConvOp_f16f16_f16(%arg208, %arg209, %arg210) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x64x56x56xf16, "cuda">, memref<64x64x3x3xf16, "cuda">, memref<1x64x56x56xf16, "cuda">
    %3 = "byre.alias"(%alloc) <{offset = 224768 : i64}> : (memref<1838592xi8, "cuda">) -> memref<1x64x56x56xf16, "cuda">
    %4 = "byre.alias"(%alloc) <{offset = 6912 : i64}> : (memref<1838592xi8, "cuda">) -> memref<64xf32, "cuda">
    %5 = "byre.alias"(%alloc) <{offset = 6656 : i64}> : (memref<1838592xi8, "cuda">) -> memref<64xf32, "cuda">
    byre.compute @BatchNormTrainingOp_f16f32f32_f16f32f32(%arg210, %arg6, %arg5, %3, %4, %5) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x64x56x56xf16, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<1x64x56x56xf16, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">
    byre.compute @PTXOp(%3, %arg211) {BlockSize.x = 256 : i32, GridSize.x = 196 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown6", memory_effects = [1 : i32, 2 : i32]} : memref<1x64x56x56xf16, "cuda">, memref<1x64x56x56xf16, "cuda">
    byre.compute @PTXOp(%arg10, %arg212) {BlockSize.x = 256 : i32, GridSize.x = 36 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown4", memory_effects = [1 : i32, 2 : i32]} : memref<64x64x3x3xf32, "cuda">, memref<64x64x3x3xf16, "cuda">
    byre.compute @ConvOp_f16f16_f16(%arg211, %arg212, %arg213) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x64x56x56xf16, "cuda">, memref<64x64x3x3xf16, "cuda">, memref<1x64x56x56xf16, "cuda">
    %6 = "byre.alias"(%alloc) <{offset = 6400 : i64}> : (memref<1838592xi8, "cuda">) -> memref<64xf32, "cuda">
    %7 = "byre.alias"(%alloc) <{offset = 6144 : i64}> : (memref<1838592xi8, "cuda">) -> memref<64xf32, "cuda">
    byre.compute @BatchNormTrainingOp_f16f32f32_f16f32f32(%arg213, %arg8, %arg7, %3, %6, %7) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x64x56x56xf16, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<1x64x56x56xf16, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">
    byre.compute @PTXOp(%3, %arg208, %arg214) {BlockSize.x = 256 : i32, GridSize.x = 196 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown9", memory_effects = [1 : i32, 1 : i32, 2 : i32]} : memref<1x64x56x56xf16, "cuda">, memref<1x64x56x56xf16, "cuda">, memref<1x64x56x56xf16, "cuda">
    byre.compute @PTXOp(%arg15, %arg215) {BlockSize.x = 256 : i32, GridSize.x = 36 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown4", memory_effects = [1 : i32, 2 : i32]} : memref<64x64x3x3xf32, "cuda">, memref<64x64x3x3xf16, "cuda">
    byre.compute @ConvOp_f16f16_f16(%arg214, %arg215, %arg216) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x64x56x56xf16, "cuda">, memref<64x64x3x3xf16, "cuda">, memref<1x64x56x56xf16, "cuda">
    %8 = "byre.alias"(%alloc) <{offset = 5888 : i64}> : (memref<1838592xi8, "cuda">) -> memref<64xf32, "cuda">
    %9 = "byre.alias"(%alloc) <{offset = 5632 : i64}> : (memref<1838592xi8, "cuda">) -> memref<64xf32, "cuda">
    byre.compute @BatchNormTrainingOp_f16f32f32_f16f32f32(%arg216, %arg12, %arg11, %3, %8, %9) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x64x56x56xf16, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<1x64x56x56xf16, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">
    byre.compute @PTXOp(%3, %arg217) {BlockSize.x = 256 : i32, GridSize.x = 196 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown6", memory_effects = [1 : i32, 2 : i32]} : memref<1x64x56x56xf16, "cuda">, memref<1x64x56x56xf16, "cuda">
    byre.compute @PTXOp(%arg16, %arg218) {BlockSize.x = 256 : i32, GridSize.x = 36 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown4", memory_effects = [1 : i32, 2 : i32]} : memref<64x64x3x3xf32, "cuda">, memref<64x64x3x3xf16, "cuda">
    byre.compute @ConvOp_f16f16_f16(%arg217, %arg218, %arg219) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x64x56x56xf16, "cuda">, memref<64x64x3x3xf16, "cuda">, memref<1x64x56x56xf16, "cuda">
    %10 = "byre.alias"(%alloc) <{offset = 5376 : i64}> : (memref<1838592xi8, "cuda">) -> memref<64xf32, "cuda">
    %11 = "byre.alias"(%alloc) <{offset = 0 : i64}> : (memref<1838592xi8, "cuda">) -> memref<64xf32, "cuda">
    byre.compute @BatchNormTrainingOp_f16f32f32_f16f32f32(%arg219, %arg14, %arg13, %3, %10, %11) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x64x56x56xf16, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<1x64x56x56xf16, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">
    byre.compute @PTXOp(%3, %arg214, %arg220) {BlockSize.x = 256 : i32, GridSize.x = 196 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown9", memory_effects = [1 : i32, 1 : i32, 2 : i32]} : memref<1x64x56x56xf16, "cuda">, memref<1x64x56x56xf16, "cuda">, memref<1x64x56x56xf16, "cuda">
    byre.compute @PTXOp(%arg23, %arg226) {BlockSize.x = 256 : i32, GridSize.x = 8 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown16", memory_effects = [1 : i32, 2 : i32]} : memref<128x64x1x1xf32, "cuda">, memref<128x64x1x1xf16, "cuda">
    byre.compute @ConvOp_f16f16_f16(%arg220, %arg226, %arg227) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<1x64x56x56xf16, "cuda">, memref<128x64x1x1xf16, "cuda">, memref<1x128x28x28xf16, "cuda">
    %12 = "byre.alias"(%alloc) <{offset = 8704 : i64}> : (memref<1838592xi8, "cuda">) -> memref<1x128x28x28xf16, "cuda">
    %13 = "byre.alias"(%alloc) <{offset = 256 : i64}> : (memref<1838592xi8, "cuda">) -> memref<128xf32, "cuda">
    %14 = "byre.alias"(%alloc) <{offset = 768 : i64}> : (memref<1838592xi8, "cuda">) -> memref<128xf32, "cuda">
    byre.compute @BatchNormTrainingOp_f16f32f32_f16f32f32(%arg227, %arg25, %arg24, %12, %13, %14) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<1x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">
    byre.compute @PTXOp(%arg21, %arg221) {BlockSize.x = 256 : i32, GridSize.x = 72 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown18", memory_effects = [1 : i32, 2 : i32]} : memref<128x64x3x3xf32, "cuda">, memref<128x64x3x3xf16, "cuda">
    byre.compute @ConvOp_f16f16_f16(%arg220, %arg221, %arg222) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<1x64x56x56xf16, "cuda">, memref<128x64x3x3xf16, "cuda">, memref<1x128x28x28xf16, "cuda">
    %15 = "byre.alias"(%alloc) <{offset = 224768 : i64}> : (memref<1838592xi8, "cuda">) -> memref<1x128x28x28xf16, "cuda">
    %16 = "byre.alias"(%alloc) <{offset = 4864 : i64}> : (memref<1838592xi8, "cuda">) -> memref<128xf32, "cuda">
    %17 = "byre.alias"(%alloc) <{offset = 1280 : i64}> : (memref<1838592xi8, "cuda">) -> memref<128xf32, "cuda">
    byre.compute @BatchNormTrainingOp_f16f32f32_f16f32f32(%arg222, %arg18, %arg17, %15, %16, %17) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<1x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">
    byre.compute @PTXOp(%15, %arg223) {BlockSize.x = 256 : i32, GridSize.x = 98 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown20", memory_effects = [1 : i32, 2 : i32]} : memref<1x128x28x28xf16, "cuda">, memref<1x128x28x28xf16, "cuda">
    byre.compute @PTXOp(%arg22, %arg224) {BlockSize.x = 256 : i32, GridSize.x = 144 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown21", memory_effects = [1 : i32, 2 : i32]} : memref<128x128x3x3xf32, "cuda">, memref<128x128x3x3xf16, "cuda">
    byre.compute @ConvOp_f16f16_f16(%arg223, %arg224, %arg225) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x128x28x28xf16, "cuda">, memref<128x128x3x3xf16, "cuda">, memref<1x128x28x28xf16, "cuda">
    %18 = "byre.alias"(%alloc) <{offset = 1792 : i64}> : (memref<1838592xi8, "cuda">) -> memref<128xf32, "cuda">
    %19 = "byre.alias"(%alloc) <{offset = 2304 : i64}> : (memref<1838592xi8, "cuda">) -> memref<128xf32, "cuda">
    byre.compute @BatchNormTrainingOp_f16f32f32_f16f32f32(%arg225, %arg20, %arg19, %15, %18, %19) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<1x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">
    byre.compute @PTXOp(%15, %12, %arg228) {BlockSize.x = 256 : i32, GridSize.x = 98 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown23", memory_effects = [1 : i32, 1 : i32, 2 : i32]} : memref<1x128x28x28xf16, "cuda">, memref<1x128x28x28xf16, "cuda">, memref<1x128x28x28xf16, "cuda">
    byre.compute @PTXOp(%arg30, %arg229) {BlockSize.x = 256 : i32, GridSize.x = 144 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown21", memory_effects = [1 : i32, 2 : i32]} : memref<128x128x3x3xf32, "cuda">, memref<128x128x3x3xf16, "cuda">
    byre.compute @ConvOp_f16f16_f16(%arg228, %arg229, %arg230) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x128x28x28xf16, "cuda">, memref<128x128x3x3xf16, "cuda">, memref<1x128x28x28xf16, "cuda">
    %20 = "byre.alias"(%alloc) <{offset = 2816 : i64}> : (memref<1838592xi8, "cuda">) -> memref<128xf32, "cuda">
    %21 = "byre.alias"(%alloc) <{offset = 3328 : i64}> : (memref<1838592xi8, "cuda">) -> memref<128xf32, "cuda">
    byre.compute @BatchNormTrainingOp_f16f32f32_f16f32f32(%arg230, %arg27, %arg26, %15, %20, %21) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<1x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">
    byre.compute @PTXOp(%15, %arg231) {BlockSize.x = 256 : i32, GridSize.x = 98 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown20", memory_effects = [1 : i32, 2 : i32]} : memref<1x128x28x28xf16, "cuda">, memref<1x128x28x28xf16, "cuda">
    byre.compute @PTXOp(%arg31, %arg232) {BlockSize.x = 256 : i32, GridSize.x = 144 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown21", memory_effects = [1 : i32, 2 : i32]} : memref<128x128x3x3xf32, "cuda">, memref<128x128x3x3xf16, "cuda">
    byre.compute @ConvOp_f16f16_f16(%arg231, %arg232, %arg233) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x128x28x28xf16, "cuda">, memref<128x128x3x3xf16, "cuda">, memref<1x128x28x28xf16, "cuda">
    %22 = "byre.alias"(%alloc) <{offset = 3840 : i64}> : (memref<1838592xi8, "cuda">) -> memref<128xf32, "cuda">
    %23 = "byre.alias"(%alloc) <{offset = 4352 : i64}> : (memref<1838592xi8, "cuda">) -> memref<128xf32, "cuda">
    byre.compute @BatchNormTrainingOp_f16f32f32_f16f32f32(%arg233, %arg29, %arg28, %15, %22, %23) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<1x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">
    byre.compute @PTXOp(%15, %arg228, %arg234) {BlockSize.x = 256 : i32, GridSize.x = 98 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown23", memory_effects = [1 : i32, 1 : i32, 2 : i32]} : memref<1x128x28x28xf16, "cuda">, memref<1x128x28x28xf16, "cuda">, memref<1x128x28x28xf16, "cuda">
    byre.compute @PTXOp(%arg38, %arg240) {BlockSize.x = 256 : i32, GridSize.x = 32 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown30", memory_effects = [1 : i32, 2 : i32]} : memref<256x128x1x1xf32, "cuda">, memref<256x128x1x1xf16, "cuda">
    byre.compute @ConvOp_f16f16_f16(%arg234, %arg240, %arg241) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<1x128x28x28xf16, "cuda">, memref<256x128x1x1xf16, "cuda">, memref<1x256x14x14xf16, "cuda">
    %24 = "byre.alias"(%alloc) <{offset = 224768 : i64}> : (memref<1838592xi8, "cuda">) -> memref<1x256x14x14xf16, "cuda">
    %25 = "byre.alias"(%alloc) <{offset = 223744 : i64}> : (memref<1838592xi8, "cuda">) -> memref<256xf32, "cuda">
    %26 = "byre.alias"(%alloc) <{offset = 1836544 : i64}> : (memref<1838592xi8, "cuda">) -> memref<256xf32, "cuda">
    byre.compute @BatchNormTrainingOp_f16f32f32_f16f32f32(%arg241, %arg40, %arg39, %24, %25, %26) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<1x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">
    byre.compute @PTXOp(%arg36, %arg235) {BlockSize.x = 256 : i32, GridSize.x = 288 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown32", memory_effects = [1 : i32, 2 : i32]} : memref<256x128x3x3xf32, "cuda">, memref<256x128x3x3xf16, "cuda">
    byre.compute @ConvOp_f16f16_f16(%arg234, %arg235, %arg236) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<1x128x28x28xf16, "cuda">, memref<256x128x3x3xf16, "cuda">, memref<1x256x14x14xf16, "cuda">
    %27 = "byre.alias"(%alloc) <{offset = 325120 : i64}> : (memref<1838592xi8, "cuda">) -> memref<1x256x14x14xf16, "cuda">
    %28 = "byre.alias"(%alloc) <{offset = 1835520 : i64}> : (memref<1838592xi8, "cuda">) -> memref<256xf32, "cuda">
    %29 = "byre.alias"(%alloc) <{offset = 1834496 : i64}> : (memref<1838592xi8, "cuda">) -> memref<256xf32, "cuda">
    byre.compute @BatchNormTrainingOp_f16f32f32_f16f32f32(%arg236, %arg33, %arg32, %27, %28, %29) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<1x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">
    byre.compute @PTXOp(%27, %arg237) {BlockSize.x = 256 : i32, GridSize.x = 49 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown34", memory_effects = [1 : i32, 2 : i32]} : memref<1x256x14x14xf16, "cuda">, memref<1x256x14x14xf16, "cuda">
    byre.compute @PTXOp(%arg37, %arg238) {BlockSize.x = 256 : i32, GridSize.x = 576 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown35", memory_effects = [1 : i32, 2 : i32]} : memref<256x256x3x3xf32, "cuda">, memref<256x256x3x3xf16, "cuda">
    byre.compute @ConvOp_f16f16_f16(%arg237, %arg238, %arg239) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x256x14x14xf16, "cuda">, memref<256x256x3x3xf16, "cuda">, memref<1x256x14x14xf16, "cuda">
    %30 = "byre.alias"(%alloc) <{offset = 1833472 : i64}> : (memref<1838592xi8, "cuda">) -> memref<256xf32, "cuda">
    %31 = "byre.alias"(%alloc) <{offset = 1837568 : i64}> : (memref<1838592xi8, "cuda">) -> memref<256xf32, "cuda">
    byre.compute @BatchNormTrainingOp_f16f32f32_f16f32f32(%arg239, %arg35, %arg34, %27, %30, %31) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<1x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">
    byre.compute @PTXOp(%27, %24, %arg242) {BlockSize.x = 256 : i32, GridSize.x = 49 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown37", memory_effects = [1 : i32, 1 : i32, 2 : i32]} : memref<1x256x14x14xf16, "cuda">, memref<1x256x14x14xf16, "cuda">, memref<1x256x14x14xf16, "cuda">
    byre.compute @PTXOp(%arg45, %arg243) {BlockSize.x = 256 : i32, GridSize.x = 576 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown35", memory_effects = [1 : i32, 2 : i32]} : memref<256x256x3x3xf32, "cuda">, memref<256x256x3x3xf16, "cuda">
    byre.compute @ConvOp_f16f16_f16(%arg242, %arg243, %arg244) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x256x14x14xf16, "cuda">, memref<256x256x3x3xf16, "cuda">, memref<1x256x14x14xf16, "cuda">
    %32 = "byre.alias"(%alloc) <{offset = 1832448 : i64}> : (memref<1838592xi8, "cuda">) -> memref<256xf32, "cuda">
    %33 = "byre.alias"(%alloc) <{offset = 1831424 : i64}> : (memref<1838592xi8, "cuda">) -> memref<256xf32, "cuda">
    byre.compute @BatchNormTrainingOp_f16f32f32_f16f32f32(%arg244, %arg42, %arg41, %24, %32, %33) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<1x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">
    byre.compute @PTXOp(%24, %arg245) {BlockSize.x = 256 : i32, GridSize.x = 49 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown34", memory_effects = [1 : i32, 2 : i32]} : memref<1x256x14x14xf16, "cuda">, memref<1x256x14x14xf16, "cuda">
    byre.compute @PTXOp(%arg46, %arg246) {BlockSize.x = 256 : i32, GridSize.x = 576 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown35", memory_effects = [1 : i32, 2 : i32]} : memref<256x256x3x3xf32, "cuda">, memref<256x256x3x3xf16, "cuda">
    byre.compute @ConvOp_f16f16_f16(%arg245, %arg246, %arg247) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x256x14x14xf16, "cuda">, memref<256x256x3x3xf16, "cuda">, memref<1x256x14x14xf16, "cuda">
    %34 = "byre.alias"(%alloc) <{offset = 1830400 : i64}> : (memref<1838592xi8, "cuda">) -> memref<256xf32, "cuda">
    %35 = "byre.alias"(%alloc) <{offset = 7680 : i64}> : (memref<1838592xi8, "cuda">) -> memref<256xf32, "cuda">
    byre.compute @BatchNormTrainingOp_f16f32f32_f16f32f32(%arg247, %arg44, %arg43, %24, %34, %35) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<1x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">
    byre.compute @PTXOp(%24, %arg242, %arg248) {BlockSize.x = 256 : i32, GridSize.x = 49 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown37", memory_effects = [1 : i32, 1 : i32, 2 : i32]} : memref<1x256x14x14xf16, "cuda">, memref<1x256x14x14xf16, "cuda">, memref<1x256x14x14xf16, "cuda">
    byre.compute @PTXOp(%arg53, %arg254) {BlockSize.x = 256 : i32, GridSize.x = 128 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown44", memory_effects = [1 : i32, 2 : i32]} : memref<512x256x1x1xf32, "cuda">, memref<512x256x1x1xf16, "cuda">
    byre.compute @ConvOp_f16f16_f16(%arg248, %arg254, %arg255) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<1x256x14x14xf16, "cuda">, memref<512x256x1x1xf16, "cuda">, memref<1x512x7x7xf16, "cuda">
    %36 = "byre.alias"(%alloc) <{offset = 224768 : i64}> : (memref<1838592xi8, "cuda">) -> memref<1x512x7x7xf16, "cuda">
    %37 = "byre.alias"(%alloc) <{offset = 8704 : i64}> : (memref<1838592xi8, "cuda">) -> memref<512xf32, "cuda">
    %38 = "byre.alias"(%alloc) <{offset = 209408 : i64}> : (memref<1838592xi8, "cuda">) -> memref<512xf32, "cuda">
    byre.compute @BatchNormTrainingOp_f16f32f32_f16f32f32(%arg255, %arg55, %arg54, %36, %37, %38) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<1x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">
    byre.compute @PTXOp(%arg51, %arg249) {BlockSize.x = 256 : i32, GridSize.x = 1152 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown46", memory_effects = [1 : i32, 2 : i32]} : memref<512x256x3x3xf32, "cuda">, memref<512x256x3x3xf16, "cuda">
    byre.compute @ConvOp_f16f16_f16(%arg248, %arg249, %arg250) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<1x256x14x14xf16, "cuda">, memref<512x256x3x3xf16, "cuda">, memref<1x512x7x7xf16, "cuda">
    %39 = "byre.alias"(%alloc) <{offset = 274944 : i64}> : (memref<1838592xi8, "cuda">) -> memref<1x512x7x7xf16, "cuda">
    %40 = "byre.alias"(%alloc) <{offset = 12800 : i64}> : (memref<1838592xi8, "cuda">) -> memref<512xf32, "cuda">
    %41 = "byre.alias"(%alloc) <{offset = 10752 : i64}> : (memref<1838592xi8, "cuda">) -> memref<512xf32, "cuda">
    byre.compute @BatchNormTrainingOp_f16f32f32_f16f32f32(%arg250, %arg48, %arg47, %39, %40, %41) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<1x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">
    byre.compute @PTXOp(%39, %arg251) {BlockSize.x = 256 : i32, GridSize.x = 25 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown48", memory_effects = [1 : i32, 2 : i32]} : memref<1x512x7x7xf16, "cuda">, memref<1x512x7x7xf16, "cuda">
    byre.compute @PTXOp(%arg52, %arg252) {BlockSize.x = 256 : i32, GridSize.x = 2304 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown49", memory_effects = [1 : i32, 2 : i32]} : memref<512x512x3x3xf32, "cuda">, memref<512x512x3x3xf16, "cuda">
    byre.compute @ConvOp_f16f16_f16(%arg251, %arg252, %arg253) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x512x7x7xf16, "cuda">, memref<512x512x3x3xf16, "cuda">, memref<1x512x7x7xf16, "cuda">
    %42 = "byre.alias"(%alloc) <{offset = 211456 : i64}> : (memref<1838592xi8, "cuda">) -> memref<512xf32, "cuda">
    %43 = "byre.alias"(%alloc) <{offset = 213504 : i64}> : (memref<1838592xi8, "cuda">) -> memref<512xf32, "cuda">
    byre.compute @BatchNormTrainingOp_f16f32f32_f16f32f32(%arg253, %arg50, %arg49, %39, %42, %43) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<1x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">
    byre.compute @PTXOp(%39, %36, %arg256) {BlockSize.x = 256 : i32, GridSize.x = 25 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown51", memory_effects = [1 : i32, 1 : i32, 2 : i32]} : memref<1x512x7x7xf16, "cuda">, memref<1x512x7x7xf16, "cuda">, memref<1x512x7x7xf16, "cuda">
    byre.compute @PTXOp(%arg60, %arg257) {BlockSize.x = 256 : i32, GridSize.x = 2304 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown49", memory_effects = [1 : i32, 2 : i32]} : memref<512x512x3x3xf32, "cuda">, memref<512x512x3x3xf16, "cuda">
    byre.compute @ConvOp_f16f16_f16(%arg256, %arg257, %arg258) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x512x7x7xf16, "cuda">, memref<512x512x3x3xf16, "cuda">, memref<1x512x7x7xf16, "cuda">
    %44 = "byre.alias"(%alloc) <{offset = 215552 : i64}> : (memref<1838592xi8, "cuda">) -> memref<512xf32, "cuda">
    %45 = "byre.alias"(%alloc) <{offset = 217600 : i64}> : (memref<1838592xi8, "cuda">) -> memref<512xf32, "cuda">
    byre.compute @BatchNormTrainingOp_f16f32f32_f16f32f32(%arg258, %arg57, %arg56, %36, %44, %45) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<1x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">
    byre.compute @PTXOp(%36, %arg259) {BlockSize.x = 256 : i32, GridSize.x = 25 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown48", memory_effects = [1 : i32, 2 : i32]} : memref<1x512x7x7xf16, "cuda">, memref<1x512x7x7xf16, "cuda">
    byre.compute @PTXOp(%arg61, %arg260) {BlockSize.x = 256 : i32, GridSize.x = 2304 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown49", memory_effects = [1 : i32, 2 : i32]} : memref<512x512x3x3xf32, "cuda">, memref<512x512x3x3xf16, "cuda">
    byre.compute @ConvOp_f16f16_f16(%arg259, %arg260, %arg261) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x512x7x7xf16, "cuda">, memref<512x512x3x3xf16, "cuda">, memref<1x512x7x7xf16, "cuda">
    %46 = "byre.alias"(%alloc) <{offset = 219648 : i64}> : (memref<1838592xi8, "cuda">) -> memref<512xf32, "cuda">
    %47 = "byre.alias"(%alloc) <{offset = 221696 : i64}> : (memref<1838592xi8, "cuda">) -> memref<512xf32, "cuda">
    byre.compute @BatchNormTrainingOp_f16f32f32_f16f32f32(%arg261, %arg59, %arg58, %36, %46, %47) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<1x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">
    byre.compute @PTXOp(%36, %arg256, %arg262) {BlockSize.x = 256 : i32, GridSize.x = 25 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown51", memory_effects = [1 : i32, 1 : i32, 2 : i32]} : memref<1x512x7x7xf16, "cuda">, memref<1x512x7x7xf16, "cuda">, memref<1x512x7x7xf16, "cuda">
    %48 = "byre.alias"(%arg262) <{offset = 0 : i64}> : (memref<1x512x7x7xf16, "cuda">) -> memref<512x49xf16, "cuda">
    %49 = "byre.alias"(%alloc) <{offset = 224768 : i64}> : (memref<1838592xi8, "cuda">) -> memref<512xf16, "cuda">
    byre.compute @PTXOp(%48, %49) {BlockSize.x = 64 : i32, BlockSize.y = 1 : i32, BlockSize.z = 1 : i32, GridSize.x = 512 : i32, GridSize.y = 1 : i32, GridSize.z = 1 : i32, device = "cuda", kernel_name = "Unknown58_kernel"} : memref<512x49xf16, "cuda">, memref<512xf16, "cuda">
    %50 = "byre.alias"(%alloc) <{offset = 224768 : i64}> : (memref<1838592xi8, "cuda">) -> memref<1x512xf16, "cuda">
    byre.compute @PTXOp(%50, %arg263) {BlockSize.x = 256 : i32, GridSize.x = 1 : i32, arg_ranks = [2 : i32, 2 : i32], kernel_name = "Unknown59", memory_effects = [1 : i32, 2 : i32]} : memref<1x512xf16, "cuda">, memref<1x512xf16, "cuda">
    %51 = "byre.alias"(%alloc) <{offset = 224768 : i64}> : (memref<1838592xi8, "cuda">) -> memref<1000x512xf16, "cuda">
    byre.compute @PTXOp(%arg4, %51) {BlockSize.x = 256 : i32, GridSize.x = 500 : i32, arg_ranks = [2 : i32, 2 : i32], kernel_name = "Unknown60", memory_effects = [1 : i32, 2 : i32]} : memref<1000x512xf32, "cuda">, memref<1000x512xf16, "cuda">
    byre.compute @TransposeOp_f16_f16(%51, %arg264) {device = "cuda", memory_effects = [1 : i32, 2 : i32], minor_to_major = dense<[0, 1]> : tensor<2xindex>, permutation = dense<[1, 0]> : tensor<2xi64>} : memref<1000x512xf16, "cuda">, memref<512x1000xf16, "cuda">
    %52 = "byre.alias"(%alloc) <{offset = 14848 : i64}> : (memref<1838592xi8, "cuda">) -> memref<1x1000xf16, "cuda">
    byre.compute @MatmulOp_f16f16_f16(%arg263, %51, %52) {device = "cuda", lhs_contracting_dimension = 1 : i64, memory_effects = [1 : i32, 1 : i32, 2 : i32], rhs_contracting_dimension = 1 : i64} : memref<1x512xf16, "cuda">, memref<1000x512xf16, "cuda">, memref<1x1000xf16, "cuda">
    byre.compute @PTXOp(%arg3, %52, %arg123) {BlockSize.x = 256 : i32, GridSize.x = 1 : i32, arg_ranks = [1 : i32, 2 : i32, 2 : i32], kernel_name = "Unknown61", memory_effects = [1 : i32, 1 : i32, 2 : i32]} : memref<1000xf32, "cuda">, memref<1x1000xf16, "cuda">, memref<1x1000xf16, "cuda">
    byre.compute @PTXOp(%1, %arg63, %arg164) {BlockSize.x = 256 : i32, GridSize.x = 1 : i32, arg_ranks = [1 : i32, 1 : i32, 1 : i32], kernel_name = "Unknown62", memory_effects = [1 : i32, 1 : i32, 2 : i32]} : memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">
    byre.compute @PTXOp(%2, %arg64, %arg165) {BlockSize.x = 256 : i32, GridSize.x = 1 : i32, arg_ranks = [1 : i32, 1 : i32, 1 : i32], kernel_name = "Unknown62", memory_effects = [1 : i32, 1 : i32, 2 : i32]} : memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">
    byre.compute @PTXOp(%4, %arg66, %arg166) {BlockSize.x = 256 : i32, GridSize.x = 1 : i32, arg_ranks = [1 : i32, 1 : i32, 1 : i32], kernel_name = "Unknown62", memory_effects = [1 : i32, 1 : i32, 2 : i32]} : memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">
    byre.compute @PTXOp(%5, %arg67, %arg167) {BlockSize.x = 256 : i32, GridSize.x = 1 : i32, arg_ranks = [1 : i32, 1 : i32, 1 : i32], kernel_name = "Unknown62", memory_effects = [1 : i32, 1 : i32, 2 : i32]} : memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">
    byre.compute @PTXOp(%6, %arg69, %arg168) {BlockSize.x = 256 : i32, GridSize.x = 1 : i32, arg_ranks = [1 : i32, 1 : i32, 1 : i32], kernel_name = "Unknown62", memory_effects = [1 : i32, 1 : i32, 2 : i32]} : memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">
    byre.compute @PTXOp(%7, %arg70, %arg169) {BlockSize.x = 256 : i32, GridSize.x = 1 : i32, arg_ranks = [1 : i32, 1 : i32, 1 : i32], kernel_name = "Unknown62", memory_effects = [1 : i32, 1 : i32, 2 : i32]} : memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">
    byre.compute @PTXOp(%8, %arg72, %arg170) {BlockSize.x = 256 : i32, GridSize.x = 1 : i32, arg_ranks = [1 : i32, 1 : i32, 1 : i32], kernel_name = "Unknown62", memory_effects = [1 : i32, 1 : i32, 2 : i32]} : memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">
    byre.compute @PTXOp(%9, %arg73, %arg171) {BlockSize.x = 256 : i32, GridSize.x = 1 : i32, arg_ranks = [1 : i32, 1 : i32, 1 : i32], kernel_name = "Unknown62", memory_effects = [1 : i32, 1 : i32, 2 : i32]} : memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">
    byre.compute @PTXOp(%10, %arg75, %arg172) {BlockSize.x = 256 : i32, GridSize.x = 1 : i32, arg_ranks = [1 : i32, 1 : i32, 1 : i32], kernel_name = "Unknown62", memory_effects = [1 : i32, 1 : i32, 2 : i32]} : memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">
    byre.compute @PTXOp(%11, %arg76, %arg173) {BlockSize.x = 256 : i32, GridSize.x = 1 : i32, arg_ranks = [1 : i32, 1 : i32, 1 : i32], kernel_name = "Unknown62", memory_effects = [1 : i32, 1 : i32, 2 : i32]} : memref<64xf32, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">
    byre.compute @PTXOp(%16, %arg78, %arg174) {BlockSize.x = 256 : i32, GridSize.x = 1 : i32, arg_ranks = [1 : i32, 1 : i32, 1 : i32], kernel_name = "Unknown72", memory_effects = [1 : i32, 1 : i32, 2 : i32]} : memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">
    byre.compute @PTXOp(%17, %arg79, %arg175) {BlockSize.x = 256 : i32, GridSize.x = 1 : i32, arg_ranks = [1 : i32, 1 : i32, 1 : i32], kernel_name = "Unknown72", memory_effects = [1 : i32, 1 : i32, 2 : i32]} : memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">
    byre.compute @PTXOp(%18, %arg81, %arg176) {BlockSize.x = 256 : i32, GridSize.x = 1 : i32, arg_ranks = [1 : i32, 1 : i32, 1 : i32], kernel_name = "Unknown72", memory_effects = [1 : i32, 1 : i32, 2 : i32]} : memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">
    byre.compute @PTXOp(%19, %arg82, %arg177) {BlockSize.x = 256 : i32, GridSize.x = 1 : i32, arg_ranks = [1 : i32, 1 : i32, 1 : i32], kernel_name = "Unknown72", memory_effects = [1 : i32, 1 : i32, 2 : i32]} : memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">
    byre.compute @PTXOp(%13, %arg84, %arg178) {BlockSize.x = 256 : i32, GridSize.x = 1 : i32, arg_ranks = [1 : i32, 1 : i32, 1 : i32], kernel_name = "Unknown72", memory_effects = [1 : i32, 1 : i32, 2 : i32]} : memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">
    byre.compute @PTXOp(%14, %arg85, %arg179) {BlockSize.x = 256 : i32, GridSize.x = 1 : i32, arg_ranks = [1 : i32, 1 : i32, 1 : i32], kernel_name = "Unknown72", memory_effects = [1 : i32, 1 : i32, 2 : i32]} : memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">
    byre.compute @PTXOp(%20, %arg87, %arg180) {BlockSize.x = 256 : i32, GridSize.x = 1 : i32, arg_ranks = [1 : i32, 1 : i32, 1 : i32], kernel_name = "Unknown72", memory_effects = [1 : i32, 1 : i32, 2 : i32]} : memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">
    byre.compute @PTXOp(%21, %arg88, %arg181) {BlockSize.x = 256 : i32, GridSize.x = 1 : i32, arg_ranks = [1 : i32, 1 : i32, 1 : i32], kernel_name = "Unknown72", memory_effects = [1 : i32, 1 : i32, 2 : i32]} : memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">
    byre.compute @PTXOp(%22, %arg90, %arg182) {BlockSize.x = 256 : i32, GridSize.x = 1 : i32, arg_ranks = [1 : i32, 1 : i32, 1 : i32], kernel_name = "Unknown72", memory_effects = [1 : i32, 1 : i32, 2 : i32]} : memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">
    byre.compute @PTXOp(%23, %arg91, %arg183) {BlockSize.x = 256 : i32, GridSize.x = 1 : i32, arg_ranks = [1 : i32, 1 : i32, 1 : i32], kernel_name = "Unknown72", memory_effects = [1 : i32, 1 : i32, 2 : i32]} : memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">
    byre.compute @PTXOp(%28, %arg93, %arg184) {BlockSize.x = 256 : i32, GridSize.x = 1 : i32, arg_ranks = [1 : i32, 1 : i32, 1 : i32], kernel_name = "Unknown82", memory_effects = [1 : i32, 1 : i32, 2 : i32]} : memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">
    byre.compute @PTXOp(%29, %arg94, %arg185) {BlockSize.x = 256 : i32, GridSize.x = 1 : i32, arg_ranks = [1 : i32, 1 : i32, 1 : i32], kernel_name = "Unknown82", memory_effects = [1 : i32, 1 : i32, 2 : i32]} : memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">
    byre.compute @PTXOp(%30, %arg96, %arg186) {BlockSize.x = 256 : i32, GridSize.x = 1 : i32, arg_ranks = [1 : i32, 1 : i32, 1 : i32], kernel_name = "Unknown82", memory_effects = [1 : i32, 1 : i32, 2 : i32]} : memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">
    byre.compute @PTXOp(%31, %arg97, %arg187) {BlockSize.x = 256 : i32, GridSize.x = 1 : i32, arg_ranks = [1 : i32, 1 : i32, 1 : i32], kernel_name = "Unknown82", memory_effects = [1 : i32, 1 : i32, 2 : i32]} : memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">
    byre.compute @PTXOp(%25, %arg99, %arg188) {BlockSize.x = 256 : i32, GridSize.x = 1 : i32, arg_ranks = [1 : i32, 1 : i32, 1 : i32], kernel_name = "Unknown82", memory_effects = [1 : i32, 1 : i32, 2 : i32]} : memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">
    byre.compute @PTXOp(%26, %arg100, %arg189) {BlockSize.x = 256 : i32, GridSize.x = 1 : i32, arg_ranks = [1 : i32, 1 : i32, 1 : i32], kernel_name = "Unknown82", memory_effects = [1 : i32, 1 : i32, 2 : i32]} : memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">
    byre.compute @PTXOp(%32, %arg102, %arg190) {BlockSize.x = 256 : i32, GridSize.x = 1 : i32, arg_ranks = [1 : i32, 1 : i32, 1 : i32], kernel_name = "Unknown82", memory_effects = [1 : i32, 1 : i32, 2 : i32]} : memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">
    byre.compute @PTXOp(%33, %arg103, %arg191) {BlockSize.x = 256 : i32, GridSize.x = 1 : i32, arg_ranks = [1 : i32, 1 : i32, 1 : i32], kernel_name = "Unknown82", memory_effects = [1 : i32, 1 : i32, 2 : i32]} : memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">
    byre.compute @PTXOp(%34, %arg105, %arg192) {BlockSize.x = 256 : i32, GridSize.x = 1 : i32, arg_ranks = [1 : i32, 1 : i32, 1 : i32], kernel_name = "Unknown82", memory_effects = [1 : i32, 1 : i32, 2 : i32]} : memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">
    byre.compute @PTXOp(%35, %arg106, %arg193) {BlockSize.x = 256 : i32, GridSize.x = 1 : i32, arg_ranks = [1 : i32, 1 : i32, 1 : i32], kernel_name = "Unknown82", memory_effects = [1 : i32, 1 : i32, 2 : i32]} : memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">
    byre.compute @PTXOp(%40, %arg108, %arg194) {BlockSize.x = 256 : i32, GridSize.x = 1 : i32, arg_ranks = [1 : i32, 1 : i32, 1 : i32], kernel_name = "Unknown92", memory_effects = [1 : i32, 1 : i32, 2 : i32]} : memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">
    byre.compute @PTXOp(%41, %arg109, %arg195) {BlockSize.x = 256 : i32, GridSize.x = 1 : i32, arg_ranks = [1 : i32, 1 : i32, 1 : i32], kernel_name = "Unknown92", memory_effects = [1 : i32, 1 : i32, 2 : i32]} : memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">
    byre.compute @PTXOp(%42, %arg111, %arg196) {BlockSize.x = 256 : i32, GridSize.x = 1 : i32, arg_ranks = [1 : i32, 1 : i32, 1 : i32], kernel_name = "Unknown92", memory_effects = [1 : i32, 1 : i32, 2 : i32]} : memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">
    byre.compute @PTXOp(%43, %arg112, %arg197) {BlockSize.x = 256 : i32, GridSize.x = 1 : i32, arg_ranks = [1 : i32, 1 : i32, 1 : i32], kernel_name = "Unknown92", memory_effects = [1 : i32, 1 : i32, 2 : i32]} : memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">
    byre.compute @PTXOp(%37, %arg114, %arg198) {BlockSize.x = 256 : i32, GridSize.x = 1 : i32, arg_ranks = [1 : i32, 1 : i32, 1 : i32], kernel_name = "Unknown92", memory_effects = [1 : i32, 1 : i32, 2 : i32]} : memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">
    byre.compute @PTXOp(%38, %arg115, %arg199) {BlockSize.x = 256 : i32, GridSize.x = 1 : i32, arg_ranks = [1 : i32, 1 : i32, 1 : i32], kernel_name = "Unknown92", memory_effects = [1 : i32, 1 : i32, 2 : i32]} : memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">
    byre.compute @PTXOp(%44, %arg117, %arg200) {BlockSize.x = 256 : i32, GridSize.x = 1 : i32, arg_ranks = [1 : i32, 1 : i32, 1 : i32], kernel_name = "Unknown92", memory_effects = [1 : i32, 1 : i32, 2 : i32]} : memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">
    byre.compute @PTXOp(%45, %arg118, %arg201) {BlockSize.x = 256 : i32, GridSize.x = 1 : i32, arg_ranks = [1 : i32, 1 : i32, 1 : i32], kernel_name = "Unknown92", memory_effects = [1 : i32, 1 : i32, 2 : i32]} : memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">
    byre.compute @PTXOp(%46, %arg120, %arg202) {BlockSize.x = 256 : i32, GridSize.x = 1 : i32, arg_ranks = [1 : i32, 1 : i32, 1 : i32], kernel_name = "Unknown92", memory_effects = [1 : i32, 1 : i32, 2 : i32]} : memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">
    byre.compute @PTXOp(%47, %arg121, %arg203) {BlockSize.x = 256 : i32, GridSize.x = 1 : i32, arg_ranks = [1 : i32, 1 : i32, 1 : i32], kernel_name = "Unknown92", memory_effects = [1 : i32, 1 : i32, 2 : i32]} : memref<512xf32, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">
    byre.copy(%arg0, %arg124) {callee = "cuda2cuda"} : memref<64xf32, "cuda">, memref<64xf32, "cuda">
    byre.copy(%arg1, %arg125) {callee = "cuda2cuda"} : memref<64xf32, "cuda">, memref<64xf32, "cuda">
    byre.copy(%arg5, %arg126) {callee = "cuda2cuda"} : memref<64xf32, "cuda">, memref<64xf32, "cuda">
    byre.copy(%arg6, %arg127) {callee = "cuda2cuda"} : memref<64xf32, "cuda">, memref<64xf32, "cuda">
    byre.copy(%arg7, %arg128) {callee = "cuda2cuda"} : memref<64xf32, "cuda">, memref<64xf32, "cuda">
    byre.copy(%arg8, %arg129) {callee = "cuda2cuda"} : memref<64xf32, "cuda">, memref<64xf32, "cuda">
    byre.copy(%arg11, %arg130) {callee = "cuda2cuda"} : memref<64xf32, "cuda">, memref<64xf32, "cuda">
    byre.copy(%arg12, %arg131) {callee = "cuda2cuda"} : memref<64xf32, "cuda">, memref<64xf32, "cuda">
    byre.copy(%arg13, %arg132) {callee = "cuda2cuda"} : memref<64xf32, "cuda">, memref<64xf32, "cuda">
    byre.copy(%arg14, %arg133) {callee = "cuda2cuda"} : memref<64xf32, "cuda">, memref<64xf32, "cuda">
    byre.copy(%arg17, %arg134) {callee = "cuda2cuda"} : memref<128xf32, "cuda">, memref<128xf32, "cuda">
    byre.copy(%arg18, %arg135) {callee = "cuda2cuda"} : memref<128xf32, "cuda">, memref<128xf32, "cuda">
    byre.copy(%arg19, %arg136) {callee = "cuda2cuda"} : memref<128xf32, "cuda">, memref<128xf32, "cuda">
    byre.copy(%arg20, %arg137) {callee = "cuda2cuda"} : memref<128xf32, "cuda">, memref<128xf32, "cuda">
    byre.copy(%arg24, %arg138) {callee = "cuda2cuda"} : memref<128xf32, "cuda">, memref<128xf32, "cuda">
    byre.copy(%arg25, %arg139) {callee = "cuda2cuda"} : memref<128xf32, "cuda">, memref<128xf32, "cuda">
    byre.copy(%arg26, %arg140) {callee = "cuda2cuda"} : memref<128xf32, "cuda">, memref<128xf32, "cuda">
    byre.copy(%arg27, %arg141) {callee = "cuda2cuda"} : memref<128xf32, "cuda">, memref<128xf32, "cuda">
    byre.copy(%arg28, %arg142) {callee = "cuda2cuda"} : memref<128xf32, "cuda">, memref<128xf32, "cuda">
    byre.copy(%arg29, %arg143) {callee = "cuda2cuda"} : memref<128xf32, "cuda">, memref<128xf32, "cuda">
    byre.copy(%arg32, %arg144) {callee = "cuda2cuda"} : memref<256xf32, "cuda">, memref<256xf32, "cuda">
    byre.copy(%arg33, %arg145) {callee = "cuda2cuda"} : memref<256xf32, "cuda">, memref<256xf32, "cuda">
    byre.copy(%arg34, %arg146) {callee = "cuda2cuda"} : memref<256xf32, "cuda">, memref<256xf32, "cuda">
    byre.copy(%arg35, %arg147) {callee = "cuda2cuda"} : memref<256xf32, "cuda">, memref<256xf32, "cuda">
    byre.copy(%arg39, %arg148) {callee = "cuda2cuda"} : memref<256xf32, "cuda">, memref<256xf32, "cuda">
    byre.copy(%arg40, %arg149) {callee = "cuda2cuda"} : memref<256xf32, "cuda">, memref<256xf32, "cuda">
    byre.copy(%arg41, %arg150) {callee = "cuda2cuda"} : memref<256xf32, "cuda">, memref<256xf32, "cuda">
    byre.copy(%arg42, %arg151) {callee = "cuda2cuda"} : memref<256xf32, "cuda">, memref<256xf32, "cuda">
    byre.copy(%arg43, %arg152) {callee = "cuda2cuda"} : memref<256xf32, "cuda">, memref<256xf32, "cuda">
    byre.copy(%arg44, %arg153) {callee = "cuda2cuda"} : memref<256xf32, "cuda">, memref<256xf32, "cuda">
    byre.copy(%arg47, %arg154) {callee = "cuda2cuda"} : memref<512xf32, "cuda">, memref<512xf32, "cuda">
    byre.copy(%arg48, %arg155) {callee = "cuda2cuda"} : memref<512xf32, "cuda">, memref<512xf32, "cuda">
    byre.copy(%arg49, %arg156) {callee = "cuda2cuda"} : memref<512xf32, "cuda">, memref<512xf32, "cuda">
    byre.copy(%arg50, %arg157) {callee = "cuda2cuda"} : memref<512xf32, "cuda">, memref<512xf32, "cuda">
    byre.copy(%arg54, %arg158) {callee = "cuda2cuda"} : memref<512xf32, "cuda">, memref<512xf32, "cuda">
    byre.copy(%arg55, %arg159) {callee = "cuda2cuda"} : memref<512xf32, "cuda">, memref<512xf32, "cuda">
    byre.copy(%arg56, %arg160) {callee = "cuda2cuda"} : memref<512xf32, "cuda">, memref<512xf32, "cuda">
    byre.copy(%arg57, %arg161) {callee = "cuda2cuda"} : memref<512xf32, "cuda">, memref<512xf32, "cuda">
    byre.copy(%arg58, %arg162) {callee = "cuda2cuda"} : memref<512xf32, "cuda">, memref<512xf32, "cuda">
    byre.copy(%arg59, %arg163) {callee = "cuda2cuda"} : memref<512xf32, "cuda">, memref<512xf32, "cuda">
    return
  }
}