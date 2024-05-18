// RUN: byteir-opt %s -nvvm-codegen | FileCheck %s

// CHECK-LABEL: gpu.module @unified

module attributes {byre.container_module, gpu.container_module} {
  gpu.module @unified {
    gpu.func @Unknown96(%arg0: memref<512x256x1x1xf16>, %arg1: memref<512x256x1x1xf32>) kernel {
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
        %9 = memref.load %arg0[%8, %7, %c0, %c0] : memref<512x256x1x1xf16>
        %10 = arith.extf %9 : f16 to f32
        memref.store %10, %arg1[%8, %7, %c0, %c0] : memref<512x256x1x1xf32>
      }
      gpu.return
    }
    gpu.func @Unknown95(%arg0: memref<512x512x3x3xf16>, %arg1: memref<512x512x3x3xf32>) kernel {
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
        %13 = memref.load %arg0[%12, %11, %9, %7] : memref<512x512x3x3xf16>
        %14 = arith.extf %13 : f16 to f32
        memref.store %14, %arg1[%12, %11, %9, %7] : memref<512x512x3x3xf32>
      }
      gpu.return
    }
    gpu.func @Unknown94(%arg0: memref<512x256x3x3xf16>, %arg1: memref<512x256x3x3xf32>) kernel {
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
        %13 = memref.load %arg0[%12, %11, %9, %7] : memref<512x256x3x3xf16>
        %14 = arith.extf %13 : f16 to f32
        memref.store %14, %arg1[%12, %11, %9, %7] : memref<512x256x3x3xf32>
      }
      gpu.return
    }
    gpu.func @Unknown91(%arg0: memref<256x128x1x1xf16>, %arg1: memref<256x128x1x1xf32>) kernel {
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
        %9 = memref.load %arg0[%8, %7, %c0, %c0] : memref<256x128x1x1xf16>
        %10 = arith.extf %9 : f16 to f32
        memref.store %10, %arg1[%8, %7, %c0, %c0] : memref<256x128x1x1xf32>
      }
      gpu.return
    }
    gpu.func @Unknown90(%arg0: memref<256x256x3x3xf16>, %arg1: memref<256x256x3x3xf32>) kernel {
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
        %13 = memref.load %arg0[%12, %11, %9, %7] : memref<256x256x3x3xf16>
        %14 = arith.extf %13 : f16 to f32
        memref.store %14, %arg1[%12, %11, %9, %7] : memref<256x256x3x3xf32>
      }
      gpu.return
    }
    gpu.func @Unknown89(%arg0: memref<256x128x3x3xf16>, %arg1: memref<256x128x3x3xf32>) kernel {
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
        %13 = memref.load %arg0[%12, %11, %9, %7] : memref<256x128x3x3xf16>
        %14 = arith.extf %13 : f16 to f32
        memref.store %14, %arg1[%12, %11, %9, %7] : memref<256x128x3x3xf32>
      }
      gpu.return
    }
    gpu.func @Unknown86(%arg0: memref<128x64x1x1xf16>, %arg1: memref<128x64x1x1xf32>) kernel {
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
        %9 = memref.load %arg0[%8, %7, %c0, %c0] : memref<128x64x1x1xf16>
        %10 = arith.extf %9 : f16 to f32
        memref.store %10, %arg1[%8, %7, %c0, %c0] : memref<128x64x1x1xf32>
      }
      gpu.return
    }
    gpu.func @Unknown85(%arg0: memref<128x128x3x3xf16>, %arg1: memref<128x128x3x3xf32>) kernel {
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
        %13 = memref.load %arg0[%12, %11, %9, %7] : memref<128x128x3x3xf16>
        %14 = arith.extf %13 : f16 to f32
        memref.store %14, %arg1[%12, %11, %9, %7] : memref<128x128x3x3xf32>
      }
      gpu.return
    }
    gpu.func @Unknown84(%arg0: memref<128x64x3x3xf16>, %arg1: memref<128x64x3x3xf32>) kernel {
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
        %13 = memref.load %arg0[%12, %11, %9, %7] : memref<128x64x3x3xf16>
        %14 = arith.extf %13 : f16 to f32
        memref.store %14, %arg1[%12, %11, %9, %7] : memref<128x64x3x3xf32>
      }
      gpu.return
    }
    gpu.func @Unknown80(%arg0: memref<64x64x3x3xf16>, %arg1: memref<64x64x3x3xf32>) kernel {
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
        %13 = memref.load %arg0[%12, %11, %9, %7] : memref<64x64x3x3xf16>
        %14 = arith.extf %13 : f16 to f32
        memref.store %14, %arg1[%12, %11, %9, %7] : memref<64x64x3x3xf32>
      }
      gpu.return
    }
    gpu.func @Unknown79(%arg0: memref<1000x512xf16>, %arg1: memref<1000x512xf32>) kernel {
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
        %9 = memref.load %arg0[%8, %7] : memref<1000x512xf16>
        %10 = arith.extf %9 : f16 to f32
        memref.store %10, %arg1[%8, %7] : memref<1000x512xf32>
      }
      gpu.return
    }
    gpu.func @Unknown78(%arg0: memref<1x1000xf16>, %arg1: memref<1x1000xf32>) kernel {
      %c0 = arith.constant 0 : index
      %c1000 = arith.constant 1000 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = gpu.grid_dim  x
      %6 = arith.muli %1, %5 : index
      scf.for %arg2 = %4 to %c1000 step %6 {
        %7 = memref.load %arg0[%c0, %arg2] : memref<1x1000xf16>
        %8 = arith.extf %7 : f16 to f32
        %9 = arith.truncf %8 : f32 to f16
        %10 = arith.extf %9 : f16 to f32
        memref.store %10, %arg1[%c0, %arg2] : memref<1x1000xf32>
      }
      gpu.return
    }
    gpu.func @Unknown77(%arg0: memref<64x3x7x7xf16>, %arg1: memref<64x3x7x7xf32>) kernel {
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
        %13 = memref.load %arg0[%12, %11, %9, %7] : memref<64x3x7x7xf16>
        %14 = arith.extf %13 : f16 to f32
        memref.store %14, %arg1[%12, %11, %9, %7] : memref<64x3x7x7xf32>
      }
      gpu.return
    }
    gpu.func @Unknown74(%arg0: memref<1x64x112x112xf16>, %arg1: memref<1x64x112x112xf16>, %arg2: memref<1x64x112x112xf16>) kernel {
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
      scf.for %arg3 = %4 to %c802816 step %6 {
        %7 = arith.remsi %arg3, %c112 : index
        %8 = arith.divsi %arg3, %c112 : index
        %9 = arith.remsi %8, %c112 : index
        %10 = arith.divsi %8, %c112 : index
        %11 = memref.load %arg0[%c0, %10, %9, %7] : memref<1x64x112x112xf16>
        %12 = memref.load %arg1[%c0, %10, %9, %7] : memref<1x64x112x112xf16>
        %13 = arith.cmpf ogt, %11, %cst : f16
        %14 = arith.select %13, %12, %cst : f16
        memref.store %14, %arg2[%c0, %10, %9, %7] : memref<1x64x112x112xf16>
      }
      gpu.return
    }
    gpu.func @Unknown73(%arg0: memref<1x64x56x56xf16>, %arg1: memref<1x64x56x56xf16>, %arg2: memref<1x64x56x56xf16>) kernel {
      %c200704 = arith.constant 200704 : index
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
        memref.store %13, %arg2[%c0, %10, %9, %7] : memref<1x64x56x56xf16>
      }
      gpu.return
    }
    gpu.func @Unknown61(%arg0: memref<1x64x56x56xf16>, %arg1: memref<1x64x56x56xf16>, %arg2: memref<1x64x56x56xf16>) kernel {
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
        %13 = arith.cmpf ogt, %11, %cst : f16
        %14 = arith.select %13, %12, %cst : f16
        memref.store %14, %arg2[%c0, %10, %9, %7] : memref<1x64x56x56xf16>
      }
      gpu.return
    }
    gpu.func @Unknown57(%arg0: memref<1x64x56x56xf16>, %arg1: memref<1x64x56x56xf16>, %arg2: memref<1x64x56x56xf16>, %arg3: memref<1x64x56x56xf16>) kernel {
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
      scf.for %arg4 = %4 to %c200704 step %6 {
        %7 = arith.remsi %arg4, %c56 : index
        %8 = arith.divsi %arg4, %c56 : index
        %9 = arith.remsi %8, %c56 : index
        %10 = arith.divsi %8, %c56 : index
        %11 = memref.load %arg0[%c0, %10, %9, %7] : memref<1x64x56x56xf16>
        %12 = memref.load %arg1[%c0, %10, %9, %7] : memref<1x64x56x56xf16>
        %13 = memref.load %arg2[%c0, %10, %9, %7] : memref<1x64x56x56xf16>
        %14 = arith.addf %11, %12 : f16
        %15 = arith.cmpf ogt, %13, %cst : f16
        %16 = arith.select %15, %14, %cst : f16
        memref.store %16, %arg3[%c0, %10, %9, %7] : memref<1x64x56x56xf16>
      }
      gpu.return
    }
    gpu.func @Unknown42(%arg0: memref<1x128x28x28xf16>, %arg1: memref<1x128x28x28xf16>, %arg2: memref<1x128x28x28xf16>) kernel {
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
        %13 = arith.cmpf ogt, %11, %cst : f16
        %14 = arith.select %13, %12, %cst : f16
        memref.store %14, %arg2[%c0, %10, %9, %7] : memref<1x128x28x28xf16>
      }
      gpu.return
    }
    gpu.func @Unknown38(%arg0: memref<1x128x28x28xf16>, %arg1: memref<1x128x28x28xf16>, %arg2: memref<1x128x28x28xf16>, %arg3: memref<1x128x28x28xf16>) kernel {
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
      scf.for %arg4 = %4 to %c100352 step %6 {
        %7 = arith.remsi %arg4, %c28 : index
        %8 = arith.divsi %arg4, %c28 : index
        %9 = arith.remsi %8, %c28 : index
        %10 = arith.divsi %8, %c28 : index
        %11 = memref.load %arg0[%c0, %10, %9, %7] : memref<1x128x28x28xf16>
        %12 = memref.load %arg1[%c0, %10, %9, %7] : memref<1x128x28x28xf16>
        %13 = memref.load %arg2[%c0, %10, %9, %7] : memref<1x128x28x28xf16>
        %14 = arith.addf %11, %12 : f16
        %15 = arith.cmpf ogt, %13, %cst : f16
        %16 = arith.select %15, %14, %cst : f16
        memref.store %16, %arg3[%c0, %10, %9, %7] : memref<1x128x28x28xf16>
      }
      gpu.return
    }
    gpu.func @Unknown23(%arg0: memref<1x256x14x14xf16>, %arg1: memref<1x256x14x14xf16>, %arg2: memref<1x256x14x14xf16>) kernel {
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
        %13 = arith.cmpf ogt, %11, %cst : f16
        %14 = arith.select %13, %12, %cst : f16
        memref.store %14, %arg2[%c0, %10, %9, %7] : memref<1x256x14x14xf16>
      }
      gpu.return
    }
    gpu.func @Unknown19(%arg0: memref<1x256x14x14xf16>, %arg1: memref<1x256x14x14xf16>, %arg2: memref<1x256x14x14xf16>, %arg3: memref<1x256x14x14xf16>) kernel {
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
      scf.for %arg4 = %4 to %c50176 step %6 {
        %7 = arith.remsi %arg4, %c14 : index
        %8 = arith.divsi %arg4, %c14 : index
        %9 = arith.remsi %8, %c14 : index
        %10 = arith.divsi %8, %c14 : index
        %11 = memref.load %arg0[%c0, %10, %9, %7] : memref<1x256x14x14xf16>
        %12 = memref.load %arg1[%c0, %10, %9, %7] : memref<1x256x14x14xf16>
        %13 = memref.load %arg2[%c0, %10, %9, %7] : memref<1x256x14x14xf16>
        %14 = arith.addf %11, %12 : f16
        %15 = arith.cmpf ogt, %13, %cst : f16
        %16 = arith.select %15, %14, %cst : f16
        memref.store %16, %arg3[%c0, %10, %9, %7] : memref<1x256x14x14xf16>
      }
      gpu.return
    }
    gpu.func @Unknown8(%arg0: memref<1x512x7x7xf16>, %arg1: memref<1x512x7x7xf16>, %arg2: memref<1x512x7x7xf16>, %arg3: memref<1x512x7x7xf16>) kernel {
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
      scf.for %arg4 = %4 to %c25088 step %6 {
        %7 = arith.remsi %arg4, %c7 : index
        %8 = arith.divsi %arg4, %c7 : index
        %9 = arith.remsi %8, %c7 : index
        %10 = arith.divsi %8, %c7 : index
        %11 = memref.load %arg0[%c0, %10, %9, %7] : memref<1x512x7x7xf16>
        %12 = memref.load %arg1[%c0, %10, %9, %7] : memref<1x512x7x7xf16>
        %13 = memref.load %arg2[%c0, %10, %9, %7] : memref<1x512x7x7xf16>
        %14 = arith.addf %11, %12 : f16
        %15 = arith.cmpf ogt, %13, %cst : f16
        %16 = arith.select %15, %14, %cst : f16
        memref.store %16, %arg3[%c0, %10, %9, %7] : memref<1x512x7x7xf16>
      }
      gpu.return
    }
    gpu.func @Unknown4(%arg0: memref<1x512x7x7xf16>, %arg1: memref<1x512x7x7xf16>, %arg2: memref<1x512x7x7xf16>) kernel {
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
        %13 = arith.cmpf ogt, %11, %cst : f16
        %14 = arith.select %13, %12, %cst : f16
        memref.store %14, %arg2[%c0, %10, %9, %7] : memref<1x512x7x7xf16>
      }
      gpu.return
    }
    gpu.func @Unknown0(%arg0: memref<1x512xf16>, %arg1: memref<1x512x7x7xf16>, %arg2: memref<1x512x7x7xf16>) kernel {
      %c25088 = arith.constant 25088 : index
      %cst = arith.constant 4.900000e+01 : f16
      %cst_0 = arith.constant 0.000000e+00 : f16
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
        %11 = memref.load %arg0[%c0, %10] : memref<1x512xf16>
        %12 = memref.load %arg1[%c0, %10, %9, %7] : memref<1x512x7x7xf16>
        %13 = arith.divf %11, %cst : f16
        %14 = arith.cmpf ogt, %12, %cst_0 : f16
        %15 = arith.select %14, %13, %cst_0 : f16
        memref.store %15, %arg2[%c0, %10, %9, %7] : memref<1x512x7x7xf16>
      }
      gpu.return
    }
  }
  func.func @main(%arg0: memref<64xf32, "cuda"> {byre.argname = "Input0", byre.argtype = 1 : i32}, %arg1: memref<64xf32, "cuda"> {byre.argname = "Input1", byre.argtype = 1 : i32}, %arg2: memref<64xf32, "cuda"> {byre.argname = "Input2", byre.argtype = 1 : i32}, %arg3: memref<64xf32, "cuda"> {byre.argname = "Input3", byre.argtype = 1 : i32}, %arg4: memref<64xf32, "cuda"> {byre.argname = "Input4", byre.argtype = 1 : i32}, %arg5: memref<64xf32, "cuda"> {byre.argname = "Input5", byre.argtype = 1 : i32}, %arg6: memref<64xf32, "cuda"> {byre.argname = "Input6", byre.argtype = 1 : i32}, %arg7: memref<64xf32, "cuda"> {byre.argname = "Input7", byre.argtype = 1 : i32}, %arg8: memref<64xf32, "cuda"> {byre.argname = "Input8", byre.argtype = 1 : i32}, %arg9: memref<64xf32, "cuda"> {byre.argname = "Input9", byre.argtype = 1 : i32}, %arg10: memref<128xf32, "cuda"> {byre.argname = "Input10", byre.argtype = 1 : i32}, %arg11: memref<128xf32, "cuda"> {byre.argname = "Input11", byre.argtype = 1 : i32}, %arg12: memref<128xf32, "cuda"> {byre.argname = "Input12", byre.argtype = 1 : i32}, %arg13: memref<128xf32, "cuda"> {byre.argname = "Input13", byre.argtype = 1 : i32}, %arg14: memref<128xf32, "cuda"> {byre.argname = "Input14", byre.argtype = 1 : i32}, %arg15: memref<128xf32, "cuda"> {byre.argname = "Input15", byre.argtype = 1 : i32}, %arg16: memref<128xf32, "cuda"> {byre.argname = "Input16", byre.argtype = 1 : i32}, %arg17: memref<128xf32, "cuda"> {byre.argname = "Input17", byre.argtype = 1 : i32}, %arg18: memref<128xf32, "cuda"> {byre.argname = "Input18", byre.argtype = 1 : i32}, %arg19: memref<128xf32, "cuda"> {byre.argname = "Input19", byre.argtype = 1 : i32}, %arg20: memref<256xf32, "cuda"> {byre.argname = "Input20", byre.argtype = 1 : i32}, %arg21: memref<256xf32, "cuda"> {byre.argname = "Input21", byre.argtype = 1 : i32}, %arg22: memref<256xf32, "cuda"> {byre.argname = "Input22", byre.argtype = 1 : i32}, %arg23: memref<256xf32, "cuda"> {byre.argname = "Input23", byre.argtype = 1 : i32}, %arg24: memref<256xf32, "cuda"> {byre.argname = "Input24", byre.argtype = 1 : i32}, %arg25: memref<256xf32, "cuda"> {byre.argname = "Input25", byre.argtype = 1 : i32}, %arg26: memref<256xf32, "cuda"> {byre.argname = "Input26", byre.argtype = 1 : i32}, %arg27: memref<256xf32, "cuda"> {byre.argname = "Input27", byre.argtype = 1 : i32}, %arg28: memref<256xf32, "cuda"> {byre.argname = "Input28", byre.argtype = 1 : i32}, %arg29: memref<256xf32, "cuda"> {byre.argname = "Input29", byre.argtype = 1 : i32}, %arg30: memref<512xf32, "cuda"> {byre.argname = "Input30", byre.argtype = 1 : i32}, %arg31: memref<512xf32, "cuda"> {byre.argname = "Input31", byre.argtype = 1 : i32}, %arg32: memref<512xf32, "cuda"> {byre.argname = "Input32", byre.argtype = 1 : i32}, %arg33: memref<512xf32, "cuda"> {byre.argname = "Input33", byre.argtype = 1 : i32}, %arg34: memref<512xf32, "cuda"> {byre.argname = "Input34", byre.argtype = 1 : i32}, %arg35: memref<512xf32, "cuda"> {byre.argname = "Input35", byre.argtype = 1 : i32}, %arg36: memref<512xf32, "cuda"> {byre.argname = "Input36", byre.argtype = 1 : i32}, %arg37: memref<512xf32, "cuda"> {byre.argname = "Input37", byre.argtype = 1 : i32}, %arg38: memref<512xf32, "cuda"> {byre.argname = "Input38", byre.argtype = 1 : i32}, %arg39: memref<512xf32, "cuda"> {byre.argname = "Input39", byre.argtype = 1 : i32}, %arg40: memref<64xf32, "cuda"> {byre.argname = "Input40", byre.argtype = 1 : i32}, %arg41: memref<64xf32, "cuda"> {byre.argname = "Input41", byre.argtype = 1 : i32}, %arg42: memref<64xf32, "cuda"> {byre.argname = "Input42", byre.argtype = 1 : i32}, %arg43: memref<64xf32, "cuda"> {byre.argname = "Input43", byre.argtype = 1 : i32}, %arg44: memref<64xf32, "cuda"> {byre.argname = "Input44", byre.argtype = 1 : i32}, %arg45: memref<64xf32, "cuda"> {byre.argname = "Input45", byre.argtype = 1 : i32}, %arg46: memref<64xf32, "cuda"> {byre.argname = "Input46", byre.argtype = 1 : i32}, %arg47: memref<64xf32, "cuda"> {byre.argname = "Input47", byre.argtype = 1 : i32}, %arg48: memref<64xf32, "cuda"> {byre.argname = "Input48", byre.argtype = 1 : i32}, %arg49: memref<64xf32, "cuda"> {byre.argname = "Input49", byre.argtype = 1 : i32}, %arg50: memref<128xf32, "cuda"> {byre.argname = "Input50", byre.argtype = 1 : i32}, %arg51: memref<128xf32, "cuda"> {byre.argname = "Input51", byre.argtype = 1 : i32}, %arg52: memref<128xf32, "cuda"> {byre.argname = "Input52", byre.argtype = 1 : i32}, %arg53: memref<128xf32, "cuda"> {byre.argname = "Input53", byre.argtype = 1 : i32}, %arg54: memref<128xf32, "cuda"> {byre.argname = "Input54", byre.argtype = 1 : i32}, %arg55: memref<128xf32, "cuda"> {byre.argname = "Input55", byre.argtype = 1 : i32}, %arg56: memref<128xf32, "cuda"> {byre.argname = "Input56", byre.argtype = 1 : i32}, %arg57: memref<128xf32, "cuda"> {byre.argname = "Input57", byre.argtype = 1 : i32}, %arg58: memref<128xf32, "cuda"> {byre.argname = "Input58", byre.argtype = 1 : i32}, %arg59: memref<128xf32, "cuda"> {byre.argname = "Input59", byre.argtype = 1 : i32}, %arg60: memref<256xf32, "cuda"> {byre.argname = "Input60", byre.argtype = 1 : i32}, %arg61: memref<256xf32, "cuda"> {byre.argname = "Input61", byre.argtype = 1 : i32}, %arg62: memref<256xf32, "cuda"> {byre.argname = "Input62", byre.argtype = 1 : i32}, %arg63: memref<256xf32, "cuda"> {byre.argname = "Input63", byre.argtype = 1 : i32}, %arg64: memref<256xf32, "cuda"> {byre.argname = "Input64", byre.argtype = 1 : i32}, %arg65: memref<256xf32, "cuda"> {byre.argname = "Input65", byre.argtype = 1 : i32}, %arg66: memref<256xf32, "cuda"> {byre.argname = "Input66", byre.argtype = 1 : i32}, %arg67: memref<256xf32, "cuda"> {byre.argname = "Input67", byre.argtype = 1 : i32}, %arg68: memref<256xf32, "cuda"> {byre.argname = "Input68", byre.argtype = 1 : i32}, %arg69: memref<256xf32, "cuda"> {byre.argname = "Input69", byre.argtype = 1 : i32}, %arg70: memref<512xf32, "cuda"> {byre.argname = "Input70", byre.argtype = 1 : i32}, %arg71: memref<512xf32, "cuda"> {byre.argname = "Input71", byre.argtype = 1 : i32}, %arg72: memref<512xf32, "cuda"> {byre.argname = "Input72", byre.argtype = 1 : i32}, %arg73: memref<512xf32, "cuda"> {byre.argname = "Input73", byre.argtype = 1 : i32}, %arg74: memref<512xf32, "cuda"> {byre.argname = "Input74", byre.argtype = 1 : i32}, %arg75: memref<512xf32, "cuda"> {byre.argname = "Input75", byre.argtype = 1 : i32}, %arg76: memref<512xf32, "cuda"> {byre.argname = "Input76", byre.argtype = 1 : i32}, %arg77: memref<512xf32, "cuda"> {byre.argname = "Input77", byre.argtype = 1 : i32}, %arg78: memref<512xf32, "cuda"> {byre.argname = "Input78", byre.argtype = 1 : i32}, %arg79: memref<512xf32, "cuda"> {byre.argname = "Input79", byre.argtype = 1 : i32}, %arg80: memref<64x3x7x7xf16, "cuda"> {byre.argname = "Input80", byre.argtype = 1 : i32}, %arg81: memref<1x3x224x224xf16, "cuda"> {byre.argname = "Input81", byre.argtype = 1 : i32}, %arg82: memref<1x64x112x112xf16, "cuda"> {byre.argname = "Input82", byre.argtype = 1 : i32}, %arg83: memref<1x64x112x112xf16, "cuda"> {byre.argname = "Input83", byre.argtype = 1 : i32}, %arg84: memref<1x64x56x56xf16, "cuda"> {byre.argname = "Input84", byre.argtype = 1 : i32}, %arg85: memref<64x64x3x3xf16, "cuda"> {byre.argname = "Input85", byre.argtype = 1 : i32}, %arg86: memref<1x64x56x56xf16, "cuda"> {byre.argname = "Input86", byre.argtype = 1 : i32}, %arg87: memref<1x64x56x56xf16, "cuda"> {byre.argname = "Input87", byre.argtype = 1 : i32}, %arg88: memref<64x64x3x3xf16, "cuda"> {byre.argname = "Input88", byre.argtype = 1 : i32}, %arg89: memref<1x64x56x56xf16, "cuda"> {byre.argname = "Input89", byre.argtype = 1 : i32}, %arg90: memref<1x64x56x56xf16, "cuda"> {byre.argname = "Input90", byre.argtype = 1 : i32}, %arg91: memref<64x64x3x3xf16, "cuda"> {byre.argname = "Input91", byre.argtype = 1 : i32}, %arg92: memref<1x64x56x56xf16, "cuda"> {byre.argname = "Input92", byre.argtype = 1 : i32}, %arg93: memref<1x64x56x56xf16, "cuda"> {byre.argname = "Input93", byre.argtype = 1 : i32}, %arg94: memref<64x64x3x3xf16, "cuda"> {byre.argname = "Input94", byre.argtype = 1 : i32}, %arg95: memref<1x64x56x56xf16, "cuda"> {byre.argname = "Input95", byre.argtype = 1 : i32}, %arg96: memref<1x64x56x56xf16, "cuda"> {byre.argname = "Input96", byre.argtype = 1 : i32}, %arg97: memref<128x64x3x3xf16, "cuda"> {byre.argname = "Input97", byre.argtype = 1 : i32}, %arg98: memref<1x128x28x28xf16, "cuda"> {byre.argname = "Input98", byre.argtype = 1 : i32}, %arg99: memref<1x128x28x28xf16, "cuda"> {byre.argname = "Input99", byre.argtype = 1 : i32}, %arg100: memref<128x128x3x3xf16, "cuda"> {byre.argname = "Input100", byre.argtype = 1 : i32}, %arg101: memref<1x128x28x28xf16, "cuda"> {byre.argname = "Input101", byre.argtype = 1 : i32}, %arg102: memref<128x64x1x1xf16, "cuda"> {byre.argname = "Input102", byre.argtype = 1 : i32}, %arg103: memref<1x128x28x28xf16, "cuda"> {byre.argname = "Input103", byre.argtype = 1 : i32}, %arg104: memref<1x128x28x28xf16, "cuda"> {byre.argname = "Input104", byre.argtype = 1 : i32}, %arg105: memref<128x128x3x3xf16, "cuda"> {byre.argname = "Input105", byre.argtype = 1 : i32}, %arg106: memref<1x128x28x28xf16, "cuda"> {byre.argname = "Input106", byre.argtype = 1 : i32}, %arg107: memref<1x128x28x28xf16, "cuda"> {byre.argname = "Input107", byre.argtype = 1 : i32}, %arg108: memref<128x128x3x3xf16, "cuda"> {byre.argname = "Input108", byre.argtype = 1 : i32}, %arg109: memref<1x128x28x28xf16, "cuda"> {byre.argname = "Input109", byre.argtype = 1 : i32}, %arg110: memref<1x128x28x28xf16, "cuda"> {byre.argname = "Input110", byre.argtype = 1 : i32}, %arg111: memref<256x128x3x3xf16, "cuda"> {byre.argname = "Input111", byre.argtype = 1 : i32}, %arg112: memref<1x256x14x14xf16, "cuda"> {byre.argname = "Input112", byre.argtype = 1 : i32}, %arg113: memref<1x256x14x14xf16, "cuda"> {byre.argname = "Input113", byre.argtype = 1 : i32}, %arg114: memref<256x256x3x3xf16, "cuda"> {byre.argname = "Input114", byre.argtype = 1 : i32}, %arg115: memref<1x256x14x14xf16, "cuda"> {byre.argname = "Input115", byre.argtype = 1 : i32}, %arg116: memref<256x128x1x1xf16, "cuda"> {byre.argname = "Input116", byre.argtype = 1 : i32}, %arg117: memref<1x256x14x14xf16, "cuda"> {byre.argname = "Input117", byre.argtype = 1 : i32}, %arg118: memref<1x256x14x14xf16, "cuda"> {byre.argname = "Input118", byre.argtype = 1 : i32}, %arg119: memref<256x256x3x3xf16, "cuda"> {byre.argname = "Input119", byre.argtype = 1 : i32}, %arg120: memref<1x256x14x14xf16, "cuda"> {byre.argname = "Input120", byre.argtype = 1 : i32}, %arg121: memref<1x256x14x14xf16, "cuda"> {byre.argname = "Input121", byre.argtype = 1 : i32}, %arg122: memref<256x256x3x3xf16, "cuda"> {byre.argname = "Input122", byre.argtype = 1 : i32}, %arg123: memref<1x256x14x14xf16, "cuda"> {byre.argname = "Input123", byre.argtype = 1 : i32}, %arg124: memref<1x256x14x14xf16, "cuda"> {byre.argname = "Input124", byre.argtype = 1 : i32}, %arg125: memref<512x256x3x3xf16, "cuda"> {byre.argname = "Input125", byre.argtype = 1 : i32}, %arg126: memref<1x512x7x7xf16, "cuda"> {byre.argname = "Input126", byre.argtype = 1 : i32}, %arg127: memref<1x512x7x7xf16, "cuda"> {byre.argname = "Input127", byre.argtype = 1 : i32}, %arg128: memref<512x512x3x3xf16, "cuda"> {byre.argname = "Input128", byre.argtype = 1 : i32}, %arg129: memref<1x512x7x7xf16, "cuda"> {byre.argname = "Input129", byre.argtype = 1 : i32}, %arg130: memref<512x256x1x1xf16, "cuda"> {byre.argname = "Input130", byre.argtype = 1 : i32}, %arg131: memref<1x512x7x7xf16, "cuda"> {byre.argname = "Input131", byre.argtype = 1 : i32}, %arg132: memref<1x512x7x7xf16, "cuda"> {byre.argname = "Input132", byre.argtype = 1 : i32}, %arg133: memref<512x512x3x3xf16, "cuda"> {byre.argname = "Input133", byre.argtype = 1 : i32}, %arg134: memref<1x512x7x7xf16, "cuda"> {byre.argname = "Input134", byre.argtype = 1 : i32}, %arg135: memref<1x512x7x7xf16, "cuda"> {byre.argname = "Input135", byre.argtype = 1 : i32}, %arg136: memref<512x512x3x3xf16, "cuda"> {byre.argname = "Input136", byre.argtype = 1 : i32}, %arg137: memref<1x512x7x7xf16, "cuda"> {byre.argname = "Input137", byre.argtype = 1 : i32}, %arg138: memref<1x512x7x7xf16, "cuda"> {byre.argname = "Input138", byre.argtype = 1 : i32}, %arg139: memref<1x512xf16, "cuda"> {byre.argname = "Input139", byre.argtype = 1 : i32}, %arg140: memref<512x1000xf16, "cuda"> {byre.argname = "Input140", byre.argtype = 1 : i32}, %arg141: memref<1x1000xf16, "cuda"> {byre.argname = "Input141", byre.argtype = 1 : i32}, %arg142: memref<64xf32, "cuda"> {byre.argname = "Output0", byre.argtype = 2 : i32}, %arg143: memref<64xf32, "cuda"> {byre.argname = "Output1", byre.argtype = 2 : i32}, %arg144: memref<64x3x7x7xf32, "cuda"> {byre.argname = "Output2", byre.argtype = 2 : i32}, %arg145: memref<1000xf32, "cuda"> {byre.argname = "Output3", byre.argtype = 2 : i32}, %arg146: memref<1000x512xf32, "cuda"> {byre.argname = "Output4", byre.argtype = 2 : i32}, %arg147: memref<64xf32, "cuda"> {byre.argname = "Output5", byre.argtype = 2 : i32}, %arg148: memref<64xf32, "cuda"> {byre.argname = "Output6", byre.argtype = 2 : i32}, %arg149: memref<64xf32, "cuda"> {byre.argname = "Output7", byre.argtype = 2 : i32}, %arg150: memref<64xf32, "cuda"> {byre.argname = "Output8", byre.argtype = 2 : i32}, %arg151: memref<64x64x3x3xf32, "cuda"> {byre.argname = "Output9", byre.argtype = 2 : i32}, %arg152: memref<64x64x3x3xf32, "cuda"> {byre.argname = "Output10", byre.argtype = 2 : i32}, %arg153: memref<64xf32, "cuda"> {byre.argname = "Output11", byre.argtype = 2 : i32}, %arg154: memref<64xf32, "cuda"> {byre.argname = "Output12", byre.argtype = 2 : i32}, %arg155: memref<64xf32, "cuda"> {byre.argname = "Output13", byre.argtype = 2 : i32}, %arg156: memref<64xf32, "cuda"> {byre.argname = "Output14", byre.argtype = 2 : i32}, %arg157: memref<64x64x3x3xf32, "cuda"> {byre.argname = "Output15", byre.argtype = 2 : i32}, %arg158: memref<64x64x3x3xf32, "cuda"> {byre.argname = "Output16", byre.argtype = 2 : i32}, %arg159: memref<128xf32, "cuda"> {byre.argname = "Output17", byre.argtype = 2 : i32}, %arg160: memref<128xf32, "cuda"> {byre.argname = "Output18", byre.argtype = 2 : i32}, %arg161: memref<128xf32, "cuda"> {byre.argname = "Output19", byre.argtype = 2 : i32}, %arg162: memref<128xf32, "cuda"> {byre.argname = "Output20", byre.argtype = 2 : i32}, %arg163: memref<128x64x3x3xf32, "cuda"> {byre.argname = "Output21", byre.argtype = 2 : i32}, %arg164: memref<128x128x3x3xf32, "cuda"> {byre.argname = "Output22", byre.argtype = 2 : i32}, %arg165: memref<128x64x1x1xf32, "cuda"> {byre.argname = "Output23", byre.argtype = 2 : i32}, %arg166: memref<128xf32, "cuda"> {byre.argname = "Output24", byre.argtype = 2 : i32}, %arg167: memref<128xf32, "cuda"> {byre.argname = "Output25", byre.argtype = 2 : i32}, %arg168: memref<128xf32, "cuda"> {byre.argname = "Output26", byre.argtype = 2 : i32}, %arg169: memref<128xf32, "cuda"> {byre.argname = "Output27", byre.argtype = 2 : i32}, %arg170: memref<128xf32, "cuda"> {byre.argname = "Output28", byre.argtype = 2 : i32}, %arg171: memref<128xf32, "cuda"> {byre.argname = "Output29", byre.argtype = 2 : i32}, %arg172: memref<128x128x3x3xf32, "cuda"> {byre.argname = "Output30", byre.argtype = 2 : i32}, %arg173: memref<128x128x3x3xf32, "cuda"> {byre.argname = "Output31", byre.argtype = 2 : i32}, %arg174: memref<256xf32, "cuda"> {byre.argname = "Output32", byre.argtype = 2 : i32}, %arg175: memref<256xf32, "cuda"> {byre.argname = "Output33", byre.argtype = 2 : i32}, %arg176: memref<256xf32, "cuda"> {byre.argname = "Output34", byre.argtype = 2 : i32}, %arg177: memref<256xf32, "cuda"> {byre.argname = "Output35", byre.argtype = 2 : i32}, %arg178: memref<256x128x3x3xf32, "cuda"> {byre.argname = "Output36", byre.argtype = 2 : i32}, %arg179: memref<256x256x3x3xf32, "cuda"> {byre.argname = "Output37", byre.argtype = 2 : i32}, %arg180: memref<256x128x1x1xf32, "cuda"> {byre.argname = "Output38", byre.argtype = 2 : i32}, %arg181: memref<256xf32, "cuda"> {byre.argname = "Output39", byre.argtype = 2 : i32}, %arg182: memref<256xf32, "cuda"> {byre.argname = "Output40", byre.argtype = 2 : i32}, %arg183: memref<256xf32, "cuda"> {byre.argname = "Output41", byre.argtype = 2 : i32}, %arg184: memref<256xf32, "cuda"> {byre.argname = "Output42", byre.argtype = 2 : i32}, %arg185: memref<256xf32, "cuda"> {byre.argname = "Output43", byre.argtype = 2 : i32}, %arg186: memref<256xf32, "cuda"> {byre.argname = "Output44", byre.argtype = 2 : i32}, %arg187: memref<256x256x3x3xf32, "cuda"> {byre.argname = "Output45", byre.argtype = 2 : i32}, %arg188: memref<256x256x3x3xf32, "cuda"> {byre.argname = "Output46", byre.argtype = 2 : i32}, %arg189: memref<512xf32, "cuda"> {byre.argname = "Output47", byre.argtype = 2 : i32}, %arg190: memref<512xf32, "cuda"> {byre.argname = "Output48", byre.argtype = 2 : i32}, %arg191: memref<512xf32, "cuda"> {byre.argname = "Output49", byre.argtype = 2 : i32}, %arg192: memref<512xf32, "cuda"> {byre.argname = "Output50", byre.argtype = 2 : i32}, %arg193: memref<512x256x3x3xf32, "cuda"> {byre.argname = "Output51", byre.argtype = 2 : i32}, %arg194: memref<512x512x3x3xf32, "cuda"> {byre.argname = "Output52", byre.argtype = 2 : i32}, %arg195: memref<512x256x1x1xf32, "cuda"> {byre.argname = "Output53", byre.argtype = 2 : i32}, %arg196: memref<512xf32, "cuda"> {byre.argname = "Output54", byre.argtype = 2 : i32}, %arg197: memref<512xf32, "cuda"> {byre.argname = "Output55", byre.argtype = 2 : i32}, %arg198: memref<512xf32, "cuda"> {byre.argname = "Output56", byre.argtype = 2 : i32}, %arg199: memref<512xf32, "cuda"> {byre.argname = "Output57", byre.argtype = 2 : i32}, %arg200: memref<512xf32, "cuda"> {byre.argname = "Output58", byre.argtype = 2 : i32}, %arg201: memref<512xf32, "cuda"> {byre.argname = "Output59", byre.argtype = 2 : i32}, %arg202: memref<512x512x3x3xf32, "cuda"> {byre.argname = "Output60", byre.argtype = 2 : i32}, %arg203: memref<512x512x3x3xf32, "cuda"> {byre.argname = "Output61", byre.argtype = 2 : i32}) attributes {byre.entry_point} {
    %alloc = memref.alloc() : memref<25927680xi8, "cuda">
    %0 = "byre.alias"(%alloc) <{offset = 1671168 : i64}> : (memref<25927680xi8, "cuda">) -> memref<1x512xf16, "cuda">
    byre.compute @MatmulOp_f16f16_f16(%arg141, %arg140, %0) {device = "cuda", lhs_contracting_dimension = 1 : i64, memory_effects = [1 : i32, 1 : i32, 2 : i32], rhs_contracting_dimension = 1 : i64} : memref<1x1000xf16, "cuda">, memref<512x1000xf16, "cuda">, memref<1x512xf16, "cuda">
    %1 = "byre.alias"(%alloc) <{offset = 16490496 : i64}> : (memref<25927680xi8, "cuda">) -> memref<1x512x7x7xf16, "cuda">
    byre.compute @PTXOp(%0, %arg138, %1) {BlockSize.x = 256 : i32, GridSize.x = 25 : i32, arg_ranks = [2 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown0", memory_effects = [1 : i32, 1 : i32, 2 : i32]} : memref<1x512xf16, "cuda">, memref<1x512x7x7xf16, "cuda">, memref<1x512x7x7xf16, "cuda">
    %2 = "byre.alias"(%alloc) <{offset = 21209088 : i64}> : (memref<25927680xi8, "cuda">) -> memref<1x512x7x7xf16, "cuda">
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%arg137, %arg39, %1, %2, %arg201, %arg200) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<1x512x7x7xf16, "cuda">, memref<1x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">
    %3 = "byre.alias"(%alloc) <{offset = 16540672 : i64}> : (memref<25927680xi8, "cuda">) -> memref<1x512x7x7xf16, "cuda">
    byre.compute @ConvBackwardDataOp_f16f16_f16(%2, %arg136, %3) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x512x7x7xf16, "cuda">, memref<512x512x3x3xf16, "cuda">, memref<1x512x7x7xf16, "cuda">
    %4 = "byre.alias"(%alloc) <{offset = 1671168 : i64}> : (memref<25927680xi8, "cuda">) -> memref<512x512x3x3xf16, "cuda">
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%arg135, %2, %4) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x512x7x7xf16, "cuda">, memref<1x512x7x7xf16, "cuda">, memref<512x512x3x3xf16, "cuda">
    byre.compute @PTXOp(%arg135, %3, %2) {BlockSize.x = 256 : i32, GridSize.x = 25 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown4", memory_effects = [1 : i32, 1 : i32, 2 : i32]} : memref<1x512x7x7xf16, "cuda">, memref<1x512x7x7xf16, "cuda">, memref<1x512x7x7xf16, "cuda">
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%arg134, %arg37, %2, %3, %arg199, %arg198) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<1x512x7x7xf16, "cuda">, memref<1x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">
    %5 = "byre.alias"(%alloc) <{offset = 14131200 : i64}> : (memref<25927680xi8, "cuda">) -> memref<1x512x7x7xf16, "cuda">
    byre.compute @ConvBackwardDataOp_f16f16_f16(%3, %arg133, %5) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x512x7x7xf16, "cuda">, memref<512x512x3x3xf16, "cuda">, memref<1x512x7x7xf16, "cuda">
    %6 = "byre.alias"(%alloc) <{offset = 21209088 : i64}> : (memref<25927680xi8, "cuda">) -> memref<512x512x3x3xf16, "cuda">
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%arg132, %3, %6) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x512x7x7xf16, "cuda">, memref<1x512x7x7xf16, "cuda">, memref<512x512x3x3xf16, "cuda">
    %7 = "byre.alias"(%alloc) <{offset = 10919936 : i64}> : (memref<25927680xi8, "cuda">) -> memref<1x512x7x7xf16, "cuda">
    byre.compute @PTXOp(%1, %5, %arg132, %7) {BlockSize.x = 256 : i32, GridSize.x = 25 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown8", memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<1x512x7x7xf16, "cuda">, memref<1x512x7x7xf16, "cuda">, memref<1x512x7x7xf16, "cuda">, memref<1x512x7x7xf16, "cuda">
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%arg129, %arg33, %7, %5, %arg192, %arg191) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<1x512x7x7xf16, "cuda">, memref<1x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">
    %8 = "byre.alias"(%alloc) <{offset = 12525568 : i64}> : (memref<25927680xi8, "cuda">) -> memref<1x512x7x7xf16, "cuda">
    byre.compute @ConvBackwardDataOp_f16f16_f16(%5, %arg128, %8) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x512x7x7xf16, "cuda">, memref<512x512x3x3xf16, "cuda">, memref<1x512x7x7xf16, "cuda">
    %9 = "byre.alias"(%alloc) <{offset = 16490496 : i64}> : (memref<25927680xi8, "cuda">) -> memref<512x512x3x3xf16, "cuda">
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%arg127, %5, %9) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x512x7x7xf16, "cuda">, memref<1x512x7x7xf16, "cuda">, memref<512x512x3x3xf16, "cuda">
    byre.compute @PTXOp(%arg127, %8, %5) {BlockSize.x = 256 : i32, GridSize.x = 25 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown4", memory_effects = [1 : i32, 1 : i32, 2 : i32]} : memref<1x512x7x7xf16, "cuda">, memref<1x512x7x7xf16, "cuda">, memref<1x512x7x7xf16, "cuda">
    %10 = "byre.alias"(%alloc) <{offset = 10970112 : i64}> : (memref<25927680xi8, "cuda">) -> memref<1x512x7x7xf16, "cuda">
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%arg126, %arg31, %5, %10, %arg190, %arg189) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<1x512x7x7xf16, "cuda">, memref<1x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">
    %11 = "byre.alias"(%alloc) <{offset = 12525568 : i64}> : (memref<25927680xi8, "cuda">) -> memref<1x256x14x14xf16, "cuda">
    byre.compute @ConvBackwardDataOp_f16f16_f16(%10, %arg125, %11) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<1x512x7x7xf16, "cuda">, memref<512x256x3x3xf16, "cuda">, memref<1x256x14x14xf16, "cuda">
    %12 = "byre.alias"(%alloc) <{offset = 14131200 : i64}> : (memref<25927680xi8, "cuda">) -> memref<512x256x3x3xf16, "cuda">
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%arg124, %10, %12) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<1x256x14x14xf16, "cuda">, memref<1x512x7x7xf16, "cuda">, memref<512x256x3x3xf16, "cuda">
    %13 = "byre.alias"(%alloc) <{offset = 9740288 : i64}> : (memref<25927680xi8, "cuda">) -> memref<1x512x7x7xf16, "cuda">
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%arg131, %arg35, %7, %13, %arg197, %arg196) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<1x512x7x7xf16, "cuda">, memref<1x512x7x7xf16, "cuda">, memref<512xf32, "cuda">, memref<512xf32, "cuda">
    %14 = "byre.alias"(%alloc) <{offset = 12625920 : i64}> : (memref<25927680xi8, "cuda">) -> memref<1x256x14x14xf16, "cuda">
    byre.compute @ConvBackwardDataOp_f16f16_f16(%13, %arg130, %14) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<0> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<1x512x7x7xf16, "cuda">, memref<512x256x1x1xf16, "cuda">, memref<1x256x14x14xf16, "cuda">
    %15 = "byre.alias"(%alloc) <{offset = 819200 : i64}> : (memref<25927680xi8, "cuda">) -> memref<512x256x1x1xf16, "cuda">
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%arg124, %13, %15) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<0> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<1x256x14x14xf16, "cuda">, memref<1x512x7x7xf16, "cuda">, memref<512x256x1x1xf16, "cuda">
    %16 = "byre.alias"(%alloc) <{offset = 10919936 : i64}> : (memref<25927680xi8, "cuda">) -> memref<1x256x14x14xf16, "cuda">
    byre.compute @PTXOp(%14, %11, %arg124, %16) {BlockSize.x = 256 : i32, GridSize.x = 49 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown19", memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<1x256x14x14xf16, "cuda">, memref<1x256x14x14xf16, "cuda">, memref<1x256x14x14xf16, "cuda">, memref<1x256x14x14xf16, "cuda">
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%arg123, %arg29, %16, %11, %arg186, %arg185) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<1x256x14x14xf16, "cuda">, memref<1x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">
    %17 = "byre.alias"(%alloc) <{offset = 11020288 : i64}> : (memref<25927680xi8, "cuda">) -> memref<1x256x14x14xf16, "cuda">
    byre.compute @ConvBackwardDataOp_f16f16_f16(%11, %arg122, %17) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x256x14x14xf16, "cuda">, memref<256x256x3x3xf16, "cuda">, memref<1x256x14x14xf16, "cuda">
    %18 = "byre.alias"(%alloc) <{offset = 9740288 : i64}> : (memref<25927680xi8, "cuda">) -> memref<256x256x3x3xf16, "cuda">
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%arg121, %11, %18) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x256x14x14xf16, "cuda">, memref<1x256x14x14xf16, "cuda">, memref<256x256x3x3xf16, "cuda">
    byre.compute @PTXOp(%arg121, %17, %11) {BlockSize.x = 256 : i32, GridSize.x = 49 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown23", memory_effects = [1 : i32, 1 : i32, 2 : i32]} : memref<1x256x14x14xf16, "cuda">, memref<1x256x14x14xf16, "cuda">, memref<1x256x14x14xf16, "cuda">
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%arg120, %arg27, %11, %17, %arg184, %arg183) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<1x256x14x14xf16, "cuda">, memref<1x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">
    %19 = "byre.alias"(%alloc) <{offset = 8560640 : i64}> : (memref<25927680xi8, "cuda">) -> memref<1x256x14x14xf16, "cuda">
    byre.compute @ConvBackwardDataOp_f16f16_f16(%17, %arg119, %19) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x256x14x14xf16, "cuda">, memref<256x256x3x3xf16, "cuda">, memref<1x256x14x14xf16, "cuda">
    %20 = "byre.alias"(%alloc) <{offset = 7380992 : i64}> : (memref<25927680xi8, "cuda">) -> memref<256x256x3x3xf16, "cuda">
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%arg118, %17, %20) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x256x14x14xf16, "cuda">, memref<1x256x14x14xf16, "cuda">, memref<256x256x3x3xf16, "cuda">
    byre.compute @PTXOp(%16, %19, %arg118, %11) {BlockSize.x = 256 : i32, GridSize.x = 49 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown19", memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<1x256x14x14xf16, "cuda">, memref<1x256x14x14xf16, "cuda">, memref<1x256x14x14xf16, "cuda">, memref<1x256x14x14xf16, "cuda">
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%arg115, %arg23, %11, %16, %arg177, %arg176) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<1x256x14x14xf16, "cuda">, memref<1x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">
    byre.compute @ConvBackwardDataOp_f16f16_f16(%16, %arg114, %14) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x256x14x14xf16, "cuda">, memref<256x256x3x3xf16, "cuda">, memref<1x256x14x14xf16, "cuda">
    %21 = "byre.alias"(%alloc) <{offset = 8560640 : i64}> : (memref<25927680xi8, "cuda">) -> memref<256x256x3x3xf16, "cuda">
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%arg113, %16, %21) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x256x14x14xf16, "cuda">, memref<1x256x14x14xf16, "cuda">, memref<256x256x3x3xf16, "cuda">
    byre.compute @PTXOp(%arg113, %14, %16) {BlockSize.x = 256 : i32, GridSize.x = 49 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown23", memory_effects = [1 : i32, 1 : i32, 2 : i32]} : memref<1x256x14x14xf16, "cuda">, memref<1x256x14x14xf16, "cuda">, memref<1x256x14x14xf16, "cuda">
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%arg112, %arg21, %16, %14, %arg175, %arg174) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<1x256x14x14xf16, "cuda">, memref<1x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">
    %22 = "byre.alias"(%alloc) <{offset = 10919936 : i64}> : (memref<25927680xi8, "cuda">) -> memref<1x128x28x28xf16, "cuda">
    byre.compute @ConvBackwardDataOp_f16f16_f16(%14, %arg111, %22) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<1x256x14x14xf16, "cuda">, memref<256x128x3x3xf16, "cuda">, memref<1x128x28x28xf16, "cuda">
    %23 = "byre.alias"(%alloc) <{offset = 6791168 : i64}> : (memref<25927680xi8, "cuda">) -> memref<256x128x3x3xf16, "cuda">
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%arg110, %14, %23) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<1x128x28x28xf16, "cuda">, memref<1x256x14x14xf16, "cuda">, memref<256x128x3x3xf16, "cuda">
    %24 = "byre.alias"(%alloc) <{offset = 11120640 : i64}> : (memref<25927680xi8, "cuda">) -> memref<1x256x14x14xf16, "cuda">
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%arg117, %arg25, %11, %24, %arg182, %arg181) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<1x256x14x14xf16, "cuda">, memref<1x256x14x14xf16, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">
    %25 = "byre.alias"(%alloc) <{offset = 12525568 : i64}> : (memref<25927680xi8, "cuda">) -> memref<1x128x28x28xf16, "cuda">
    byre.compute @ConvBackwardDataOp_f16f16_f16(%24, %arg116, %25) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<0> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<1x256x14x14xf16, "cuda">, memref<256x128x1x1xf16, "cuda">, memref<1x128x28x28xf16, "cuda">
    %26 = "byre.alias"(%alloc) <{offset = 311296 : i64}> : (memref<25927680xi8, "cuda">) -> memref<256x128x1x1xf16, "cuda">
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%arg110, %24, %26) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<0> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<1x128x28x28xf16, "cuda">, memref<1x256x14x14xf16, "cuda">, memref<256x128x1x1xf16, "cuda">
    %27 = "byre.alias"(%alloc) <{offset = 1081344 : i64}> : (memref<25927680xi8, "cuda">) -> memref<1x128x28x28xf16, "cuda">
    byre.compute @PTXOp(%25, %22, %arg110, %27) {BlockSize.x = 256 : i32, GridSize.x = 98 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown38", memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<1x128x28x28xf16, "cuda">, memref<1x128x28x28xf16, "cuda">, memref<1x128x28x28xf16, "cuda">, memref<1x128x28x28xf16, "cuda">
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%arg109, %arg19, %27, %25, %arg171, %arg170) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<1x128x28x28xf16, "cuda">, memref<1x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">
    byre.compute @ConvBackwardDataOp_f16f16_f16(%25, %arg108, %22) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x128x28x28xf16, "cuda">, memref<128x128x3x3xf16, "cuda">, memref<1x128x28x28xf16, "cuda">
    %28 = "byre.alias"(%alloc) <{offset = 0 : i64}> : (memref<25927680xi8, "cuda">) -> memref<128x128x3x3xf16, "cuda">
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%arg107, %25, %28) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x128x28x28xf16, "cuda">, memref<1x128x28x28xf16, "cuda">, memref<128x128x3x3xf16, "cuda">
    byre.compute @PTXOp(%arg107, %22, %25) {BlockSize.x = 256 : i32, GridSize.x = 98 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown42", memory_effects = [1 : i32, 1 : i32, 2 : i32]} : memref<1x128x28x28xf16, "cuda">, memref<1x128x28x28xf16, "cuda">, memref<1x128x28x28xf16, "cuda">
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%arg106, %arg17, %25, %22, %arg169, %arg168) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<1x128x28x28xf16, "cuda">, memref<1x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">
    byre.compute @ConvBackwardDataOp_f16f16_f16(%22, %arg105, %25) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x128x28x28xf16, "cuda">, memref<128x128x3x3xf16, "cuda">, memref<1x128x28x28xf16, "cuda">
    %29 = "byre.alias"(%alloc) <{offset = 1376256 : i64}> : (memref<25927680xi8, "cuda">) -> memref<128x128x3x3xf16, "cuda">
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%arg104, %22, %29) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x128x28x28xf16, "cuda">, memref<1x128x28x28xf16, "cuda">, memref<128x128x3x3xf16, "cuda">
    %30 = "byre.alias"(%alloc) <{offset = 6389760 : i64}> : (memref<25927680xi8, "cuda">) -> memref<1x128x28x28xf16, "cuda">
    byre.compute @PTXOp(%27, %25, %arg104, %30) {BlockSize.x = 256 : i32, GridSize.x = 98 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown38", memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<1x128x28x28xf16, "cuda">, memref<1x128x28x28xf16, "cuda">, memref<1x128x28x28xf16, "cuda">, memref<1x128x28x28xf16, "cuda">
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%arg101, %arg13, %30, %25, %arg162, %arg161) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<1x128x28x28xf16, "cuda">, memref<1x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">
    byre.compute @ConvBackwardDataOp_f16f16_f16(%25, %arg100, %22) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x128x28x28xf16, "cuda">, memref<128x128x3x3xf16, "cuda">, memref<1x128x28x28xf16, "cuda">
    %31 = "byre.alias"(%alloc) <{offset = 1081344 : i64}> : (memref<25927680xi8, "cuda">) -> memref<128x128x3x3xf16, "cuda">
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%arg99, %25, %31) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x128x28x28xf16, "cuda">, memref<1x128x28x28xf16, "cuda">, memref<128x128x3x3xf16, "cuda">
    byre.compute @PTXOp(%arg99, %22, %25) {BlockSize.x = 256 : i32, GridSize.x = 98 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown42", memory_effects = [1 : i32, 1 : i32, 2 : i32]} : memref<1x128x28x28xf16, "cuda">, memref<1x128x28x28xf16, "cuda">, memref<1x128x28x28xf16, "cuda">
    %32 = "byre.alias"(%alloc) <{offset = 6590464 : i64}> : (memref<25927680xi8, "cuda">) -> memref<1x128x28x28xf16, "cuda">
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%arg98, %arg11, %25, %32, %arg160, %arg159) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<1x128x28x28xf16, "cuda">, memref<1x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">
    %33 = "byre.alias"(%alloc) <{offset = 10919936 : i64}> : (memref<25927680xi8, "cuda">) -> memref<1x64x56x56xf16, "cuda">
    byre.compute @ConvBackwardDataOp_f16f16_f16(%32, %arg97, %33) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<1x128x28x28xf16, "cuda">, memref<128x64x3x3xf16, "cuda">, memref<1x64x56x56xf16, "cuda">
    %34 = "byre.alias"(%alloc) <{offset = 671744 : i64}> : (memref<25927680xi8, "cuda">) -> memref<128x64x3x3xf16, "cuda">
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%arg96, %32, %34) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<1x64x56x56xf16, "cuda">, memref<1x128x28x28xf16, "cuda">, memref<128x64x3x3xf16, "cuda">
    %35 = "byre.alias"(%alloc) <{offset = 11321344 : i64}> : (memref<25927680xi8, "cuda">) -> memref<1x128x28x28xf16, "cuda">
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%arg103, %arg15, %30, %35, %arg167, %arg166) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<1x128x28x28xf16, "cuda">, memref<1x128x28x28xf16, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">
    %36 = "byre.alias"(%alloc) <{offset = 12525568 : i64}> : (memref<25927680xi8, "cuda">) -> memref<1x64x56x56xf16, "cuda">
    byre.compute @ConvBackwardDataOp_f16f16_f16(%35, %arg102, %36) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<0> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<1x128x28x28xf16, "cuda">, memref<128x64x1x1xf16, "cuda">, memref<1x64x56x56xf16, "cuda">
    %37 = "byre.alias"(%alloc) <{offset = 294912 : i64}> : (memref<25927680xi8, "cuda">) -> memref<128x64x1x1xf16, "cuda">
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%arg96, %35, %37) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<0> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<1x64x56x56xf16, "cuda">, memref<1x128x28x28xf16, "cuda">, memref<128x64x1x1xf16, "cuda">
    %38 = "byre.alias"(%alloc) <{offset = 6389760 : i64}> : (memref<25927680xi8, "cuda">) -> memref<1x64x56x56xf16, "cuda">
    byre.compute @PTXOp(%36, %33, %arg96, %38) {BlockSize.x = 256 : i32, GridSize.x = 196 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown57", memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<1x64x56x56xf16, "cuda">, memref<1x64x56x56xf16, "cuda">, memref<1x64x56x56xf16, "cuda">, memref<1x64x56x56xf16, "cuda">
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%arg95, %arg9, %38, %36, %arg156, %arg155) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x64x56x56xf16, "cuda">, memref<64xf32, "cuda">, memref<1x64x56x56xf16, "cuda">, memref<1x64x56x56xf16, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">
    byre.compute @ConvBackwardDataOp_f16f16_f16(%36, %arg94, %33) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x64x56x56xf16, "cuda">, memref<64x64x3x3xf16, "cuda">, memref<1x64x56x56xf16, "cuda">
    %39 = "byre.alias"(%alloc) <{offset = 598016 : i64}> : (memref<25927680xi8, "cuda">) -> memref<64x64x3x3xf16, "cuda">
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%arg93, %36, %39) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x64x56x56xf16, "cuda">, memref<1x64x56x56xf16, "cuda">, memref<64x64x3x3xf16, "cuda">
    byre.compute @PTXOp(%arg93, %33, %36) {BlockSize.x = 256 : i32, GridSize.x = 196 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown61", memory_effects = [1 : i32, 1 : i32, 2 : i32]} : memref<1x64x56x56xf16, "cuda">, memref<1x64x56x56xf16, "cuda">, memref<1x64x56x56xf16, "cuda">
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%arg92, %arg7, %36, %33, %arg154, %arg153) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x64x56x56xf16, "cuda">, memref<64xf32, "cuda">, memref<1x64x56x56xf16, "cuda">, memref<1x64x56x56xf16, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">
    byre.compute @ConvBackwardDataOp_f16f16_f16(%33, %arg91, %36) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x64x56x56xf16, "cuda">, memref<64x64x3x3xf16, "cuda">, memref<1x64x56x56xf16, "cuda">
    %40 = "byre.alias"(%alloc) <{offset = 524288 : i64}> : (memref<25927680xi8, "cuda">) -> memref<64x64x3x3xf16, "cuda">
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%arg90, %33, %40) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x64x56x56xf16, "cuda">, memref<1x64x56x56xf16, "cuda">, memref<64x64x3x3xf16, "cuda">
    byre.compute @PTXOp(%38, %36, %arg90, %33) {BlockSize.x = 256 : i32, GridSize.x = 196 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown57", memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<1x64x56x56xf16, "cuda">, memref<1x64x56x56xf16, "cuda">, memref<1x64x56x56xf16, "cuda">, memref<1x64x56x56xf16, "cuda">
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%arg89, %arg5, %33, %36, %arg150, %arg149) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x64x56x56xf16, "cuda">, memref<64xf32, "cuda">, memref<1x64x56x56xf16, "cuda">, memref<1x64x56x56xf16, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">
    %41 = "byre.alias"(%alloc) <{offset = 11321344 : i64}> : (memref<25927680xi8, "cuda">) -> memref<1x64x56x56xf16, "cuda">
    byre.compute @ConvBackwardDataOp_f16f16_f16(%36, %arg88, %41) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x64x56x56xf16, "cuda">, memref<64x64x3x3xf16, "cuda">, memref<1x64x56x56xf16, "cuda">
    %42 = "byre.alias"(%alloc) <{offset = 376832 : i64}> : (memref<25927680xi8, "cuda">) -> memref<64x64x3x3xf16, "cuda">
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%arg87, %36, %42) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x64x56x56xf16, "cuda">, memref<1x64x56x56xf16, "cuda">, memref<64x64x3x3xf16, "cuda">
    byre.compute @PTXOp(%arg87, %41, %36) {BlockSize.x = 256 : i32, GridSize.x = 196 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown61", memory_effects = [1 : i32, 1 : i32, 2 : i32]} : memref<1x64x56x56xf16, "cuda">, memref<1x64x56x56xf16, "cuda">, memref<1x64x56x56xf16, "cuda">
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%arg86, %arg3, %36, %41, %arg148, %arg147) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x64x56x56xf16, "cuda">, memref<64xf32, "cuda">, memref<1x64x56x56xf16, "cuda">, memref<1x64x56x56xf16, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">
    byre.compute @ConvBackwardDataOp_f16f16_f16(%41, %arg85, %36) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x64x56x56xf16, "cuda">, memref<64x64x3x3xf16, "cuda">, memref<1x64x56x56xf16, "cuda">
    %43 = "byre.alias"(%alloc) <{offset = 450560 : i64}> : (memref<25927680xi8, "cuda">) -> memref<64x64x3x3xf16, "cuda">
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%arg84, %41, %43) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<1x64x56x56xf16, "cuda">, memref<1x64x56x56xf16, "cuda">, memref<64x64x3x3xf16, "cuda">
    byre.compute @PTXOp(%33, %36, %38) {BlockSize.x = 256 : i32, GridSize.x = 196 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown73", memory_effects = [1 : i32, 1 : i32, 2 : i32]} : memref<1x64x56x56xf16, "cuda">, memref<1x64x56x56xf16, "cuda">, memref<1x64x56x56xf16, "cuda">
    %44 = "byre.alias"(%alloc) <{offset = 12525568 : i64}> : (memref<25927680xi8, "cuda">) -> memref<1x64x112x112xf16, "cuda">
    byre.compute @PoolMaxGradOp_f16f16_f16(%arg83, %38, %44) {device = "cuda", memory_effects = [1 : i32, 1 : i32, 2 : i32], padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} : memref<1x64x112x112xf16, "cuda">, memref<1x64x56x56xf16, "cuda">, memref<1x64x112x112xf16, "cuda">
    %45 = "byre.alias"(%alloc) <{offset = 10919936 : i64}> : (memref<25927680xi8, "cuda">) -> memref<1x64x112x112xf16, "cuda">
    byre.compute @PTXOp(%arg83, %44, %45) {BlockSize.x = 256 : i32, GridSize.x = 784 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown74", memory_effects = [1 : i32, 1 : i32, 2 : i32]} : memref<1x64x112x112xf16, "cuda">, memref<1x64x112x112xf16, "cuda">, memref<1x64x112x112xf16, "cuda">
    byre.compute @BatchNormGradOp_f16f32f16_f16f32f32(%arg82, %arg1, %45, %44, %arg143, %arg142) {device = "cuda", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64, memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32]} : memref<1x64x112x112xf16, "cuda">, memref<64xf32, "cuda">, memref<1x64x112x112xf16, "cuda">, memref<1x64x112x112xf16, "cuda">, memref<64xf32, "cuda">, memref<64xf32, "cuda">
    %46 = "byre.alias"(%alloc) <{offset = 10919936 : i64}> : (memref<25927680xi8, "cuda">) -> memref<64x3x7x7xf16, "cuda">
    byre.compute @ConvBackwardFilterOp_f16f16_f16(%arg81, %44, %46) {batch_group_count = 1 : i64, device = "cuda", feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", memory_effects = [1 : i32, 1 : i32, 2 : i32], output_layout = "NCHW", padding = dense<3> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<1x3x224x224xf16, "cuda">, memref<1x64x112x112xf16, "cuda">, memref<64x3x7x7xf16, "cuda">
    byre.compute @PTXOp(%46, %arg144) {BlockSize.x = 256 : i32, GridSize.x = 10 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown77", memory_effects = [1 : i32, 2 : i32]} : memref<64x3x7x7xf16, "cuda">, memref<64x3x7x7xf32, "cuda">
    byre.compute @PTXOp(%arg141, %arg145) {BlockSize.x = 256 : i32, GridSize.x = 1 : i32, arg_ranks = [2 : i32, 2 : i32], kernel_name = "Unknown78", memory_effects = [1 : i32, 2 : i32]} : memref<1x1000xf16, "cuda">, memref<1000xf32, "cuda">
    %47 = "byre.alias"(%arg141) <{offset = 0 : i64}> : (memref<1x1000xf16, "cuda">) -> memref<1000x1xf16, "cuda">
    %48 = "byre.alias"(%alloc) <{offset = 12525568 : i64}> : (memref<25927680xi8, "cuda">) -> memref<1000x512xf16, "cuda">
    byre.compute @MatmulOp_f16f16_f16(%47, %arg139, %48) {device = "cuda", lhs_contracting_dimension = 1 : i64, memory_effects = [1 : i32, 1 : i32, 2 : i32], rhs_contracting_dimension = 0 : i64} : memref<1000x1xf16, "cuda">, memref<1x512xf16, "cuda">, memref<1000x512xf16, "cuda">
    byre.compute @PTXOp(%48, %arg146) {BlockSize.x = 256 : i32, GridSize.x = 500 : i32, arg_ranks = [2 : i32, 2 : i32], kernel_name = "Unknown79", memory_effects = [1 : i32, 2 : i32]} : memref<1000x512xf16, "cuda">, memref<1000x512xf32, "cuda">
    byre.compute @PTXOp(%43, %arg151) {BlockSize.x = 256 : i32, GridSize.x = 36 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown80", memory_effects = [1 : i32, 2 : i32]} : memref<64x64x3x3xf16, "cuda">, memref<64x64x3x3xf32, "cuda">
    byre.compute @PTXOp(%42, %arg152) {BlockSize.x = 256 : i32, GridSize.x = 36 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown80", memory_effects = [1 : i32, 2 : i32]} : memref<64x64x3x3xf16, "cuda">, memref<64x64x3x3xf32, "cuda">
    byre.compute @PTXOp(%40, %arg157) {BlockSize.x = 256 : i32, GridSize.x = 36 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown80", memory_effects = [1 : i32, 2 : i32]} : memref<64x64x3x3xf16, "cuda">, memref<64x64x3x3xf32, "cuda">
    byre.compute @PTXOp(%39, %arg158) {BlockSize.x = 256 : i32, GridSize.x = 36 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown80", memory_effects = [1 : i32, 2 : i32]} : memref<64x64x3x3xf16, "cuda">, memref<64x64x3x3xf32, "cuda">
    byre.compute @PTXOp(%34, %arg163) {BlockSize.x = 256 : i32, GridSize.x = 72 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown84", memory_effects = [1 : i32, 2 : i32]} : memref<128x64x3x3xf16, "cuda">, memref<128x64x3x3xf32, "cuda">
    byre.compute @PTXOp(%31, %arg164) {BlockSize.x = 256 : i32, GridSize.x = 144 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown85", memory_effects = [1 : i32, 2 : i32]} : memref<128x128x3x3xf16, "cuda">, memref<128x128x3x3xf32, "cuda">
    byre.compute @PTXOp(%37, %arg165) {BlockSize.x = 256 : i32, GridSize.x = 8 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown86", memory_effects = [1 : i32, 2 : i32]} : memref<128x64x1x1xf16, "cuda">, memref<128x64x1x1xf32, "cuda">
    byre.compute @PTXOp(%29, %arg172) {BlockSize.x = 256 : i32, GridSize.x = 144 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown85", memory_effects = [1 : i32, 2 : i32]} : memref<128x128x3x3xf16, "cuda">, memref<128x128x3x3xf32, "cuda">
    byre.compute @PTXOp(%28, %arg173) {BlockSize.x = 256 : i32, GridSize.x = 144 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown85", memory_effects = [1 : i32, 2 : i32]} : memref<128x128x3x3xf16, "cuda">, memref<128x128x3x3xf32, "cuda">
    byre.compute @PTXOp(%23, %arg178) {BlockSize.x = 256 : i32, GridSize.x = 288 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown89", memory_effects = [1 : i32, 2 : i32]} : memref<256x128x3x3xf16, "cuda">, memref<256x128x3x3xf32, "cuda">
    byre.compute @PTXOp(%21, %arg179) {BlockSize.x = 256 : i32, GridSize.x = 576 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown90", memory_effects = [1 : i32, 2 : i32]} : memref<256x256x3x3xf16, "cuda">, memref<256x256x3x3xf32, "cuda">
    byre.compute @PTXOp(%26, %arg180) {BlockSize.x = 256 : i32, GridSize.x = 32 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown91", memory_effects = [1 : i32, 2 : i32]} : memref<256x128x1x1xf16, "cuda">, memref<256x128x1x1xf32, "cuda">
    byre.compute @PTXOp(%20, %arg187) {BlockSize.x = 256 : i32, GridSize.x = 576 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown90", memory_effects = [1 : i32, 2 : i32]} : memref<256x256x3x3xf16, "cuda">, memref<256x256x3x3xf32, "cuda">
    byre.compute @PTXOp(%18, %arg188) {BlockSize.x = 256 : i32, GridSize.x = 576 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown90", memory_effects = [1 : i32, 2 : i32]} : memref<256x256x3x3xf16, "cuda">, memref<256x256x3x3xf32, "cuda">
    byre.compute @PTXOp(%12, %arg193) {BlockSize.x = 256 : i32, GridSize.x = 1152 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown94", memory_effects = [1 : i32, 2 : i32]} : memref<512x256x3x3xf16, "cuda">, memref<512x256x3x3xf32, "cuda">
    byre.compute @PTXOp(%9, %arg194) {BlockSize.x = 256 : i32, GridSize.x = 2304 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown95", memory_effects = [1 : i32, 2 : i32]} : memref<512x512x3x3xf16, "cuda">, memref<512x512x3x3xf32, "cuda">
    byre.compute @PTXOp(%15, %arg195) {BlockSize.x = 256 : i32, GridSize.x = 128 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown96", memory_effects = [1 : i32, 2 : i32]} : memref<512x256x1x1xf16, "cuda">, memref<512x256x1x1xf32, "cuda">
    byre.compute @PTXOp(%6, %arg202) {BlockSize.x = 256 : i32, GridSize.x = 2304 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown95", memory_effects = [1 : i32, 2 : i32]} : memref<512x512x3x3xf16, "cuda">, memref<512x512x3x3xf32, "cuda">
    byre.compute @PTXOp(%4, %arg203) {BlockSize.x = 256 : i32, GridSize.x = 2304 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown95", memory_effects = [1 : i32, 2 : i32]} : memref<512x512x3x3xf16, "cuda">, memref<512x512x3x3xf32, "cuda">
    return
  }
}