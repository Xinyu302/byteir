// RUN: byteir-translate %s --mlir-to-llvmir | FileCheck %s

// CHECK-LABEL: define void @_mlir_ciface_Unknown

module attributes {byre.container_module} {
  llvm.func @Unknown0(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: !llvm.ptr, %arg4: !llvm.ptr, %arg5: i64, %arg6: !llvm.ptr, %arg7: !llvm.ptr, %arg8: i64, %arg9: i64, %arg10: i64, %arg11: i64, %arg12: i64) attributes {__byre__kernel_name = "Unknown0", __byre__llvm_file_name = "host_kernels.ll", __byteir_hlo_aggressive_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "LLVMJITOp", byre_force_compute_name, llvm.emit_c_interface} {
    %0 = llvm.mlir.constant(97 : index) : i64
    %1 = llvm.mlir.constant(0 : index) : i64
    %2 = llvm.mlir.constant(1 : index) : i64
    %3 = llvm.mlir.constant(1103515245 : i32) : i32
    %4 = llvm.mlir.constant(12345 : i32) : i32
    %5 = llvm.mlir.constant(3528531795 : i64) : i64
    %6 = llvm.mlir.constant(3449720151 : i64) : i64
    %7 = llvm.mlir.constant(32 : i64) : i64
    %8 = llvm.mlir.constant(-1640531527 : i32) : i32
    %9 = llvm.mlir.constant(-1150833019 : i32) : i32
    %10 = llvm.mlir.constant(2.32830644E-10 : f32) : f32
    %11 = llvm.mlir.constant(1.16415322E-10 : f32) : f32
    %12 = llvm.mlir.constant(-2.000000e+00 : f32) : f32
    %13 = llvm.mlir.constant(1013904242 : i32) : i32
    %14 = llvm.mlir.constant(1993301258 : i32) : i32
    %15 = llvm.mlir.constant(-626627285 : i32) : i32
    %16 = llvm.mlir.constant(842468239 : i32) : i32
    %17 = llvm.mlir.constant(2027808484 : i32) : i32
    %18 = llvm.mlir.constant(-308364780 : i32) : i32
    %19 = llvm.mlir.constant(387276957 : i32) : i32
    %20 = llvm.mlir.constant(-1459197799 : i32) : i32
    %21 = llvm.mlir.constant(-1253254570 : i32) : i32
    %22 = llvm.mlir.constant(1684936478 : i32) : i32
    %23 = llvm.mlir.constant(1401181199 : i32) : i32
    %24 = llvm.mlir.constant(534103459 : i32) : i32
    %25 = llvm.mlir.constant(-616729560 : i32) : i32
    %26 = llvm.mlir.constant(-1879881855 : i32) : i32
    %27 = llvm.mlir.constant(6.283185 : f32) : f32
    %28 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    llvm.br ^bb1(%1 : i64)
  ^bb1(%29: i64):  // 2 preds: ^bb0, ^bb2
    %30 = llvm.icmp "slt" %29, %0 : i64
    llvm.cond_br %30, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %31 = llvm.load %arg1 : !llvm.ptr -> i64
    %32 = llvm.load %arg4 : !llvm.ptr -> i64
    %33 = llvm.trunc %31 : i64 to i32
    %34 = llvm.trunc %32 : i64 to i32
    %35 = llvm.add %33, %34  : i32
    %36 = llvm.mul %35, %3  : i32
    %37 = llvm.add %36, %4  : i32
    %38 = llvm.trunc %29 : i64 to i32
    %39 = llvm.add %38, %37  : i32
    %40 = llvm.mul %39, %3  : i32
    %41 = llvm.add %40, %4  : i32
    %42 = llvm.zext %41 : i32 to i64
    %43 = llvm.mul %42, %5  : i64
    %44 = llvm.trunc %43 : i64 to i32
    %45 = llvm.lshr %43, %7  : i64
    %46 = llvm.trunc %45 : i64 to i32
    %47 = llvm.xor %46, %34  : i32
    %48 = llvm.add %33, %8  : i32
    %49 = llvm.add %34, %9  : i32
    %50 = llvm.zext %33 : i32 to i64
    %51 = llvm.zext %47 : i32 to i64
    %52 = llvm.mul %50, %5  : i64
    %53 = llvm.trunc %52 : i64 to i32
    %54 = llvm.lshr %52, %7  : i64
    %55 = llvm.trunc %54 : i64 to i32
    %56 = llvm.mul %51, %6  : i64
    %57 = llvm.trunc %56 : i64 to i32
    %58 = llvm.lshr %56, %7  : i64
    %59 = llvm.trunc %58 : i64 to i32
    %60 = llvm.xor %59, %48  : i32
    %61 = llvm.xor %55, %44  : i32
    %62 = llvm.xor %61, %49  : i32
    %63 = llvm.add %33, %13  : i32
    %64 = llvm.add %34, %14  : i32
    %65 = llvm.zext %60 : i32 to i64
    %66 = llvm.zext %62 : i32 to i64
    %67 = llvm.mul %65, %5  : i64
    %68 = llvm.trunc %67 : i64 to i32
    %69 = llvm.lshr %67, %7  : i64
    %70 = llvm.trunc %69 : i64 to i32
    %71 = llvm.mul %66, %6  : i64
    %72 = llvm.trunc %71 : i64 to i32
    %73 = llvm.lshr %71, %7  : i64
    %74 = llvm.trunc %73 : i64 to i32
    %75 = llvm.xor %74, %57  : i32
    %76 = llvm.xor %75, %63  : i32
    %77 = llvm.xor %70, %53  : i32
    %78 = llvm.xor %77, %64  : i32
    %79 = llvm.add %33, %15  : i32
    %80 = llvm.add %34, %16  : i32
    %81 = llvm.zext %76 : i32 to i64
    %82 = llvm.zext %78 : i32 to i64
    %83 = llvm.mul %81, %5  : i64
    %84 = llvm.trunc %83 : i64 to i32
    %85 = llvm.lshr %83, %7  : i64
    %86 = llvm.trunc %85 : i64 to i32
    %87 = llvm.mul %82, %6  : i64
    %88 = llvm.trunc %87 : i64 to i32
    %89 = llvm.lshr %87, %7  : i64
    %90 = llvm.trunc %89 : i64 to i32
    %91 = llvm.xor %90, %72  : i32
    %92 = llvm.xor %91, %79  : i32
    %93 = llvm.xor %86, %68  : i32
    %94 = llvm.xor %93, %80  : i32
    %95 = llvm.add %33, %17  : i32
    %96 = llvm.add %34, %18  : i32
    %97 = llvm.zext %92 : i32 to i64
    %98 = llvm.zext %94 : i32 to i64
    %99 = llvm.mul %97, %5  : i64
    %100 = llvm.trunc %99 : i64 to i32
    %101 = llvm.lshr %99, %7  : i64
    %102 = llvm.trunc %101 : i64 to i32
    %103 = llvm.mul %98, %6  : i64
    %104 = llvm.trunc %103 : i64 to i32
    %105 = llvm.lshr %103, %7  : i64
    %106 = llvm.trunc %105 : i64 to i32
    %107 = llvm.xor %106, %88  : i32
    %108 = llvm.xor %107, %95  : i32
    %109 = llvm.xor %102, %84  : i32
    %110 = llvm.xor %109, %96  : i32
    %111 = llvm.add %33, %19  : i32
    %112 = llvm.add %34, %20  : i32
    %113 = llvm.zext %108 : i32 to i64
    %114 = llvm.zext %110 : i32 to i64
    %115 = llvm.mul %113, %5  : i64
    %116 = llvm.trunc %115 : i64 to i32
    %117 = llvm.lshr %115, %7  : i64
    %118 = llvm.trunc %117 : i64 to i32
    %119 = llvm.mul %114, %6  : i64
    %120 = llvm.trunc %119 : i64 to i32
    %121 = llvm.lshr %119, %7  : i64
    %122 = llvm.trunc %121 : i64 to i32
    %123 = llvm.xor %122, %104  : i32
    %124 = llvm.xor %123, %111  : i32
    %125 = llvm.xor %118, %100  : i32
    %126 = llvm.xor %125, %112  : i32
    %127 = llvm.add %33, %21  : i32
    %128 = llvm.add %34, %22  : i32
    %129 = llvm.zext %124 : i32 to i64
    %130 = llvm.zext %126 : i32 to i64
    %131 = llvm.mul %129, %5  : i64
    %132 = llvm.trunc %131 : i64 to i32
    %133 = llvm.lshr %131, %7  : i64
    %134 = llvm.trunc %133 : i64 to i32
    %135 = llvm.mul %130, %6  : i64
    %136 = llvm.trunc %135 : i64 to i32
    %137 = llvm.lshr %135, %7  : i64
    %138 = llvm.trunc %137 : i64 to i32
    %139 = llvm.xor %138, %120  : i32
    %140 = llvm.xor %139, %127  : i32
    %141 = llvm.xor %134, %116  : i32
    %142 = llvm.xor %141, %128  : i32
    %143 = llvm.add %33, %23  : i32
    %144 = llvm.add %34, %24  : i32
    %145 = llvm.zext %140 : i32 to i64
    %146 = llvm.zext %142 : i32 to i64
    %147 = llvm.mul %145, %5  : i64
    %148 = llvm.trunc %147 : i64 to i32
    %149 = llvm.lshr %147, %7  : i64
    %150 = llvm.trunc %149 : i64 to i32
    %151 = llvm.mul %146, %6  : i64
    %152 = llvm.lshr %151, %7  : i64
    %153 = llvm.trunc %152 : i64 to i32
    %154 = llvm.xor %153, %136  : i32
    %155 = llvm.xor %154, %143  : i32
    %156 = llvm.xor %150, %132  : i32
    %157 = llvm.xor %156, %144  : i32
    %158 = llvm.add %34, %25  : i32
    %159 = llvm.zext %155 : i32 to i64
    %160 = llvm.zext %157 : i32 to i64
    %161 = llvm.mul %159, %5  : i64
    %162 = llvm.lshr %161, %7  : i64
    %163 = llvm.trunc %162 : i64 to i32
    %164 = llvm.mul %160, %6  : i64
    %165 = llvm.trunc %164 : i64 to i32
    %166 = llvm.xor %163, %148  : i32
    %167 = llvm.xor %166, %158  : i32
    %168 = llvm.add %33, %26  : i32
    %169 = llvm.zext %167 : i32 to i64
    %170 = llvm.mul %169, %6  : i64
    %171 = llvm.trunc %170 : i64 to i32
    %172 = llvm.lshr %170, %7  : i64
    %173 = llvm.trunc %172 : i64 to i32
    %174 = llvm.xor %173, %165  : i32
    %175 = llvm.xor %174, %168  : i32
    %176 = llvm.uitofp %175 : i32 to f32
    %177 = llvm.fmul %176, %10  : f32
    %178 = llvm.fadd %177, %11  : f32
    %179 = llvm.uitofp %171 : i32 to f32
    %180 = llvm.fmul %179, %10  : f32
    %181 = llvm.fadd %180, %11  : f32
    %182 = llvm.intr.log(%178)  : (f32) -> f32
    %183 = llvm.fmul %182, %12  : f32
    %184 = llvm.intr.sqrt(%183)  : (f32) -> f32
    %185 = llvm.fmul %181, %27  : f32
    %186 = llvm.intr.cos(%185)  : (f32) -> f32
    %187 = llvm.fmul %184, %186  : f32
    %188 = llvm.fadd %187, %28  : f32
    %189 = llvm.mul %1, %0  : i64
    %190 = llvm.add %189, %29  : i64
    %191 = llvm.getelementptr %arg7[%190] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %188, %191 : f32, !llvm.ptr
    %192 = llvm.add %29, %2  : i64
    llvm.br ^bb1(%192 : i64)
  ^bb3:  // pred: ^bb1
    llvm.return
  }
  llvm.func @_mlir_ciface_Unknown0(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) attributes {__byre__kernel_name = "Unknown0", __byre__llvm_file_name = "host_kernels.ll", __byteir_hlo_aggressive_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "LLVMJITOp", byre_force_compute_name, llvm.emit_c_interface} {
    %0 = llvm.load %arg0 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64)>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(ptr, ptr, i64)> 
    %2 = llvm.extractvalue %0[1] : !llvm.struct<(ptr, ptr, i64)> 
    %3 = llvm.extractvalue %0[2] : !llvm.struct<(ptr, ptr, i64)> 
    %4 = llvm.load %arg1 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64)>
    %5 = llvm.extractvalue %4[0] : !llvm.struct<(ptr, ptr, i64)> 
    %6 = llvm.extractvalue %4[1] : !llvm.struct<(ptr, ptr, i64)> 
    %7 = llvm.extractvalue %4[2] : !llvm.struct<(ptr, ptr, i64)> 
    %8 = llvm.load %arg2 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %9 = llvm.extractvalue %8[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %10 = llvm.extractvalue %8[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %11 = llvm.extractvalue %8[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %12 = llvm.extractvalue %8[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %13 = llvm.extractvalue %8[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %14 = llvm.extractvalue %8[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %15 = llvm.extractvalue %8[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.call @Unknown0(%1, %2, %3, %5, %6, %7, %9, %10, %11, %12, %13, %14, %15) : (!llvm.ptr, !llvm.ptr, i64, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64) -> ()
    llvm.return
  }
}