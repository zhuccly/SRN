// See docs in ../ops/image_ops.cc.

#define EIGEN_USE_THREADS

#include <vector>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

using namespace tensorflow;

REGISTER_OP("ExtractPatches").Input("input: float32")
                             .Input("input2: int32")
                             .Input("input3: float32")
                             .Output("output: float32")
                            .Doc(R"doc(Extract patches from the input image. output: [batch_size, num_patches, patch_height, patch_width, depth)doc");

void ExtractPatchesKernelLauncher(typename TTypes<float, 4>::ConstTensor input_images,
                                  typename TTypes<float, 3>::ConstTensor lms,
                                  typename TTypes<float, 5>::Tensor output_patches, 
                                  const int32  batch_size, 
                                  const int32  num_patches, 
                                  const int32  input_height,
                                  const int32  input_width,
                                  const int32  patch_height, 
                                  const int32  patch_width, 
                                  const int32  depth);

class ExtractPatchesOp: public OpKernel{
  public:
    explicit ExtractPatchesOp(OpKernelConstruction* context): OpKernel(context){}

    void Compute(OpKernelContext* context) override{
      const Tensor& input = context->input(0);
      const TensorShape input_shape = input.shape();
      const int32 num_dims = input_shape.dims();
      OP_REQUIRES(
        context, num_dims == 4, 
        errors::InvalidArgument(
          "input must be 4-dimensional (batch_size, height, width, depth)",
          input_shape.DebugString()));

      const int32 batch_size = input_shape.dim_size(0);
      const int32 input_height = input_shape.dim_size(1);
      const int32 input_width = input_shape.dim_size(2);
      const int32 depth = input_shape.dim_size(3);

      const Tensor& window_size = context->input(1);
      OP_REQUIRES(context, (window_size.shape().dims() == 1) && window_size.shape().dim_size(0) == 2,
                  errors::InvalidArgument(
                    "patch shape must be a vector of size 2 (height, width)",
                    window_size.shape().DebugString()));

      const int32 patch_height = window_size.tensor<int, 1>()(0);
      const int32 patch_width  = window_size.tensor<int, 1>()(1);

      const Tensor& offsets = context->input(2);
      OP_REQUIRES(context, offsets.shape().dims() == 3,
                  errors::InvalidArgument(
                    "input must be a tensor [batch_size, num_patches, 2]",
                    offsets.shape().DebugString()));
      OP_REQUIRES(context, offsets.shape().dim_size(0) == batch_size,
                  errors::InvalidArgument(
                    "first dimension should be batch",
                    offsets.shape().DebugString()));
      OP_REQUIRES(context, offsets.shape().dim_size(2) == 2,
                  errors::InvalidArgument(
                    "third dimension should be of size 2 (y,x)",
                    offsets.shape().DebugString()));

      auto num_patches = offsets.shape().dim_size(1);
      TensorShape output_shape({batch_size, num_patches, patch_height, patch_width, depth});
      Tensor* output = nullptr;

      OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));

      if(output->NumElements() == 0){
        return;
      }

      typename TTypes<float, 5>::Tensor output_patches = output->tensor<float, 5>();
      typename TTypes<float, 4>::ConstTensor input_images = input.tensor<float, 4>();
      typename TTypes<float, 3>::ConstTensor lms = offsets.tensor<float, 3>();
      ExtractPatchesKernelLauncher(input_images,
                                   lms, 
                                   output_patches, 
                                   batch_size, 
                                   num_patches, 
                                   input_height,
                                   input_width,
                                   patch_height, 
                                   patch_width, 
                                   depth);
    }
};
REGISTER_KERNEL_BUILDER(Name("ExtractPatches").Device(DEVICE_GPU), ExtractPatchesOp);