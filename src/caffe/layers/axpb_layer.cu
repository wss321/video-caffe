#include <cfloat>
#include <vector>

#include "caffe/layers/axpb_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
template <typename Dtype>
__global__ void axpb_forward_kernel(const int n, const Dtype* in,
    const Dtype a, const Dtype b, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index] * a + b;
  }
}

template <typename Dtype>
__global__ void axpb_backward_kernel(const int n, const Dtype* top_diff,
                                    const Dtype a,
                                    Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    bottom_diff[index] = top_diff[index] * a;
  }
}


template <typename Dtype>
void AxpbLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  if (bottom[0] != top[0]) {
  caffe_copy(bottom[0]->count(), bottom[0]->cpu_data(),
      top[0]->mutable_gpu_data());
  }
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = bottom[0]->mutable_gpu_data();
  const int count = top[0]->count();
  axpb_forward_kernel<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
          <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          count, bottom_data, a_, b_, top_data);
  }

template <typename Dtype>
void AxpbLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const int count = top[0]->count();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    axpb_backward_kernel<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
            <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            count, top_diff, a_, bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(AxpbLayer);

}  // namespace caffe
