/*
 * tensor_transpose_layer.cu
 *
 *  Created on: Nov 11, 2017
 *      Author: cfeng
 */

#include <vector>

#include "caffe/layers/tensor_transpose_layer.hpp"

namespace caffe {

template<typename Dtype>
__global__ void tensor_transpose_forward(
    const Dtype* const X,
    const unsigned int*const strideY, const unsigned int*const strideX,
    const unsigned int*const ordersY2X, const int naxes,
    const int N,
    Dtype* const Y)
{
  CUDA_KERNEL_LOOP(iy, N) //for each element in Y
  {
    unsigned int indY=iy;
    unsigned int indX=0;
    for(int ny=0; ny<naxes; ++ny) {
      const unsigned int subY = indY / strideY[ny];
      indY -= subY*strideY[ny];
      indX += subY*strideX[ordersY2X[ny]];
    }
    Y[iy]=X[indX];
  }
}

template<typename Dtype>
__global__ void tensor_transpose_backward(
    const Dtype* const dY,
    const unsigned int*const strideY, const unsigned int*const strideX,
    const unsigned int*const ordersY2X, const int naxes,
    const int N,
    Dtype* const dX)
{
  CUDA_KERNEL_LOOP(iy, N) //for each element in Y
  {
    unsigned int indY=iy;
    unsigned int indX=0;
    for(int ny=0; ny<naxes; ++ny) {
      const unsigned int subY = indY / strideY[ny];
      indY -= subY*strideY[ny];
      indX += subY*strideX[ordersY2X[ny]];
    }
    dX[indX]=dY[iy];
  }
}

//////////////////////////////////////////////////////////////////////////////

template <typename Dtype>
void TensorTransposeLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  const Dtype* X_data = bottom[0]->gpu_data();
  Dtype* Y_data = top[0]->mutable_gpu_data();
  const unsigned int* order = this->order_.gpu_data();
  const unsigned int* Xstride = this->Xstride_.gpu_data();
  const unsigned int* Ystride = this->Ystride_.gpu_data();
  const int naxes = bottom[0]->num_axes();

  const int N = bottom[0]->count();
  tensor_transpose_forward<Dtype><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      X_data,
      Ystride, Xstride, order, naxes, N,
      Y_data
  );

  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void TensorTransposeLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  if (!propagate_down[0]) return;

  const Dtype* Y_diff = top[0]->gpu_diff();
  Dtype* X_diff = bottom[0]->mutable_gpu_diff();
  const unsigned int* order = this->order_.gpu_data();
  const unsigned int* Xstride = this->Xstride_.gpu_data();
  const unsigned int* Ystride = this->Ystride_.gpu_data();
  const int naxes = bottom[0]->num_axes();

  const int N = bottom[0]->count();
  tensor_transpose_backward<Dtype><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      Y_diff,
      Ystride, Xstride, order, naxes, N,
      X_diff
  );

  CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(TensorTransposeLayer);

}  // namespace caffe
