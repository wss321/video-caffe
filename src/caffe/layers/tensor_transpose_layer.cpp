/*
 * tensor_transpose_layer.cpp
 *
 *  Created on: Nov 10, 2017
 *      Author: cfeng
 */

#include <vector>

#include "caffe/layers/tensor_transpose_layer.hpp"

namespace caffe {

template <typename Dtype>
void TensorTransposeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK(bottom[0]!=top[0]) << "TensorTranspose layer does not support in-place operation yet!";

  const TensorTransposeParameter& tt_param = this->layer_param_.tensor_transpose_param();
  const int naxes = bottom[0]->num_axes();
  vector<int> shp(1);
  shp[0] = naxes;
  this->order_.Reshape(shp);
  this->Xstride_.Reshape(shp);
  this->Ystride_.Reshape(shp);
  CHECK(naxes==tt_param.order_size()) << "Number of axes not equal to number of orders specified!";

  unsigned int* porder = this->order_.mutable_cpu_data();
  for(int n=naxes-1; n>=0; --n) {
    porder[n] = tt_param.order(n);
  }

  Reshape(bottom, top);
}

template <typename Dtype>
void TensorTransposeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int naxes = bottom[0]->num_axes();
  CHECK(naxes>=2)
    << "X blob must be of shape (B1,B2,...,Bn) where n>=2!";
  CHECK(naxes==this->order_.shape(0))
    << "bottom[0]'s number of axes not equal to number of orders specified!";

  const unsigned int* porder = this->order_.cpu_data();
  unsigned int* pXstride = this->Xstride_.mutable_cpu_data();
  unsigned int* pYstride = this->Ystride_.mutable_cpu_data();
  vector<int> top_shape = bottom[0]->shape();
  int acc_Xstride=1;
  int acc_Ystride=1;
  for(int n=naxes-1; n>=0; --n) {
    pXstride[n] = acc_Xstride;
    pYstride[n] = acc_Ystride;

    top_shape[n] = bottom[0]->shape(porder[n]);
    acc_Xstride *= bottom[0]->shape(n);
    acc_Ystride *= top_shape[n];
  }
  top[0]->Reshape(top_shape);
}

inline unsigned int indTranspose(int indY, const unsigned int*const strideY, const unsigned int*const strideX,
                        const unsigned int*const ordersY2X, const int naxes)
{
  unsigned int indX=0;
  for(size_t ny=0; ny<naxes; ++ny) {
    const unsigned int subY = indY / strideY[ny];
    indY -= subY*strideY[ny];
    indX += subY*strideX[ordersY2X[ny]];
  }
  return indX;
}

template <typename Dtype>
void TensorTransposeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  const Dtype* X_data = bottom[0]->cpu_data();
  Dtype* Y_data = top[0]->mutable_cpu_data();
  const unsigned int* order = this->order_.cpu_data();
  const unsigned int* Xstride = this->Xstride_.cpu_data();
  const unsigned int* Ystride = this->Ystride_.cpu_data();
  const int naxes = bottom[0]->num_axes();

  const int N = bottom[0]->count();
  for(int iy=0; iy<N; ++iy) {
    const unsigned int ix = indTranspose(iy, Ystride, Xstride, order, naxes);
    Y_data[iy] = X_data[ix];
  }
}

template <typename Dtype>
void TensorTransposeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  if (!propagate_down[0]) return;

  const Dtype* Y_diff = top[0]->cpu_diff();
  Dtype* X_diff = bottom[0]->mutable_cpu_diff();
  const unsigned int* order = this->order_.cpu_data();
  const unsigned int* Xstride = this->Xstride_.cpu_data();
  const unsigned int* Ystride = this->Ystride_.cpu_data();
  const int naxes = bottom[0]->num_axes();

  const int N = bottom[0]->count();
  for(int iy=0; iy<N; ++iy) {
    const unsigned int ix = indTranspose(iy, Ystride, Xstride, order, naxes);
    X_diff[ix] = Y_diff[iy];
  }
}

#ifdef CPU_ONLY
STUB_GPU(TensorTransposeLayer);
#endif

INSTANTIATE_CLASS(TensorTransposeLayer);
REGISTER_LAYER_CLASS(TensorTranspose);

}  // namespace caffe
