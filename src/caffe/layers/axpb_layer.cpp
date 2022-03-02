#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/layers/axpb_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype>
void AxpbLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                  const vector<Blob<Dtype> *> &top) {
  const AxpbParameter &param = this->layer_param_.axpb_param();
  a_ = param.a();
  b_ = param.b();

  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template<typename Dtype>
void AxpbLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
                               const vector<Blob<Dtype> *> &top) {
  top[0]->ReshapeLike(*bottom[0]);
}

template<typename Dtype>
void AxpbLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  if (bottom[0] != top[0]) {
    caffe_copy(bottom[0]->count(), bottom[0]->cpu_data(),
               top[0]->mutable_cpu_data());
  }
  Dtype *top_data = top[0]->mutable_cpu_data();
  for (int n = 0; n < bottom[0]->count(); ++n) {
    top_data[n] = top_data[n] * a_ + b_;
  }
}

template<typename Dtype>
void AxpbLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype> *> &top,
                                    const vector<bool> &propagate_down,
                                    const vector<Blob<Dtype> *> &bottom) {
  if (propagate_down[0]) {
    const Dtype *top_diff = top[0]->cpu_diff();
    Dtype *bottom_diff = bottom[0]->mutable_cpu_diff();
    for (int n = 0; n < bottom[0]->count(); ++n) {
      bottom_diff[n] = top_diff[n] * a_;
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(AxpbLayer);
#endif

INSTANTIATE_CLASS(AxpbLayer);
REGISTER_LAYER_CLASS(Axpb);

}  // namespace caffe
