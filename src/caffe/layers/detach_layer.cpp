/**
* Copyright 2021 wss
* Created by wss on 3月,02, 2022
*/
#include <algorithm>
#include <vector>

#include "caffe/layers/detach_layer.hpp"

namespace caffe {

template<typename Dtype>
void DetachLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
const vector<Blob<Dtype> *> &top) {
  this->param_propagate_down_.resize(this->blobs_.size(), false);
}

template<typename Dtype>
void DetachLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
const vector<Blob<Dtype> *> &top) {
  top[0]->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void DetachLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
const vector<Blob<Dtype>*>& top) {
  caffe_copy(bottom[0]->count(), bottom[0]->cpu_data(),
      top[0]->mutable_cpu_data());
}

template <typename Dtype>
void DetachLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
const vector<bool>& propagate_down,
const vector<Blob<Dtype>*>& bottom) {
}


#ifdef CPU_ONLY
STUB_GPU(DetachLayer);
#endif

INSTANTIATE_CLASS(DetachLayer);
REGISTER_LAYER_CLASS(Detach);
}  // namespace caffe
