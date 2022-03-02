/**
* Copyright 2021 wss
* Created by wss on 3月,02, 2022
*/
#include <cfloat>
#include <vector>

#include "caffe/layers/detach_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
template <typename Dtype>
void DetachLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    caffe_copy(bottom[0]->count(), bottom[0]->cpu_data(),
        top[0]->mutable_gpu_data());
}

template <typename Dtype>
void DetachLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
}

INSTANTIATE_LAYER_GPU_FUNCS(DetachLayer);

}  // namespace caffe
