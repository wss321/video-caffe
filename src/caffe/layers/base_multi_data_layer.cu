#include <vector>

#include "caffe/layers/base_multi_data_layer.hpp"

namespace caffe {

    template <typename Dtype>
    void BaseMultiPrefetchingDataLayer<Dtype>::Forward_gpu(
            const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
        if (prefetch_current_) {
            prefetch_free_.push(prefetch_current_);
        }
        prefetch_current_ = prefetch_full_.pop("Waiting for data");
        // Reshape to loaded data.
        for (int i = 0; i < top.size(); ++i) {
//            top[i]->ReshapeLike(*prefetch_current_->data_[i]);
            top[i]->Reshape(prefetch_current_->data_[i]->shape());
            top[i]->set_gpu_data(prefetch_current_->data_[i]->mutable_gpu_data());
        }
    }

    INSTANTIATE_LAYER_GPU_FORWARD(BaseMultiPrefetchingDataLayer);

}  // namespace caffe
