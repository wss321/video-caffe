#include <boost/thread.hpp>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_multi_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/blocking_queue.hpp"

namespace caffe {

    template <typename Dtype>
    BaseMultiDataLayer<Dtype>::BaseMultiDataLayer(const LayerParameter& param)
            : Layer<Dtype>(param),
              transform_param_(param.transform_param()),
              transform_data_index_(param.multi_data_param().transform_data_index()),
              num_data_(param.multi_data_param().num_data()){
//                  int label_index = param.multi_data_param().label_index();
//        if (label_index<0){
//            CHECK_GT(num_data_ + label_index, -1)<<"num_data + label_index must not less than 0";
//            label_index_=num_data_ + label_index;
//        }

    }

    template <typename Dtype>
    void BaseMultiDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                          const vector<Blob<Dtype>*>& top) {
        if (top.size() < num_data_) {
            output_labels_ = false;
        } else {
            output_labels_ = true;
        }
        data_transformer_.reset(
                new DataTransformer<Dtype>(transform_param_, this->phase_));
        data_transformer_->InitRand();
        // The subclasses should setup the size of bottom and top
        MultiDataLayerSetUp(bottom, top);
    }

    template <typename Dtype>
    BaseMultiPrefetchingDataLayer<Dtype>::BaseMultiPrefetchingDataLayer(
            const LayerParameter& param)
            : BaseMultiDataLayer<Dtype>(param),
              prefetch_(param.multi_data_param().prefetch()),
              prefetch_free_(), prefetch_full_(), prefetch_current_() {
        for (int i = 0; i < prefetch_.size(); ++i) {
//            prefetch_[i].reset(new Batch<Dtype>());
            prefetch_[i]=new Batch<Dtype>();
            prefetch_free_.push(prefetch_[i]);
        }
    }
    template <typename Dtype>
    BaseMultiPrefetchingDataLayer<Dtype>::~BaseMultiPrefetchingDataLayer(){
        for (int i = 0; i < prefetch_.size(); ++i) {
//            for (int j = 0; j < prefetch_[i]->data_.size(); ++j) {
//                delete prefetch_[i]->data_[j];
//            }
            delete prefetch_[i];
        }
    }

    template <typename Dtype>
    void BaseMultiPrefetchingDataLayer<Dtype>::LayerSetUp(
            const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
        BaseMultiDataLayer<Dtype>::LayerSetUp(bottom, top);

        // Before starting the prefetch thread, we make cpu_data and gpu_data
        // calls so that the prefetch thread does not accidentally make simultaneous
        // cudaMalloc calls when the main thread is running. In some GPUs this
        // seems to cause failures if we do not so.

        for (int i = 0; i < prefetch_.size(); ++i){
            for (int j = 0; j < prefetch_[i]->data_.size(); ++j){
                prefetch_[i]->data_[j]->mutable_cpu_data();
            }

        }


#ifndef CPU_ONLY
        if (Caffe::mode() == Caffe::GPU) {
            for (int i = 0; i < prefetch_.size(); ++i){
                for (int j = 0; j < prefetch_[i]->data_.size(); ++j)
                    prefetch_[i]->data_[j]->mutable_gpu_data();
            }
        }
#endif
        DLOG(INFO) << "Initializing prefetch";
        this->data_transformer_->InitRand();
        StartInternalThread();
        DLOG(INFO) << "Prefetch initialized.";
    }

    template <typename Dtype>
    void BaseMultiPrefetchingDataLayer<Dtype>::InternalThreadEntry() {
#ifndef CPU_ONLY
        cudaStream_t stream;
        if (Caffe::mode() == Caffe::GPU) {
            CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        }
#endif

        try {
            while (!must_stop()) {
                Batch<Dtype>* batch = prefetch_free_.pop();
                load_batch(batch);
#ifndef CPU_ONLY
                if (Caffe::mode() == Caffe::GPU) {
                    for (int i = 0; i < batch->data_.size(); ++i) {
                        batch->data_[i]->data()->async_gpu_push(stream);
                    }
                    CUDA_CHECK(cudaStreamSynchronize(stream));
                }
#endif
                prefetch_full_.push(batch);
            }
        } catch (boost::thread_interrupted&) {
            // Interrupted exception is expected on shutdown
        }
#ifndef CPU_ONLY
        if (Caffe::mode() == Caffe::GPU) {
            CUDA_CHECK(cudaStreamDestroy(stream));
        }
#endif
    }

    template <typename Dtype>
    void BaseMultiPrefetchingDataLayer<Dtype>::Forward_cpu(
            const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
        if (prefetch_current_) {
            prefetch_free_.push(prefetch_current_);
        }
        prefetch_current_ = prefetch_full_.pop("Waiting for data");
        // Reshape to loaded data.
        for (int i = 0; i < top.size(); ++i) {
//            top[i]->ReshapeLike(*prefetch_current_->data_[i].get());
            top[i]->Reshape(prefetch_current_->data_[i]->shape());
            top[i]->set_cpu_data(prefetch_current_->data_[i]->mutable_cpu_data());
        }
    }

#ifdef CPU_ONLY
    STUB_GPU_FORWARD(BasePrefetchingDataLayer, Forward);
#endif

    INSTANTIATE_CLASS(BaseMultiDataLayer);
    INSTANTIATE_CLASS(BaseMultiPrefetchingDataLayer);

}  // namespace caffe
