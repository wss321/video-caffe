//
// Created by wss on 2021/10/18.
//

#ifndef CAFFE_BASE_MULTI_DATA_LAYER_HPP
#define CAFFE_BASE_MULTI_DATA_LAYER_HPP
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/blocking_queue.hpp"

namespace caffe {

/**
 * @brief Provides base for data layers that feed blobs to the Net.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
    template <typename Dtype>
    class BaseMultiDataLayer : public Layer<Dtype> {
    public:
        explicit BaseMultiDataLayer(const LayerParameter& param);
        // LayerSetUp: implements common data layer setup functionality, and calls
        // DataLayerSetUp to do special data layer setup for individual layer types.
        // This method may not be overridden except by the BasePrefetchingDataLayer.
        virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                const vector<Blob<Dtype>*>& top);
        virtual void MultiDataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                    const vector<Blob<Dtype>*>& top) {}
        // Data layers have no bottoms, so reshaping is trivial.
        virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top) {}

        virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                                  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
        virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                                  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}

    protected:
        TransformationParameter transform_param_;
        shared_ptr<DataTransformer<Dtype> > data_transformer_;
        int num_data_;
    };

    template <typename Dtype>
    class Batch {
    public:
        explicit Batch(int num=1){
            data_.resize(num);
        }
        ~Batch(){
            for (int i = 0; i < data_.size(); ++i) {
                delete data_[i];
            }
        }
//        vector<shared_ptr<Blob<Dtype>>> data_;
        vector<Blob<Dtype>*> data_;
    };

    template <typename Dtype>
    class BaseMultiPrefetchingDataLayer :
            public BaseMultiDataLayer<Dtype>, public InternalThread {
    public:
        explicit BaseMultiPrefetchingDataLayer(const LayerParameter& param);
        // LayerSetUp: implements common data layer setup functionality, and calls
        // DataLayerSetUp to do special data layer setup for individual layer types.
        // This method may not be overridden.
        ~BaseMultiPrefetchingDataLayer();
        void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                        const vector<Blob<Dtype>*>& top);

        virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                 const vector<Blob<Dtype>*>& top);
        virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                 const vector<Blob<Dtype>*>& top);

    protected:
        virtual void InternalThreadEntry();
        virtual void load_batch(Batch<Dtype>* batch) = 0;

        vector<Batch<Dtype>*> prefetch_;
        BlockingQueue<Batch<Dtype>*> prefetch_free_;
        BlockingQueue<Batch<Dtype>*> prefetch_full_;
        Batch<Dtype>* prefetch_current_;

        Blob<Dtype> transformed_data_;
    };

}  // namespace caffe

#endif //CAFFE_BASE_MULTI_DATA_LAYER_HPP
