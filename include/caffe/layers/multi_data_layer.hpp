//
// Created by wss on 2021/10/19.
//

#ifndef CAFFE_MULTI_DATA_LAYER_HPP
#define CAFFE_MULTI_DATA_LAYER_HPP

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_multi_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"

namespace caffe {

    template <typename Dtype>
    class MultiDataLayer : public BaseMultiPrefetchingDataLayer<Dtype> {
    public:
        explicit MultiDataLayer(const LayerParameter& param);
        virtual ~MultiDataLayer();
        virtual void MultiDataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                    const vector<Blob<Dtype>*>& top);
        virtual inline const char* type() const { return "MultiData"; }
        virtual inline int ExactNumBottomBlobs() const { return 0; }
        virtual inline int MinTopBlobs() const { return 1; }
        virtual inline int MaxTopBlobs() const { return 1000; }

    protected:
        void Next();
        bool Skip();
        virtual void load_batch(Batch<Dtype>* batch);

        shared_ptr<db::DB> db_;
        shared_ptr<db::Cursor> cursor_;
        uint64_t offset_;
    };

}  // namespace caffe

#endif //CAFFE_MULTI_DATA_LAYER_HPP
