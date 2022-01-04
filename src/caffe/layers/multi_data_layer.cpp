#ifdef USE_OPENCV

#include <opencv2/core/core.hpp>

#endif  // USE_OPENCV

#include <stdint.h>

#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/multi_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"

namespace caffe {

    template<typename Dtype>
    MultiDataLayer<Dtype>::MultiDataLayer(const LayerParameter &param)
            : BaseMultiPrefetchingDataLayer<Dtype>(param),
              num_data_(param.multi_data_param().num_data()), shuffle_(param.multi_data_param().shuffle()) {
        vector<MultiDataParameter::DB> backends;
        for (int i = 0; i < param.multi_data_param().backend().size(); ++i) {
            backends.push_back((MultiDataParameter::DB) param.multi_data_param().backend()[i]);
        }

        vector<string> sources;
        for (int i = 0; i < param.multi_data_param().source().size(); ++i) {
            sources.push_back(param.multi_data_param().source()[i]);
        }

        CHECK_EQ(backends.size(), num_data_) << "number of backend must be equal to number of dataset";
        CHECK_EQ(sources.size(), num_data_) << "number of backend must be equal to number of dataset";
        CHECK_EQ(param.multi_data_param().shape().size(), num_data_)
            << "size of shape must be equal to number of dataset";
        dbs_.resize(num_data_);
        cursors_.resize(num_data_);
        shapes_.resize(num_data_);
        for (int i = 0; i < num_data_; ++i) {
            dbs_[i].reset(db::GetDB((MultiDataParameter_DB) backends[i]));
            dbs_[i]->Open(sources[i], db::READ);
            cursors_[i].reset(dbs_[i]->NewCursor());
            shapes_[i] = param.multi_data_param().shape()[i];
        }

    }

    template<typename Dtype>
    MultiDataLayer<Dtype>::~MultiDataLayer() {
        this->StopInternalThread();
    }

    template<typename Dtype>
    void MultiDataLayer<Dtype>::MultiDataLayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                                    const vector<Blob<Dtype> *> &top) {
        const int batch_size = this->layer_param_.multi_data_param().batch_size();
        // Read a data point, and use it to initialize the top blob.
        CHECK_EQ(top.size(), num_data_) << "top size must be equal to number of dataset";
        offsets_.resize(num_data_);

        for (int d = 0; d < num_data_; ++d) {
            Datum datum;
            datum.ParseFromString(cursors_[d]->value());

            vector<int> top_shape(shapes_[d].dim_size() + 1);
            for (int dim_i = 0; dim_i < shapes_[d].dim_size(); ++dim_i) {
                top_shape[dim_i + 1] = shapes_[d].dim(dim_i);
            }
            top_shape[0] = batch_size;
            top[d]->Reshape(top_shape);
            for (int i = 0; i < this->prefetch_.size(); ++i) {
                this->prefetch_[i]->data_.resize(num_data_);
                for (int j = 0; j < this->prefetch_[i]->data_.size(); ++j) {
                    this->prefetch_[i]->data_[j] = new Blob<Dtype>();
                    this->prefetch_[i]->data_[j]->Reshape(top_shape);
                }

            }

        }

    }

    template<typename Dtype>
    bool MultiDataLayer<Dtype>::Skip() {
        int size = Caffe::solver_count();
        int rank = Caffe::solver_rank();
        bool keep = (offsets_[0] % size) == rank ||
                    // In test mode, only rank 0 runs, so avoid skipping
                    this->layer_param_.phase() == TEST;
        return !keep;
    }

    template<typename Dtype>
    void MultiDataLayer<Dtype>::Next() {
        if (!shuffle_) {
            for (int i = 0; i < num_data_; ++i) {
                cursors_[i]->Next();
                if (!cursors_[i]->valid()) {
                    LOG_IF(INFO, Caffe::root_solver())
                    << "Restarting data prefetching from start.";
                    cursors_[i]->SeekToFirst();

                offsets_[i]++;
                }
            }
        }else {
            int index = caffe::caffe_rng_rand() % dbs_[0]->GetDBLength();
            string key = dbs_[0]->GetKey(index);
            for (int i = 0; i < num_data_; ++i){
                cursors_[i]->SeekToKey(key);
                offsets_[i] = index + 1;
            }
        }

    }

// This function is called on prefetch thread
    template<typename Dtype>
    void MultiDataLayer<Dtype>::load_batch(Batch<Dtype> *batch) {
        CPUTimer batch_timer;
        batch_timer.Start();
        double read_time = 0;
        CPUTimer timer;
        for (int i = 0; i < batch->data_.size(); ++i) {
            CHECK(batch->data_[i]->count());
        }

        const int batch_size = this->layer_param_.multi_data_param().batch_size();

        Datum datum;
        for (int item_id = 0; item_id < batch_size; ++item_id) {
            timer.Start();
            while (Skip()) {
                Next();
            }
            if (shuffle_) Next(); // shuffle first
            for (int d = 0; d < num_data_; ++d) {
                datum.ParseFromString(cursors_[d]->value());
                read_time += timer.MicroSeconds();

                if (item_id == 0) {
                    // Reshape according to the first datum of each batch
                    // on single input batches allows for inputs of varying dimension.
                    // Use data_transformer to infer the expected blob shape from datum.
                    vector<int> top_shape(shapes_[d].dim_size() + 1);
                    for (int dim_i = 0; dim_i < shapes_[d].dim_size(); ++dim_i) {
                        top_shape[dim_i + 1] = shapes_[d].dim(dim_i);
                    }
                    top_shape[0] = batch_size;
                    batch->data_[d]->Reshape(top_shape);
                }

                int offset = batch->data_[d]->offset(item_id);
                Dtype *top_data = batch->data_[d]->mutable_cpu_data();
                DecodeDatumToBlob(datum, top_data + offset);
            }
            read_time += timer.MicroSeconds();
            if (!shuffle_) Next();
        }
        timer.Stop();
        batch_timer.Stop();
        DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
        DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
    }

    INSTANTIATE_CLASS(MultiDataLayer);

    REGISTER_LAYER_CLASS(MultiData);

}  // namespace caffe
