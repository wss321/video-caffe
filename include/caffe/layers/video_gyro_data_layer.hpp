//
// Created by wss on 2021/10/18.
//

#ifndef CAFFE_VIDEO_GYRO_DATA_LAYER_HPP
#define CAFFE_VIDEO_GYRO_DATA_LAYER_HPP

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_multi_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

// an extension the std::pair which used to store image filename and
// its label (int). now, a frame number associated with the video filename
// is needed (second param) to fully represent a video segment
struct quadruple {
    std::string first, second, third;
    int fourth;
};

namespace caffe {

/**
 * @brief Provides data to the Net from video files.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
    template <typename Dtype>
    class VideoGyroDataLayer : public BaseMultiPrefetchingDataLayer<Dtype> {
    public:
        explicit VideoGyroDataLayer(const LayerParameter& param)
                : BaseMultiPrefetchingDataLayer<Dtype>(param) {}
        virtual ~VideoGyroDataLayer();
        virtual void MultiDataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                    const vector<Blob<Dtype>*>& top);

        virtual inline const char* type() const { return "VideoGyroData"; }
        virtual inline int ExactNumBottomBlobs() const { return 0; }
        virtual inline int ExactNumTopBlobs() const { return 3; }

    protected:
        shared_ptr<Caffe::RNG> prefetch_rng_;
        virtual void ShuffleVideos();
        virtual void load_batch(Batch<Dtype>* batch);

        vector<quadruple> lines_;
        int lines_id_;
    };


}  // namespace caffe
#endif //CAFFE_VIDEO_GYRO_DATA_LAYER_HPP
