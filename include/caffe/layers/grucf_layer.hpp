//
// Created by wss on 2021/9/22.
//

#ifndef CAFFE_GRUCF_LAYER_HPP
#define CAFFE_GRUCF_LAYER_HPP

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/recurrent_layer.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

    template<typename Dtype>
    class RecurrentLayer;

/**
 *     r := \sigmoid[ W_{ir} * x + b_{ir} + W_{hr} + b_{hr} ]
 *     z := \sigmoid[ W_{iz} * x + b_{iz} + W_{hz} + b_{hz}]
 *     n := \tanh[ W_{in} * x + b_{in} + r .* (W_{hr} + b_{hr}) ]
 *     h' := (1-z) .* n + z .* h

 */
    template<typename Dtype>
    class GRUCFLayer : public RecurrentLayer<Dtype> {
    public:
        explicit GRUCFLayer(const LayerParameter &param)
                : RecurrentLayer<Dtype>(param) {}

        virtual inline const char *type() const { return "GRU"; }
        inline int MinBottomBlobs() const { return 1; }

    protected:
        virtual void FillUnrolledNet(NetParameter *net_param) const;

        virtual void RecurrentInputBlobNames(vector<string> *names) const;

        virtual void RecurrentOutputBlobNames(vector<string> *names) const;

        virtual void RecurrentInputShapes(vector<BlobShape> *shapes) const;

        virtual void OutputBlobNames(vector<string> *names) const;
    };

}  // namespace caffe
#endif //CAFFE_GRUCF_LAYER_HPP
