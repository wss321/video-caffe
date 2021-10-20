//
// Created by wss on 2021/9/23.
//

#ifndef CAFFE_ATTENTION_LAYER_HPP
#define CAFFE_ATTENTION_LAYER_HPP

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/attention_layer.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

    template<typename Dtype>
    class AttentionLayer : public Layer<Dtype> {
    public:
        explicit AttentionLayer(const LayerParameter &param)
                :Layer<Dtype>(param) {}
        virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                const vector<Blob<Dtype>*>& top);
        virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);
        virtual void Reset();

        virtual inline const char *type() const { return "Attention"; }
        inline int MinBottomBlobs() const { return 1; }

        virtual inline int ExactNumTopBlobs() const {  return 1;}
        virtual inline int MaxBottomBlobs() const { return 3; }

    protected:
        virtual void FillUnrolledNet(NetParameter *net_param) const;
        virtual void AttentionOutputBlobNames(vector<string> *names) const;
        virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                 const vector<Blob<Dtype>*>& top);
        virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                 const vector<Blob<Dtype>*>& top);
        virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                                  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

    private:
        /// @brief A Net to implement the Attention functionality.
        shared_ptr<Net<Dtype> > unrolled_net_;

        /// @brief The number of independent streams to process simultaneously.
        int B_; // batch zize
        int N_; // feature num
        int D_; // feature dim
        bool has_bias_;
        int last_layer_index_;

        vector<Blob<Dtype>* > att_input_blobs_;
        vector<Blob<Dtype>* > att_output_blobs_;


    };

}  // namespace caffe

#endif //CAFFE_ATTENTION_LAYER_HPP
