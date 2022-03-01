//
// Created by wss on 2021/9/22.
//

#ifndef CAFFE_GRU_LAYER_HPP
#define CAFFE_GRU_LAYER_HPP

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

    template <typename Dtype>
    class GRULayer : public Layer<Dtype> {
    public:
        explicit GRULayer(const LayerParameter& param)
                : Layer<Dtype>(param) {}
        virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                const vector<Blob<Dtype>*>& top);
        virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);

        virtual inline const char* type() const { return "GRU"; }
        virtual inline int MinBottomBlobs() const { return 1; }
        virtual inline int MaxBottomBlobs() const { return 1; }
        virtual inline int ExactNumTopBlobs() const { return 1; }

    protected:
        virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                 const vector<Blob<Dtype>*>& top);
        virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                 const vector<Blob<Dtype>*>& top);
        virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                                  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
        virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                                  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

        int I_; // input dimension
        int H_; // num of hidden units
        int T_; // length of sequence
        int N_; // batch size

//        Dtype clipping_threshold_; // threshold for clipped gradient
        Blob<Dtype> bias_multiplier_;

        Blob<Dtype> top_;       // output values
        Blob<Dtype> zr_hat_;  // gate values before non-linearity:z_t_hat, r_t_hat
        Blob<Dtype> zro_;      // gate values after non-linearity:z_t, r_t

        Blob<Dtype> m_; // next r_t .* h_{t-1} value: m_t

        Blob<Dtype> h_0_; // previous hidden activation value
        Blob<Dtype> o_hat_; // next o_t value

        // intermediate values
        Blob<Dtype> h_to_gate_zr_; // W_zh * h_{t-1}  and W_rh * h_{t-1}
        Blob<Dtype> h_to_gate_m_; // W_om * m_t
        Blob<Dtype> d_ph_;  // dL/dh_{t-1}
    };

}  // namespace caffe
#endif //CAFFE_GRU_LAYER_HPP
