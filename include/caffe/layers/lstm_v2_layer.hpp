//
// Created by wss on 2021/9/25.
//

#ifndef CAFFE_LSTM_V2_LAYER_HPP
#define CAFFE_LSTM_V2_LAYER_HPP

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
    class LSTMV2Layer : public Layer<Dtype> {
        /**
         * input:
         *      data:TxBxN  (#time, #batch, #input dim)
         *      clip:TxB    (#time, #batch)
         * output:
         *      hidden state: TxBxH (#time, #batch, #hidden dim)
         * */
    public:
        explicit LSTMV2Layer(const LayerParameter& param)
                : Layer<Dtype>(param) {}
        virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                const vector<Blob<Dtype>*>& top);
        virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);

        virtual inline const char* type() const { return "LSTMV2"; }
        virtual bool IsRecurrent() const { return true; }

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

        Dtype clipping_threshold_; // threshold for clipped gradient
        Blob<Dtype> bias_multiplier_;

        Blob<Dtype> top_;       // output values
        Blob<Dtype> cell_;      // memory cell
        Blob<Dtype> pre_gate_;  // gate values before nonlinearity
        Blob<Dtype> gate_;      // gate values after nonlinearity

        Blob<Dtype> c_0_; // previous cell state value
        Blob<Dtype> h_0_; // previous hidden activation value
        Blob<Dtype> c_T_; // next cell state value
        Blob<Dtype> h_T_; // next hidden activation value

        // intermediate values
        Blob<Dtype> h_to_gate_; // save W*h_{t-1}
        Blob<Dtype> h_to_h_; // dL/dh_{t-1}
    };


}  // namespace caffe

#endif //CAFFE_LSTM_V2_LAYER_HPP
