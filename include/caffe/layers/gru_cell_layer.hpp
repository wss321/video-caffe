//
// Created by wss on 2021/10/26.
//

#ifndef CAFFE_GRU_CELL_LAYER_HPP
#define CAFFE_GRU_CELL_LAYER_HPP

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

    template <typename Dtype>
    class GRUCellLayer : public Layer<Dtype> {
        /* input: xt    :batch_size x dim_x
         *        ht-1  :batch_size x dim_h
         *
         * output: ht  :batch_size x dim_h
         * */
    public:
        explicit GRUCellLayer(const LayerParameter& param)
                : Layer<Dtype>(param) {}
        virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                const vector<Blob<Dtype>*>& top);
        virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);

        virtual inline const char* type() const { return "GRUCell"; }
        virtual inline int MinBottomBlobs() const { return 2; }
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

        int feature_dim_;  // num memory cells;
        int batch_size_;  // batch size;
        int input_data_dim_;
        int M_;
        int N_;
        int K_;
        int num_axes_;
        bool has_bias;
        shared_ptr<Blob<Dtype> > z_gates_data_buffer_;
        shared_ptr<Blob<Dtype> > r_gates_data_buffer_;
        shared_ptr<Blob<Dtype> > o_gates_data_buffer_;
        shared_ptr<Blob<Dtype> > m_data_buffer_;
        Blob<Dtype> bias_multiplier_;
    };

}  // namespace caffe

#endif //CAFFE_GRU_CELL_LAYER_HPP
