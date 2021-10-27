#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/gru_cell_layer.hpp"


namespace caffe {

    template<typename Dtype>
    inline Dtype sigmoid(Dtype x) {
        return 1. / (1. + exp(-x));
    }

    template<typename Dtype>
    inline Dtype sigmoid_diff(Dtype x) {
        return x * (1. - x);
    }

    template<typename Dtype>
    inline Dtype tanh(Dtype x) {
        return 2. * sigmoid(2. * x) - 1.;
    }

    template<typename Dtype>
    inline Dtype tanh_diff(Dtype x) {
        return (1. - x * x);
    }

    template<typename Dtype>
    void GRUCellLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                         const vector<Blob<Dtype> *> &top) {
        GRUCellParameter gruCellParam = this->layer_param_.gru_cell_param();
        CHECK((gruCellParam.has_num_cells()))
        << "gru_cell_param.has_num_cells()";
        CHECK_EQ(bottom[0]->num_axes(), bottom[1]->num_axes()) << "num_axes of bottom must be equal";
        CHECK_GT(bottom[1]->num_axes(), 2) << "num_axes of bottom data must be great than 2";

        feature_dim_ = gruCellParam.num_cells();
        has_bias = gruCellParam.bias_term();
        num_axes_ = bottom[0]->num_axes();
        batch_size_ = bottom[0]->shape(0);

        if (num_axes_ > 2){
            CHECK_EQ(num_axes_, 3) << "num_axes of bottom must be 2 or 3";
            CHECK_EQ(bottom[0]->shape(1), 1) << "bottom axis 1 must be 1";
            CHECK_EQ(bottom[1]->shape(1), 1) << "bottom axis 1 must be 1";
        }

        input_data_dim_ = num_axes_ == 3 ? bottom[0]->shape(2) : bottom[0]->shape(1);

        CHECK_EQ(feature_dim_, bottom[1]->shape(num_axes_-1)) <<
                                                    "Number of input memory channels must match the number of lstm mem_cells";
        M_ = batch_size_;
        N_ = feature_dim_;
        K_ = input_data_dim_;


        this->blobs_.resize(9);
        shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
                gruCellParam.weight_filler()));

        shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
                gruCellParam.bias_filler()));


        // Wzx Wrx Wox
        for (int i = 0; i < 3; ++i) {
            this->blobs_[i].reset(new Blob<Dtype>(vector<int>{feature_dim_, input_data_dim_}));
            weight_filler->Fill(this->blobs_[i].get());
        }
        // Wzh Wrh Wom
        for (int i = 3; i < 6; ++i) {
            this->blobs_[i].reset(new Blob<Dtype>(vector<int>{feature_dim_, feature_dim_}));
            weight_filler->Fill(this->blobs_[i].get());
        }
        // bias term: bz br bo
        for (int i = 6; i < 9; ++i) {
            this->blobs_[i].reset(new Blob<Dtype>(vector<int>{1, feature_dim_}));
            bias_filler->Fill(this->blobs_[i].get());
        }

        z_gates_data_buffer_.reset(new Blob<Dtype>());
        r_gates_data_buffer_.reset(new Blob<Dtype>());
        o_gates_data_buffer_.reset(new Blob<Dtype>());
        m_data_buffer_.reset(new Blob<Dtype>());

        vector<int> multiplier_shape(1, M_);
        bias_multiplier_.Reshape(multiplier_shape);
        caffe_set(bias_multiplier_.count(), Dtype(1),
                  bias_multiplier_.mutable_cpu_data());

        // Propagate gradients to the parameters (as directed by backward pass).
        this->param_propagate_down_.resize(this->blobs_.size(), true);
        if (!has_bias)
            for (int i = 0; i < 3; ++i)
                this->param_propagate_down_[6+i] = false;
    }

    template<typename Dtype>
    void GRUCellLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
                                      const vector<Blob<Dtype> *> &top) {
        CHECK((this->layer_param_.bottom_size() == 2
               || this->layer_param_.bottom_size() == 0))
        << "gru cell must have a data and cell bottom";
        CHECK((this->layer_param_.top_size() == 1
               || this->layer_param_.top_size() == 0))
        << "gru cell must have a data top";

        z_gates_data_buffer_->Reshape(vector<int>{batch_size_, feature_dim_});
        r_gates_data_buffer_->Reshape(vector<int>{batch_size_, feature_dim_});
        o_gates_data_buffer_->Reshape(vector<int>{batch_size_, feature_dim_});
        m_data_buffer_->Reshape(vector<int>{batch_size_, feature_dim_});

        top[0]->Reshape(bottom[1]->shape());
    }

    template<typename Dtype>
    void GRUCellLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                          const vector<Blob<Dtype> *> &top) {
        const Dtype *x = bottom[0]->cpu_data();
        const Dtype *ht_1 = bottom[1]->cpu_data();

        const Dtype *Wzx = this->blobs_[0]->cpu_data();
        const Dtype *Wrx = this->blobs_[1]->cpu_data();
        const Dtype *Wox = this->blobs_[2]->cpu_data();

        const Dtype *Wzh = this->blobs_[3]->cpu_data();
        const Dtype *Wrh = this->blobs_[4]->cpu_data();
        const Dtype *Wom = this->blobs_[5]->cpu_data();

        const Dtype *Bz = this->blobs_[6]->cpu_data();
        const Dtype *Br = this->blobs_[7]->cpu_data();
        const Dtype *Bo = this->blobs_[8]->cpu_data();

        Dtype *ht = top[0]->mutable_cpu_data();

        Dtype *z = z_gates_data_buffer_->mutable_cpu_data();
        Dtype *r = r_gates_data_buffer_->mutable_cpu_data();
        Dtype *o = o_gates_data_buffer_->mutable_cpu_data();
        Dtype *m = m_data_buffer_->mutable_cpu_data();
        const int num_elem = M_*N_;

        // pre_gate
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_,
                              (Dtype) 1., x, Wzx,
                              (Dtype) 0., z);
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, N_,
                              (Dtype) 1., ht_1, Wzh,
                              (Dtype) 1., z);

        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_,
                              (Dtype) 1., x, Wrx,
                              (Dtype) 0., r);
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, N_,
                              (Dtype) 1., ht_1, Wrh,
                              (Dtype) 1., r);

        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_,
                              (Dtype) 1., x, Wox,
                              (Dtype) 0., o);

        if (has_bias) {
            caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype) 1.,
                                  bias_multiplier_.cpu_data(), Bz, (Dtype) 1., z);
            caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype) 1.,
                                  bias_multiplier_.cpu_data(), Br, (Dtype) 1., r);
            caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype) 1.,
                                  bias_multiplier_.cpu_data(), Bo, (Dtype) 1., o);
        }

        for (int idx = 0; idx < num_elem; ++idx) {
            z[idx] = sigmoid(z[idx]);
            r[idx] = sigmoid(r[idx]);
        }
        caffe_mul(num_elem, r, ht_1, m);//m=r*ht-1

        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, N_,
                              (Dtype) 1., m, Wom,
                              (Dtype) 1., o);
        for (int idx = 0; idx < num_elem; ++idx) {
            o[idx] = tanh(o[idx]);
            ht[idx] = ((Dtype)1. - z[idx]) * o[idx] + z[idx] * ht_1[idx];
        }
    }

    template<typename Dtype>
    void GRUCellLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype> *> &top,
                                           const vector<bool> &propagate_down,
                                           const vector<Blob<Dtype> *> &bottom) {
        for (int i = 0; i < 2; ++i) {
            caffe_set(bottom[i]->count(), Dtype(0), bottom[i]->mutable_cpu_diff());
        }
        const Dtype *x = bottom[0]->cpu_data();
        const Dtype *ht_1 = bottom[1]->cpu_data();

        const Dtype *Wzx = this->blobs_[0]->cpu_data();
        const Dtype *Wrx = this->blobs_[1]->cpu_data();
        const Dtype *Wox = this->blobs_[2]->cpu_data();

        const Dtype *Wzh = this->blobs_[3]->cpu_data();
        const Dtype *Wrh = this->blobs_[4]->cpu_data();
        const Dtype *Wom = this->blobs_[5]->cpu_data();

        const Dtype *z = z_gates_data_buffer_->cpu_data();
        const Dtype *r = r_gates_data_buffer_->cpu_data();
        const Dtype *o = o_gates_data_buffer_->cpu_data();
        const Dtype *m = m_data_buffer_->cpu_data();

        Dtype *dz_hat = z_gates_data_buffer_->mutable_cpu_diff();
        Dtype *dr_hat = r_gates_data_buffer_->mutable_cpu_diff();
        Dtype *do_hat = o_gates_data_buffer_->mutable_cpu_diff();
        Dtype *dm = m_data_buffer_->mutable_cpu_diff();

        Dtype *dx = bottom[0]->mutable_cpu_diff();
        Dtype *dh_1 = bottom[1]->mutable_cpu_diff();
        const Dtype *top_diff = top[0]->cpu_diff();
        const int num_elem = M_*N_;

        // pre-gate diff:dL/dz_har ...
        for (int idx = 0; idx < num_elem; ++idx) {
            dz_hat[idx] = top_diff[idx] * (ht_1[idx] - o[idx]) * sigmoid_diff(z[idx]);
            do_hat[idx] = top_diff[idx] * (Dtype(1.0) - z[idx]) * tanh_diff(o[idx]);
        }
        //dm
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, N_, (Dtype) 1.,
                              do_hat, Wom, (Dtype) 0., dm);

        for (int idx = 0; idx < num_elem; ++idx) {
            dr_hat[idx] = dm[idx] * ht_1[idx] * sigmoid_diff(r[idx]);
            dh_1[idx] += dm[idx] * r[idx] + top_diff[idx] * z[idx];
        }

        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, N_, (Dtype) 1.,
                              dz_hat, Wzh, (Dtype) 1., dh_1);//dht-1
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, N_, (Dtype) 1.,
                              dr_hat, Wrh, (Dtype) 1., dh_1);//dht-1

        if (propagate_down[0]){
            // dx
            caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, K_, N_, (Dtype) 1.,
                                  dz_hat, Wzx, (Dtype) 1., dx);
            caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, K_, N_, (Dtype) 1.,
                                  dr_hat, Wrx, (Dtype) 1., dx);
            caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, K_, N_, (Dtype) 1.,
                                  do_hat, Wox, (Dtype) 1., dx);
        }

        Dtype *Wzx_diff = this->blobs_[0]->mutable_cpu_diff();
        Dtype *Wrx_diff = this->blobs_[1]->mutable_cpu_diff();
        Dtype *Wox_diff = this->blobs_[2]->mutable_cpu_diff();
        Dtype *Wzh_diff = this->blobs_[3]->mutable_cpu_diff();
        Dtype *Wrh_diff = this->blobs_[4]->mutable_cpu_diff();
        Dtype *Wom_diff = this->blobs_[5]->mutable_cpu_diff();

        Dtype *Bz_diff = this->blobs_[6]->mutable_cpu_diff();
        Dtype *Br_diff = this->blobs_[7]->mutable_cpu_diff();
        Dtype *Bo_diff = this->blobs_[8]->mutable_cpu_diff();

        if (this->param_propagate_down_[0])
            caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
                                  feature_dim_, input_data_dim_, batch_size_,
                                  (Dtype) 1., dz_hat, x,
                                  (Dtype) 1., Wzx_diff);
        if (this->param_propagate_down_[1])
            caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
                                  feature_dim_, input_data_dim_, batch_size_,
                                  (Dtype) 1., dr_hat, x,
                                  (Dtype) 1., Wrx_diff);
        if (this->param_propagate_down_[2])
            caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
                                  feature_dim_, input_data_dim_, batch_size_,
                                  (Dtype) 1., do_hat, x,
                                  (Dtype) 1., Wox_diff);
        if (this->param_propagate_down_[3])
            caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
                                  feature_dim_, feature_dim_, batch_size_,
                                  (Dtype) 1., dz_hat, ht_1,
                                  (Dtype) 1., Wzh_diff);
        if (this->param_propagate_down_[4])
            caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
                                  feature_dim_, feature_dim_, batch_size_,
                                  (Dtype) 1., dr_hat, ht_1,
                                  (Dtype) 1., Wrh_diff);
        if (this->param_propagate_down_[5])
            caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
                                  feature_dim_, feature_dim_, batch_size_,
                                  (Dtype) 1., do_hat, m,
                                  (Dtype) 1., Wom_diff);
        if (this->param_propagate_down_[6])
            caffe_cpu_gemv<Dtype>(CblasTrans,
                                  M_, N_,
                                  (Dtype) 1., dz_hat, bias_multiplier_.cpu_data(),
                                  (Dtype) 1., Bz_diff);
        if (this->param_propagate_down_[7])
            caffe_cpu_gemv<Dtype>(CblasTrans,
                                  M_, N_,
                                  (Dtype) 1., dr_hat, bias_multiplier_.cpu_data(),
                                  (Dtype) 1., Br_diff);
        if (this->param_propagate_down_[8])
            caffe_cpu_gemv<Dtype>(CblasTrans,
                                  M_, N_,
                                  (Dtype) 1., do_hat, bias_multiplier_.cpu_data(),
                                  (Dtype) 1., Bo_diff);

    }

#ifdef CPU_ONLY
    STUB_GPU(GRUCellLayer);
#endif

    INSTANTIATE_CLASS(GRUCellLayer);

    REGISTER_LAYER_CLASS(GRUCell);

}  // namespace caffe