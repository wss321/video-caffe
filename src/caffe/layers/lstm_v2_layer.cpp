#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/lstm_v2_layer.hpp"

namespace caffe {

    template <typename Dtype>
    inline Dtype sigmoid(Dtype x) {
        return 1. / (1. + exp(-x));
    }

    template <typename Dtype>
    void LSTMV2Layer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top) {
        clipping_threshold_ = this->layer_param_.lstm_param().clipping_threshold();
        T_ = bottom[0]->shape(0);
        N_ = bottom[0]->shape(1); // batch_size
        H_ = this->layer_param_.lstm_param().num_output(); // number of hidden units
        I_ = bottom[0]->shape(2); // input dimension

        // Check if we need to set up the weights
        if (this->blobs_.size() > 0) {
            LOG(INFO) << "Skipping parameter initialization";
        } else {
            this->blobs_.resize(3);
            shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
                    this->layer_param_.lstm_param().weight_filler()));

            // input-to-hidden weights
            // Intialize the weight
            vector<int> weight_shape;
            weight_shape.push_back(4*H_);
            weight_shape.push_back(I_);
            this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
            weight_filler->Fill(this->blobs_[0].get());

            // hidden-to-hidden weights
            // Intialize the weight
            weight_shape.clear();
            weight_shape.push_back(4*H_);
            weight_shape.push_back(H_);
            this->blobs_[1].reset(new Blob<Dtype>(weight_shape));
            weight_filler->Fill(this->blobs_[1].get());

            // If necessary, intiialize and fill the bias term
            vector<int> bias_shape(1, 4*H_);
            this->blobs_[2].reset(new Blob<Dtype>(bias_shape));
            shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
                    this->layer_param_.lstm_param().bias_filler()));
            bias_filler->Fill(this->blobs_[2].get());
        }  // parameter initialization
        this->param_propagate_down_.resize(this->blobs_.size(), true);

        vector<int> cell_shape;
        cell_shape.push_back(N_);
        cell_shape.push_back(H_);
        c_0_.Reshape(cell_shape);
        h_0_.Reshape(cell_shape);
        c_T_.Reshape(cell_shape);
        h_T_.Reshape(cell_shape);
        h_to_h_.Reshape(cell_shape);

        vector<int> gate_shape;
        gate_shape.push_back(N_);
        gate_shape.push_back(4);
        gate_shape.push_back(H_);
        h_to_gate_.Reshape(gate_shape);
    }

    template <typename Dtype>
    void LSTMV2Layer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                     const vector<Blob<Dtype>*>& top) {
//        // Figure out the dimensions
//        T_ = bottom[0]->num() / N_; // length of sequence
//        CHECK_EQ(bottom[0]->num() % N_, 0) << "Input size "
//                                              "should be multiple of batch size";
//        CHECK_EQ(bottom[0]->count() / T_ / N_, I_) << "Input size "
//                                                      "incompatible with inner product parameters.";

        vector<int> original_top_shape;
        original_top_shape.push_back(T_);
        original_top_shape.push_back(N_);
        original_top_shape.push_back(H_);
        top[0]->Reshape(original_top_shape);

        // Gate initialization
        vector<int> gate_shape;
        gate_shape.push_back(T_);
        gate_shape.push_back(N_);
        gate_shape.push_back(4);
        gate_shape.push_back(H_);
        pre_gate_.Reshape(gate_shape);
        gate_.Reshape(gate_shape);

        vector<int> top_shape;
        top_shape.push_back(T_);
        top_shape.push_back(N_);
        top_shape.push_back(H_);
        cell_.Reshape(top_shape);
        top_.Reshape(top_shape);
        top_.ShareData(*top[0]);
        top_.ShareDiff(*top[0]);

        // Set up the bias multiplier
        vector<int> multiplier_shape(1, N_*T_);
        bias_multiplier_.Reshape(multiplier_shape);
        caffe_set(bias_multiplier_.count(), Dtype(1),
                  bias_multiplier_.mutable_cpu_data());
    }

    template <typename Dtype>
    void LSTMV2Layer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                         const vector<Blob<Dtype>*>& top) {
        CHECK_EQ(top[0]->cpu_data(), top_.cpu_data());
        Dtype* top_data = top_.mutable_cpu_data();
        const Dtype* bottom_data = bottom[0]->cpu_data();
        const Dtype* clip = NULL;
        if (bottom.size() > 1) {
            clip = bottom[1]->cpu_data();
            CHECK_EQ(bottom[1]->shape(0) * bottom[1]->shape(1), bottom[1]->count());
        }
        const Dtype* weight_i = this->blobs_[0]->cpu_data();
        const Dtype* weight_h = this->blobs_[1]->cpu_data();
        const Dtype* bias = this->blobs_[2]->cpu_data();
        Dtype* pre_gate_data = pre_gate_.mutable_cpu_data();
        Dtype* gate_data = gate_.mutable_cpu_data();
        Dtype* cell_data = cell_.mutable_cpu_data();
        Dtype* h_to_gate = h_to_gate_.mutable_cpu_data();

        // Initialize previous state
        if (clip) {
            caffe_copy(c_0_.count(), c_T_.cpu_data(), c_0_.mutable_cpu_data());
            caffe_copy(h_0_.count(), h_T_.cpu_data(), h_0_.mutable_cpu_data());
        }
        else {
            caffe_set(c_0_.count(), Dtype(0.), c_0_.mutable_cpu_data());
            caffe_set(h_0_.count(), Dtype(0.), h_0_.mutable_cpu_data());
        }

        // Compute input to hidden forward propagation
        // pre_gate_data := bottom_data * weight_i^T
        // MxN := MxK * KxN

        caffe_cpu_gemm(CblasNoTrans, CblasTrans, T_*N_, 4*H_, I_, Dtype(1.),
                       bottom_data, weight_i, Dtype(0.), pre_gate_data);

        // pre_gate_data := 1*pre_gate_data + bias_multiplier_ * bias, where bias_multiplier_=(1)1xNxT
        // so: pre_gate_data := pre_gate_data + bias,
        // MxN := Mx1 * 1xN = T_*N_ x 4*H_
        caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, T_*N_, 4*H_, 1, Dtype(1.),
                       bias_multiplier_.cpu_data(), bias, Dtype(1.), pre_gate_data);

//        for (int i = 0; i < T_ * N_ * 4*H_; ++i) {
//            std::cout<<pre_gate_data[i]<<' ';
//            if ((i%(N_*4*H_))==0) std::cout<<std::endl;
//        }

        // Compute recurrent forward propagation
        for (int t = 0; t < T_; ++t) {
            Dtype* h_t = top_data + top_.offset(t); // h_t
            Dtype* c_t = cell_data + cell_.offset(t); // c_t
            Dtype* pre_gate_t = pre_gate_data + pre_gate_.offset(t);// Wx_t
            Dtype* gate_t = gate_data + gate_.offset(t);  // after activate
            Dtype* h_to_gate_t = h_to_gate; // Wh_{t-1}
            const Dtype* clip_t = clip ? clip + bottom[1]->offset(t) : NULL;
            const Dtype* h_t_1 = t > 0 ? (h_t - top_.offset(1)) : h_0_.cpu_data();
            const Dtype* c_t_1 = t > 0 ? (c_t - cell_.offset(1)) : c_0_.cpu_data();

            // Hidden-to-hidden propagation
            // h_to_gate := h_t_1 * weight_h^T
            // MxN := MxK * KxN = N_xH_ * H_x4H_ = N_ x 4*H_
            caffe_cpu_gemm(CblasNoTrans, CblasTrans, N_, 4*H_, H_, Dtype(1.),
                           h_t_1, weight_h, Dtype(0.), h_to_gate);

            for (int n = 0; n < N_; ++n) {
                const bool cont = clip_t ? clip_t[n] : t > 0;
                if (cont) {
                    caffe_add(4*H_, pre_gate_t, h_to_gate, pre_gate_t); // Wx_t + Wh_{t-1} + bi
                }
                for (int d = 0; d < H_; ++d) {
                    // Apply non-linearity
                    gate_t[d] = sigmoid(pre_gate_t[d]);
                    gate_t[H_ + d] = cont ? sigmoid(pre_gate_t[H_ + d]) : Dtype(0.);
                    gate_t[2*H_ + d] = sigmoid(pre_gate_t[2*H_ + d]);
                    gate_t[3*H_ + d] = tanh(pre_gate_t[3*H_ + d]);

                    // Compute cell : c(t) = f(t)*c(t-1) + i(t)*g(t)
                    c_t[d] = gate_t[H_ + d] * c_t_1[d] + gate_t[d] * gate_t[3*H_ + d];
                    h_t[d] = gate_t[2*H_ + d] * tanh(c_t[d]);
                }
//                std::cout<<"("<<t<<","<<n<<")  [";
//                for (int i = 0; i < H_; ++i) {
//                    std::cout<<h_t[i]<<' ';
//                }
//                std::cout<<"]"<<std::endl;


                h_t += H_;
                c_t += H_;
                c_t_1 += H_;
                pre_gate_t += 4*H_;
                gate_t += 4*H_;
                h_to_gate_t += 4*H_;
            }

        }
        // Preserve cell state and output value for truncated BPTT
        caffe_copy(N_*H_, cell_data + cell_.offset(T_-1), c_T_.mutable_cpu_data());
        caffe_copy(N_*H_, top_data + top_.offset(T_-1), h_T_.mutable_cpu_data());
    }

    template <typename Dtype>
    void LSTMV2Layer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                          const vector<bool>& propagate_down,
                                          const vector<Blob<Dtype>*>& bottom) {
        const Dtype* top_data = top_.cpu_data();
        const Dtype* bottom_data = bottom[0]->cpu_data();
        const Dtype* clip = NULL;
        if (bottom.size() > 1) {
            clip = bottom[1]->cpu_data();
            CHECK_EQ(bottom[1]->shape(0) * bottom[1]->shape(1), bottom[1]->count());
        }
        const Dtype* weight_i = this->blobs_[0]->cpu_data();
        const Dtype* weight_h = this->blobs_[1]->cpu_data();
        const Dtype* gate_data = gate_.cpu_data();
        const Dtype* cell_data = cell_.cpu_data();

        Dtype* top_diff = top_.mutable_cpu_diff();
        Dtype* pre_gate_diff = pre_gate_.mutable_cpu_diff();
        Dtype* gate_diff = gate_.mutable_cpu_diff();
        Dtype* cell_diff = cell_.mutable_cpu_diff();

        caffe_copy(N_*H_, c_T_.cpu_diff(), cell_diff + cell_.offset(T_-1));

        for (int t = T_-1; t >= 0; --t) {
            Dtype* dh_t = top_diff + top_.offset(t);// dL/dh_t
            Dtype* dc_t = cell_diff + cell_.offset(t);// dL/dc_t
            Dtype* gate_diff_t = gate_diff + gate_.offset(t);// dL/di_t, dL/df_t, dL/do_t, dL/dg_t
            Dtype* pre_gate_diff_t = pre_gate_diff + pre_gate_.offset(t); // dL/di_t_hat, dL/df_t_hat, dL/do_t_hat, dL/dg_t_hat
            Dtype* dh_t_1 = t > 0 ? top_diff + top_.offset(t-1) : h_0_.mutable_cpu_diff();
            Dtype* dc_t_1 = t > 0 ? cell_diff + cell_.offset(t-1) : c_0_.mutable_cpu_diff();
            const Dtype* clip_t = clip ? clip + bottom[1]->offset(t) : NULL;
            const Dtype* c_t = cell_data + cell_.offset(t);
            const Dtype* c_t_1 = t > 0 ? cell_data + cell_.offset(t-1) : c_0_.cpu_data();
            const Dtype* gate_t = gate_data + gate_.offset(t);

            for (int n = 0; n < N_; ++n) { // for batch
                const bool cont = clip_t ? clip_t[n] : t > 0;
                for (int d = 0; d < H_; ++d) {
                    const Dtype tanh_c = tanh(c_t[d]);

                    // dL/do_t = dL/dh_t * dh_t/do_t
                    // = dL/dh_t * m_t
                    // = dL/dh_t * tanh(c_t)
                    gate_diff_t[2*H_ + d] = dh_t[d] * tanh_c;

                    // dL/dc_t =dL/dh_t * dh_t/dm_t * dm_t/dc_t
                    // = dL/dh_t .* o_t .* (1 - (m_t)^2)
                    dc_t[d] += dh_t[d] * gate_t[2*H_ + d] * (Dtype(1.) - tanh_c * tanh_c);

                    // dL/dc_{t-1} = dL/dh_t * dh_t/dc_t * dc_t/dc_{t-1}
                    // dc_t/dc_{t-1} = f_t, if cont else 0.0
                    dc_t_1[d] = cont ? dc_t[d] * gate_t[H_ + d] : Dtype(0.);

                    // dL/df_t = dL/dh_t * dh_t/df_t
                    // dh_t/df_t = dh_t/dm_t * dm_t/dc_t * dc_t/df_t
                    // = dh_t/dc_t * dc_t/df_t
                    // dc_t/df_t = c_{t-1}
                    // dL/df_t = dL/dc_t * dc_t/df_t
                    gate_diff_t[H_ + d] = cont ? dc_t[d] * c_t_1[d] : Dtype(0.);

                    // dL/di_t = dL/dh_t * dh_t/di_t
                    // dh_t/di_t = dh_t/dm_t * dm_t/dc_t * dc_t/di_t
                    // = dh_t/dc_t * dc_t/di_t
                    // dc_t/di_t = g_t
                    gate_diff_t[d] = dc_t[d] * gate_t[3*H_ + d];

                    // dL/dg_t = dL/dh_t * dh_t/dg_t
                    // dh_t/dg_t = dh_t/dm_t * dm_t/dc_t * dc_t/dg_t
                    // = dh_t/dc_t * dc_t/dg_t
                    // dc_t/di_t = i_t
                    gate_diff_t[3*H_ +d] = dc_t[d] * gate_t[d];

                    // dL/di_t_hat = dL/di_t * di_t/di_t_hat
                    // di_t/di_t_hat = i_t_hat * （1 - i_t_hat)
                    pre_gate_diff_t[d] = gate_diff_t[d] * gate_t[d] * (Dtype(1.) - gate_t[d]);

                    // dL/df_t_hat = dL/df_t * di_t/df_t_hat
                    // df_t/df_t_hat = f_t_hat * （1 - f_t_hat)
                    pre_gate_diff_t[H_ + d] = gate_diff_t[H_ + d] * gate_t[H_ + d]
                                              * (1 - gate_t[H_ + d]);

                    // dL/do_t_hat = dL/do_t * do_t/do_t_hat
                    // do_t/do_t_hat = o_t_hat * （1 - o_t_hat)
                    pre_gate_diff_t[2*H_ + d] = gate_diff_t[2*H_ + d] * gate_t[2*H_ + d]
                                                * (1 - gate_t[2*H_ + d]);

                    // dL/dg_t_hat = dL/dg_t * dg_t/dg_t_hat
                    // dg_t/dg_t_hat = 1 - (g_t_hat)^2
                    pre_gate_diff_t[3*H_ + d] = gate_diff_t[3*H_ + d] * (Dtype(1.) -
                                                                         gate_t[3*H_ + d] * gate_t[3*H_ + d]);

                }
//                std::cout<<"("<<t<<","<<n<<")  [";
//                for (int i = 0; i < H_; ++i) {
//                    std::cout<<pre_gate_diff_t[2*H_ + i]<<' ';
//                }
//                std::cout<<"]"<<std::endl;

                // Clip derivatives before non-linearity
                if (clipping_threshold_ > Dtype(0.)) {
                    caffe_bound(4*H_, pre_gate_diff_t, -clipping_threshold_,
                                clipping_threshold_, pre_gate_diff_t);
                }

                dh_t += H_;
                c_t += H_;
                c_t_1 += H_;
                dc_t += H_;
                dc_t_1 += H_;
                gate_t += 4*H_;
                gate_diff_t += 4*H_;
                pre_gate_diff_t += 4*H_;
            }

            // Backprop output errors to the previous time step
            // h_to_h_ = dL/dh_{t-1}
            // dL/dh_{t-1}_i = dL/di_t_hat x di_t_hat/dh_{t-1}
            // di_t_hat/dh_{t-1} = W_hi
            // h_to_h_[0,:] = dL/di_t_hat x W_hi
            // h_to_h_[1,:] = dL/df_t_hat x W_hf
            // h_to_h_[2,:] = dL/do_t_hat x W_ho
            // h_to_h_[3,:] = dL/dg_t_hat x W_hg
            caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, N_, H_, 4*H_,
                           Dtype(1.), pre_gate_diff + pre_gate_.offset(t),
                           weight_h, Dtype(0.), h_to_h_.mutable_cpu_data());

            // dL/dh_{t-1} = Sum for each sample ( dL/dh_{t-1}_i + dL/dh_{t-1}_f + dL/dh_{t-1}_o + dL/dh_{t-1}_g )
            for (int n = 0; n < N_; ++n) {
                const bool cont = clip_t ? clip_t[n] : t > 0;
                const Dtype* h_to_h = h_to_h_.cpu_data() + h_to_h_.offset(n);
                if (cont) {
                    // dL/dh_{t-1} += dL/dh_{t-1}_i + dL/dh_{t-1}_f + dL/dh_{t-1}_o + dL/dh_{t-1}_g
                    caffe_add(H_, dh_t_1, h_to_h, dh_t_1);
                }
//                dh_t_1 += H_;
            }// for (int n = 0; n < N_; ++n)
        } //for (int t = T_-1; t >= 0; --t)

        if (this->param_propagate_down_[0]) {
            // Gradient w.r.t. input-to-hidden weight
            // dL/dW_xi = dL/di_t_hat * di_t_hat/dW_xi
            //          = x_t^T * dL/di_t_hat
            // dL/dWx += x^T * dL/pre_gate
            caffe_cpu_gemm(CblasTrans, CblasNoTrans, 4*H_, I_, T_*N_, Dtype(1.),
                           pre_gate_diff, bottom_data, Dtype(1.), this->blobs_[0]->mutable_cpu_diff());
//            for (int t = 0; t < 4*H_; ++t){
//                std::cout<<"("<<t<<")  [";
//                for (int n = 0; n < H_; ++n){
//
//                    std::cout <<this->blobs_[0]->mutable_cpu_diff()[t*H_ + n] << ' ';
//
//                }
//                std::cout<<"]"<<std::endl;
//
//            }
        }

        if (this->param_propagate_down_[1]) {
            // Gradient w.r.t. hidden-to-hidden weight
            // dL/dW_hi = dL/di_t_hat * di_t_hat/dW_hi
            //          = h_{t-1}^T * dL/di_t_hat
            // h1 --> hT
            caffe_cpu_gemm(CblasTrans, CblasNoTrans, 4*H_, H_, (T_-1)*N_, Dtype(1.),
                           pre_gate_diff + pre_gate_.offset(1), top_data,
                           Dtype(1.), this->blobs_[1]->mutable_cpu_diff());

            // Add Gradient from previous time-step
            // h0
            caffe_cpu_gemm(CblasTrans, CblasNoTrans, 4*H_, H_, N_, Dtype(1.),
                           pre_gate_diff, h_0_.cpu_data(),
                           Dtype(1.), this->blobs_[1]->mutable_cpu_diff());
        }
        if (this->param_propagate_down_[2]) {
            // Gradient w.r.t. bias
            // dL/db_i = dL/di_t_hat * di_t_hat/db_i
            //          = vec(1)^T * dL/di_t_hat
            // bias_multiplier_ = vec(1) of shape: 1x4*H_
            // pre_gate_diff of shape: TxN_x4xH_
            caffe_cpu_gemv(CblasTrans, T_*N_, 4*H_, Dtype(1.), pre_gate_diff,
                           bias_multiplier_.cpu_data(), Dtype(1.),
                           this->blobs_[2]->mutable_cpu_diff());
        }
        if (propagate_down[0]) {
            // Gradient w.r.t. bottom data
            // dL/dx_t_i = dL/di_t_hat * di_t_hat/dx_t
            // di_t_hat/dx_t = W_xi

            // dL/dx_t = dL/di_t_hat * di_t_hat/dx_t + dL/df_t_hat * df_t_hat/dx_t
            // + dL/dg_t_hat * dg_t_hat/dx_t + dL/do_t_hat * do_t_hat/dx_t
            caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, T_*N_, I_, 4*H_, Dtype(1.),
                           pre_gate_diff, weight_i, Dtype(0.), bottom[0]->mutable_cpu_diff());
        }
    }

#ifdef CPU_ONLY
    STUB_GPU(LSTMV2Layer);
#endif

    INSTANTIATE_CLASS(LSTMV2Layer);
    REGISTER_LAYER_CLASS(LSTMV2);

}  // namespace caffe