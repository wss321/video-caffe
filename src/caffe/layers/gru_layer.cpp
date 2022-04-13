#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/gru_layer.hpp"

namespace caffe {
    template<typename Dtype>
    inline Dtype sigmoid(Dtype x) {
        return 1. / (1. + exp(-x));
    }

    template<typename Dtype>
    inline Dtype tanh(Dtype x) {
        return 2. * sigmoid(2. * x) - 1.;
    }

    template<typename Dtype>
    void GRULayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                     const vector<Blob<Dtype> *> &top) {
//        clipping_threshold_ = this->layer_param_.gru_param().clipping_threshold();
        T_ = bottom[0]->shape(0);
        N_ = bottom[0]->shape(1); // batch_size
        H_ = this->layer_param_.gru_param().num_output(); // number of hidden units
        I_ = bottom[0]->shape(2); // input dimension

        // Check if we need to set up the weights
        if (this->blobs_.size() > 0) {
            LOG(INFO) << "Skipping parameter initialization";
        } else {
            this->blobs_.resize(6);
            shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
                    this->layer_param_.gru_param().weight_filler()));

            // input-to-hidden weights
            // Intialize the weight
            vector<int> weight_shape1{2 * H_, I_};//Wzx Wrx
            this->blobs_[0].reset(new Blob<Dtype>(weight_shape1));
            weight_filler->Fill(this->blobs_[0].get());

            // hidden-to-hidden weights
            // Intialize the weight

            vector<int> weight_shape2{2 * H_, H_};//Wzh Wrh
            this->blobs_[1].reset(new Blob<Dtype>(weight_shape2));
            weight_filler->Fill(this->blobs_[1].get());

            // r_t and h_{t-1} weights
            // Intialize the weight

            vector<int> weight_shape3{H_, H_};
            //W_om
            this->blobs_[2].reset(new Blob<Dtype>(weight_shape3));
            weight_filler->Fill(this->blobs_[2].get());

            // Wox
            this->blobs_[3].reset(new Blob<Dtype>(weight_shape3));
            weight_filler->Fill(this->blobs_[3].get());

            // If necessary, intiialize and fill the bias term
            this->blobs_[4].reset(new Blob<Dtype>(vector<int>{2 * H_})); // Bz, Br
            shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
                    this->layer_param_.gru_param().bias_filler()));
            bias_filler->Fill(this->blobs_[4].get());

            this->blobs_[5].reset(new Blob<Dtype>(vector<int>{1, H_})); // Bo
            bias_filler->Fill(this->blobs_[5].get());


        }  // parameter initialization
        this->param_propagate_down_.resize(this->blobs_.size(), true);

        vector<int> cell_shape{N_, H_};
        h_0_.Reshape(cell_shape);
        d_ph_.Reshape(cell_shape);

        vector<int> pregate_shape{N_, 2 * H_};
        h_to_gate_zr_.Reshape(pregate_shape);
        h_to_gate_m_.Reshape(cell_shape);
    }

    template<typename Dtype>
    void GRULayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
                                  const vector<Blob<Dtype> *> &top) {
        vector<int> top_shape{T_, N_, H_};
        top[0]->Reshape(top_shape);
        o_hat_.Reshape(top_shape);

        // Gate initialization
        vector<int> gate_shape{T_, N_, 3, H_};
        zr_hat_.Reshape(vector<int>{T_, N_, 2, H_});
        zro_.Reshape(gate_shape);

        top_.Reshape(top_shape);
        top_.ShareData(*top[0]);
        top_.ShareDiff(*top[0]);
        m_.Reshape(top_shape);

        // Set up the bias multiplier
        vector<int> multiplier_shape(1, N_ * T_);
        bias_multiplier_.Reshape(multiplier_shape);
        caffe_set(bias_multiplier_.count(), Dtype(1),
                  bias_multiplier_.mutable_cpu_data());
    }

    template<typename Dtype>
    void GRULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                      const vector<Blob<Dtype> *> &top) {
        CHECK_EQ(top[0]->cpu_data(), top_.cpu_data());
        Dtype *top_data = top_.mutable_cpu_data();
        const Dtype *bottom_data = bottom[0]->cpu_data();

        const Dtype *weight_zr_x = this->blobs_[0]->cpu_data();//3
        const Dtype *weight_h = this->blobs_[1]->cpu_data();//2
        const Dtype *weight_om = this->blobs_[2]->cpu_data();//1
        const Dtype *weight_ox = this->blobs_[3]->cpu_data();//1
        const Dtype *bias1 = this->blobs_[4]->cpu_data();
        const Dtype *bias2 = this->blobs_[5]->cpu_data();
        Dtype *zr_hat_data = zr_hat_.mutable_cpu_data();
        Dtype *gate_data = zro_.mutable_cpu_data();
        Dtype *m_data = m_.mutable_cpu_data();
        Dtype *h_to_gate_zr = h_to_gate_zr_.mutable_cpu_data(); // 2
        Dtype *h_to_gate_m = h_to_gate_m_.mutable_cpu_data(); // 1
        Dtype *o_hat_data = o_hat_.mutable_cpu_data();

        caffe_set(h_0_.count(), (Dtype) 0., h_0_.mutable_cpu_data());

        // Compute input to hidden forward propagation
        //Wzx * xt + bz
        //Wrx * xt + br
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, T_ * N_, 2 * H_, I_, (Dtype) 1.,
                              bottom_data, weight_zr_x, (Dtype) 0., zr_hat_data);
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, T_ * N_, 2 * H_, 1, (Dtype) 1.,
                              bias_multiplier_.cpu_data(), bias1, (Dtype) 1., zr_hat_data);
        //Wox * xt + bo
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, T_ * N_, H_, I_, (Dtype) 1.,
                              bottom_data, weight_ox, (Dtype) 0., o_hat_data);
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, T_ * N_, H_, 1, (Dtype) 1.,
                              bias_multiplier_.cpu_data(), bias2, (Dtype) 1., o_hat_data);
//        std::cout<<T_<<' '<<N_<<' '<<H_<<' '<<I_<<' ';
//        for (int i = 0; i < T_ * N_ * 2*H_; ++i) {
//            std::cout<<zr_hat_data[i]<<' ';
//            if ((i%(N_*2*H_))==0) std::cout<<std::endl;
//        }

        // Compute recurrent forward propagation
        for (int t = 0; t < T_; ++t) {
            Dtype *h_t = top_data + top_.offset(t);
            Dtype *zr_hat_t = zr_hat_data + zr_hat_.offset(t);
            Dtype *gate_t = gate_data + zro_.offset(t);
            Dtype *h_to_gate_t = h_to_gate_zr;
            Dtype *m_t = m_data + m_.offset(t);
            Dtype *o_hat_data_t = o_hat_data + o_hat_.offset(t);

            const Dtype *h_t_1 = t > 0 ? (top_data + top_.offset(t - 1)) : h_0_.cpu_data();

            // Hidden-to-hidden propagation
            // h_to_gate_zr := h_t_1 * weight_h^T
            // MxN := MxK * KxN = N_ x 2*H_
            caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, N_, 2 * H_, H_, (Dtype) 1.,
                                  h_t_1, weight_h, (Dtype) 0., h_to_gate_zr);

            for (int n = 0; n < N_; ++n) { //for batch
                caffe_add(2 * H_, zr_hat_t, h_to_gate_t,
                              zr_hat_t); //Wz*xt + b  and Wr*xh_{t-1}
                // zt = sigmoid()
                // rt = sigmoid()
                // mt = rt .* h_{t-1)
                for (int d = 0; d < H_; ++d) {//for dim
                    gate_t[d] = sigmoid(zr_hat_t[d]); //zt_i
                    gate_t[H_ + d] = sigmoid(zr_hat_t[H_ + d]);//rt_i
                    m_t[d] = gate_t[H_ + d] * h_t_1[d];// mt = rt .* h_{t-1)
                }
                h_t += H_;
                m_t += H_;
                h_t_1 += H_;
                o_hat_data_t += H_;
                zr_hat_t += 2 * H_;
                gate_t += 3 * H_;
                h_to_gate_t += 2 * H_;
            }


            h_t = top_data + top_.offset(t);
            m_t = m_data + m_.offset(t);
            zr_hat_t = zr_hat_data + zr_hat_.offset(t);
            gate_t = gate_data + zro_.offset(t);
            o_hat_data_t = o_hat_data + o_hat_.offset(t);

            h_t_1 = t > 0 ? (top_data + top_.offset(t - 1)) : h_0_.cpu_data();

            // h_to_gate_m = W_om*mt
            caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, N_, H_, H_, (Dtype) 1.,
                                  m_t, weight_om, (Dtype) 0., h_to_gate_m);
            for (int n = 0; n < N_; ++n) { //for batch
                caffe_add(H_, o_hat_data_t, h_to_gate_m, o_hat_data_t); //W_om*mt + (Wox * xt + b)

//                std::cout<<"("<<t<<","<<n<<")  [";
                for (int d = 0; d < H_; ++d) {//for dim
                    // Apply non-linearity
                    gate_t[2 * H_ + d] = tanh(o_hat_data_t[d]);// o_t

                    // Compute cell : h(t) = (1 - zt).* h_(t-1) + zt .* o_t
//                    h_t[d] = (Dtype(1.0) - gate_t[d]) * h_t_1[d] + gate_t[d] * gate_t[2 * H_ + d];

                    h_t[d] = (Dtype(1.0) - gate_t[d]) * gate_t[2 * H_ + d] + gate_t[d] * h_t_1[d];
//                    std::cout<<h_t[d]<<' ';
                }
//                std::cout<<"]"<<std::endl;

                h_t += H_;
                m_t += H_;
                h_t_1 += H_;
                o_hat_data_t += H_;
                zr_hat_t += 2 * H_;
                gate_t += 3 * H_;
            }
        }
//        // Preserve output value for truncated BPTT
//        caffe_copy(N_ * H_, top_data + top_.offset(T_ - 1), h_T_.mutable_cpu_data());
    }

    template<typename Dtype>
    void GRULayer<Dtype>::Backward_cpu(const vector<Blob<Dtype> *> &top,
                                       const vector<bool> &propagate_down,
                                       const vector<Blob<Dtype> *> &bottom) {
        const Dtype *top_data = top_.cpu_data();
        const Dtype *bottom_data = bottom[0]->cpu_data();
        const Dtype *weight_zr_x = this->blobs_[0]->cpu_data();
//        const Dtype *weight_zr_h = this->blobs_[1]->cpu_data();
        const Dtype *weight_om = this->blobs_[2]->cpu_data();
        const Dtype *weight_ox = this->blobs_[3]->cpu_data();
        const Dtype *zro_data = zro_.cpu_data();
        const Dtype *zr_hat_data = zr_hat_.cpu_data();

        Dtype *top_diff = top_.mutable_cpu_diff();
        Dtype *zr_hat_diff = zr_hat_.mutable_cpu_diff();
        Dtype *zro_diff = zro_.mutable_cpu_diff();
        Dtype *m_diff = m_.mutable_cpu_diff();
        Dtype *o_hat_diff = o_hat_.mutable_cpu_diff();


        for (int t = T_ - 1; t >= 0; --t) {
            Dtype *dh_t = top_diff + top_.offset(t);// dL/dh_t
            Dtype *zro_diff_t = zro_diff + zro_.offset(t);// dL/dz_t, dL/dr_t, dL/do_t
            Dtype *zr_hat_diff_t = zr_hat_diff + zr_hat_.offset(t); // dL/dz_t_hat, dL/dr_t_hat, dL/do_t_hat

            Dtype *m_diff_t = m_diff + m_.offset(t);
            Dtype *o_hat_diff_t = o_hat_diff + o_hat_.offset(t);
            Dtype *d_ph = d_ph_.mutable_cpu_data();

            const Dtype *h_t = top_data + top_.offset(t);
            const Dtype *zro_t = zro_data + zro_.offset(t);
            const Dtype *zr_hat_t = zr_hat_data + zr_hat_.offset(t);
            const Dtype *h_t_1 = t > 0 ? (h_t - top_.offset(1)) : h_0_.cpu_data();
//            const Dtype *o_hat_t = o_hat_data + o_hat_.offset(t);

            for (int n = 0; n < N_; ++n) { // for batch
//                const bool cont = clip_t ? clip_t[n] : t > 0;
                for (int d = 0; d < H_; ++d) {
                    const Dtype o_t = zro_t[2 * H_ + d];

                    // dL/dz_t = dL/dh_t * dh_t/dz_t
                    // = dL/dh_t * (o_t - h_{t-1})
//                    zro_diff_t[d] = cont ? dh_t[d] * (o_t - h_t_1[d]) : (Dtype)0.;
//                    zro_diff_t[d] = dh_t[d] * (o_t - h_t_1[d]);

                    zro_diff_t[d] = dh_t[d] * (h_t_1[d] - o_t);
                    // dL/do_t = dL/dh_t * dh_t/do_t
                    // = dL/dh_t * (z_t)
//                    zro_diff_t[2 * H_ + d] = dh_t[d] *zro_t[d];
                    zro_diff_t[2 * H_ + d] = dh_t[d] * (1 - zro_t[d]);

                    // dL/do_t_hat = dL/do_t * do_t/do_t_hat
                    // do_t/do_t_hat = 1 - (o_t)^2
                    o_hat_diff_t[d] = zro_diff_t[2 * H_ + d] * ((Dtype) 1. -
                                                                o_t * o_t);
                }
                // dL/dm_t = dL/dh_t * dh_t/do_t_hat * do_t_hat/dm_t
                // do_t_hat/dm_t = W_om
//                std::cout<<"("<<t<<","<<n<<")  [";
//                for (int d = 0; d < H_; ++d) {
//                    std::cout << zro_diff_t[2 * H_ + d] << ' ';
//                }
//                std::cout<<"]"<<std::endl;

                caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, 1, H_, H_, (Dtype) 1.,
                                      o_hat_diff_t, weight_om, (Dtype) 0., m_diff_t);
//                std::cout<<"("<<t<<","<<n<<")  [";
//                for (int d = 0; d < H_; ++d) {
//                    std::cout <<m_diff_t[ d] << ' ';
//                }
//                std::cout<<"]"<<std::endl;

                for (int d = 0; d < H_; ++d) {
                    // dL/dr_t =dL/dm_t * dm_t/dr_t
                    // dm_t/dr_t = h_{t-1}
//                    zro_diff_t[H_ + d] = cont ? m_diff_t[d] * h_t_1[d] : (Dtype)0.;
                    zro_diff_t[H_ + d] = m_diff_t[d] * h_t_1[d];

                    // dL/dz_t_hat = dL/dz_t * dz_t/dz_t_hat
                    // dz_t/dz_t_hat = z_t * （1 - z_t)
                    zr_hat_diff_t[d] = zro_diff_t[d] * zro_t[d] * ((Dtype) 1. - zro_t[d]);

                    // dL/dr_t_hat = dL/dr_t * dr_t/dr_t_hat
                    // dr_t/dr_t_hat = r_t * （1 - r_t)
                    zr_hat_diff_t[H_ + d] = zro_diff_t[H_ + d] * zro_t[H_ + d]
                                            * ((Dtype) 1. - zro_t[H_ + d]);
//                    d_ph[d] = m_diff_t[d] * zro_t[H_ + d] + dh_t[d]*((Dtype)1. - zro_t[d]);// v1
                    d_ph[d] = m_diff_t[d] * zro_t[H_ + d] + dh_t[d] * zro_t[d]; // v2


                }//for (int d = 0; d < H_; ++d)
//                std::cout<<"("<<t<<","<<n<<")  [";
//                for (int d = 0; d < 2*H_; ++d) {
//                    std::cout <<zr_hat_diff_t[d] << ' ';
//                }
//                std::cout<<"]"<<std::endl;


//                // Clip derivatives before non-linearity
//                if (clipping_threshold_ > Dtype(0.)) {
//                    caffe_bound(2 * H_, zr_hat_diff_t, -clipping_threshold_,
//                                clipping_threshold_, zr_hat_diff_t);
//                    caffe_bound(H_, o_hat_diff_t, -clipping_threshold_,
//                                clipping_threshold_, o_hat_diff_t);
//                }

                dh_t += H_;
                d_ph += H_;
                h_t_1 += H_;
                zro_t += 3 * H_;
                zr_hat_t += 2 * H_;
                zro_diff_t += 3 * H_;
                zr_hat_diff_t += 2 * H_;
                m_diff_t += H_;
            } // for (int n = 0; n < N_; ++n)
//            caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_, H_, 2 * H_,
//                                  (Dtype) 1., zr_hat_diff + zr_hat_.offset(t),
//                                  weight_zr_h, (Dtype) 1., d_ph_.mutable_cpu_data());
//
//            Dtype *dh_t_1 = t > 0 ? top_diff + top_.offset(t - 1) : h_0_.mutable_cpu_diff();
//            for (int n = 0; n < N_; ++n) {
//                const bool cont = clip_t ? clip_t[n] : t > 0;
//                const Dtype *dh_to_h1 = d_ph_.cpu_data() + d_ph_.offset(n);
//                if (cont) {
//                    // dL/dh_{t-1} += dL/dh_{t-1}_i + dL/dh_{t-1}_f + dL/dh_{t-1}_o + dL/dh_{t-1}_g
//                    caffe_add(H_, dh_t_1, dh_to_h1, dh_t_1);
//                }
//                dh_t_1 += H_;
//
//            }// for (int n = 0; n < N_; ++n)
        } //for (int t = T_-1; t >= 0; --t)

        if (this->param_propagate_down_[0]) {
            // Gradient w.r.t. input-to-hidden weight矩阵与向量的导数
            // dL/dW_zx = dL/dz_t_hat * dz_t_hat/dW_xi
            //          = x_t^T * dL/dz_t_hat
            // dL/dW_zx dL/dW_rx

            caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
                                  2 * H_, I_, T_ * N_,
                                  (Dtype) 1., zr_hat_diff, bottom_data,
                                  (Dtype) 1., this->blobs_[0]->mutable_cpu_diff());

        }

        if (this->param_propagate_down_[1]) {
            // Gradient w.r.t. hidden-to-hidden weight
            // dL/dW_zh = dL/dz_t_hat * dz_t_hat/dW_zh
            //          = h_{t-1}^T * dL/dz_t_hat
            // dL/dW_zh dL/dW_rh

//            for (int t = 0; t < 2*H_; ++t){
//                std::cout<<"("<<t<<")  [";
//                for (int n = 0; n < H_; ++n){
//
//                    std::cout <<this->blobs_[0]->mutable_cpu_diff()[t*H_ + n] << ' ';
//
//                }
//                std::cout<<"]"<<std::endl;
//
//            }aaa
            caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
                                  2 * H_, H_, (T_ - 1) * N_,
                                  (Dtype) 1.,zr_hat_diff + zr_hat_.offset(1), top_data,
                                  (Dtype) 1., this->blobs_[1]->mutable_cpu_diff());

            // Add Gradient from previous time-step
            // h0
            caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
                                  2 * H_, H_, N_,
                                  (Dtype) 1.,   zr_hat_diff, h_0_.cpu_data(),
                                  (Dtype) 1., this->blobs_[1]->mutable_cpu_diff());

        }
        if (this->param_propagate_down_[2]) {
            // dL/dW_om = dL/do_t_hat * do_t_hat/dW_om
            caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, H_, H_, T_ * N_, (Dtype) 1.,
                                  o_hat_diff, m_.cpu_data(),
                                  (Dtype) 1., this->blobs_[2]->mutable_cpu_diff());
        }
        if (this->param_propagate_down_[3]) {
            //dL/dW_ox
            caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, H_, I_, T_ * N_, (Dtype) 1.,
                                  o_hat_diff, bottom_data, (Dtype) 1.,
                                  this->blobs_[3]->mutable_cpu_diff());
        }
        if (this->param_propagate_down_[4]) {
            // Gradient w.r.t. bias1
            // dL/db_i = dL/di_t_hat * di_t_hat/db_i
            //          = vec(1)^T * dL/di_t_hat
            // bias_multiplier_ = vec(1) of shape: N*T
            // pre_gate_diff of shape: TxN_x2xH_
            // this->blobs_[3] 1x3xH
            caffe_cpu_gemv<Dtype>(CblasTrans,
                                  T_ * N_, 2 * H_,
                                  (Dtype) 1., zr_hat_diff, bias_multiplier_.cpu_data(),
                                  (Dtype) 1., this->blobs_[4]->mutable_cpu_diff());
        }

        if (this->param_propagate_down_[5]) {
            // Gradient w.r.t. bias2
            caffe_cpu_gemv<Dtype>(CblasTrans, T_ * N_, H_, (Dtype) 1., o_hat_diff,
                                  bias_multiplier_.cpu_data(), (Dtype) 1.,
                                  this->blobs_[5]->mutable_cpu_diff());
        }
        if (propagate_down[0]) {
//            // dL/dx_t = dL/dr_hat_t * dr_hat_t/dx_t + dL/do_hat_t * do_hat_t/dx_t + dL/dz_hat_t * dz_hat_t/dx_t
//
//            // dL/dz_hat_t * dz_hat_t/dx_t
//            caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, 1, I_, H_, (Dtype)1.,
//                           pre_gate_diff_t, weight_zr_x, (Dtype)1.,
//                           bottom[0]->mutable_cpu_diff() + (t * N_ + n) * I_);
//            // dL/dr_hat_t * dr_hat_t/dx_t
//            caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, 1, I_, H_, (Dtype)1.,
//                           pre_gate_diff_t + H_, weight_zr_x + H_ * I_, (Dtype)1.,
//                           bottom[0]->mutable_cpu_diff() + (t * N_ + n) * I_);
//            // dL/do_hat_t * do_hat_t/dx_t
//            caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, 1, I_, H_, (Dtype)1.,
//                           o_hat_diff_t, weight_ox, (Dtype)1.,
//                           bottom[0]->mutable_cpu_diff() + (t * N_ + n) * I_);
            // Gradient w.r.t. bottom data
            // dL/dx_t_i = dL/di_t_hat * di_t_hat/dx_t
            // di_t_hat/dx_t = W_xi

            // dL/dx_t = dL/di_t_hat * di_t_hat/dx_t + dL/df_t_hat * df_t_hat/dx_t
            // + dL/dg_t_hat * dg_t_hat/dx_t + dL/do_t_hat * do_t_hat/dx_t
            caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, T_ * N_, I_, 2 * H_, (Dtype) 1.,
                                  zr_hat_diff, weight_zr_x, (Dtype) 0., bottom[0]->mutable_cpu_diff());

            caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, T_ * N_, I_, H_, (Dtype) 1.,
                                  o_hat_diff, weight_ox, (Dtype) 1., bottom[0]->mutable_cpu_diff());

        }
    }

#ifdef CPU_ONLY
    STUB_GPU(GRULayer);
#endif

    INSTANTIATE_CLASS(GRULayer);

    REGISTER_LAYER_CLASS(GRU);

}  // namespace caffe