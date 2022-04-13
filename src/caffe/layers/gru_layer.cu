#include <vector>
#include <algorithm>
#include <cmath>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/gru_layer.hpp"

namespace caffe {

    template<typename Dtype>
    __device__ Dtype sigmoid(Dtype x) {
        return Dtype(1) / (Dtype(1) + exp(-x));
    }

    template<typename Dtype>
    __device__ Dtype tanh(Dtype x) {
        return Dtype(2) * sigmoid(Dtype(2) * x) - Dtype(1);
    }

    template<typename Dtype>
    __global__ void ClipAdd(const int nthreads, int t, const Dtype *add_vec, Dtype *data) {
        CUDA_KERNEL_LOOP(index, nthreads) {
            data[index] += add_vec[index];
        }
    }


    template<typename Dtype>
    __global__ void GRUForwardZRM(const int nthreads, const int H,
                                  const Dtype *pre_gate, const Dtype *h_t_1, Dtype *gate, Dtype *m) {
        CUDA_KERNEL_LOOP(index, nthreads) {
            const int n = index / H;
            const int d = index % H;
            Dtype *offset_gate = gate + 3 * H * n;
            const Dtype *offset_zr = pre_gate + 2 * H * n;

            offset_gate[d] = sigmoid(offset_zr[d]);
            offset_gate[H + d] = sigmoid(offset_zr[H + d]);
            m[index] = offset_gate[H + d] * h_t_1[index];// mt = rt .* h_{t-1)
        }
    }

    template<typename Dtype>
    __global__ void GRUBackwardZO_Ohat(const int nthreads, const int H,
                                       const Dtype *zro, const Dtype *dh, const Dtype *h_t_1, Dtype *zro_diff,
                                       Dtype *o_hat_diff) {
        CUDA_KERNEL_LOOP(index, nthreads) {
            const int n = index / H; // index [0, N*H)
            const int d = index % H;
            Dtype *offset_zro_diff = zro_diff + 3 * H * n;
            const Dtype *offset_zro = zro + 3 * H * n;
            const Dtype o_t = offset_zro[2 * H + d];

            offset_zro_diff[d] = dh[index] * (h_t_1[index] - o_t);
            offset_zro_diff[2 * H + d] = dh[index] * (1 - offset_zro[d]);
            o_hat_diff[index] = offset_zro_diff[2 * H + d] * ((Dtype) 1. -
                                                              o_t * o_t);
        }
    }

    template<typename Dtype>
    __global__ void GRUBackwardR_ZRhat_h1(const int nthreads, const int H,
                                          const Dtype *zro, const Dtype *dh, const Dtype *m_diff, const Dtype *h_t_1, const Dtype *weight_zr_h,
                                          Dtype *zro_diff, Dtype *zr_hat_diff, Dtype *d_ph) {
        CUDA_KERNEL_LOOP(index, nthreads) {
            const int n = index / H; // index [0, N*H)
            const int d = index % H;
//            const int N = nthreads/H;
            Dtype *offset_zro_diff = zro_diff + 3 * H * n;
            Dtype *offset_zr_hat_diff = zr_hat_diff + 2 * H * n;
            const Dtype *offset_zro = zro + 3 * H * n;
//            const Dtype o_t = offset_zro[2 * H + d];

            offset_zro_diff[H + d] = m_diff[index] * h_t_1[index];

            // dL/dz_t_hat = dL/dz_t * dz_t/dz_t_hat
            // dz_t/dz_t_hat = z_t * （1 - z_t)
            offset_zr_hat_diff[d] = offset_zro_diff[d] * offset_zro[d] * ((Dtype) 1. - offset_zro[d]);

            // dL/dr_t_hat = dL/dr_t * dr_t/dr_t_hat
            // dr_t/dr_t_hat = r_t * （1 - r_t)
            offset_zr_hat_diff[H + d] = offset_zro_diff[H + d] * offset_zro[H + d]
                                        * ((Dtype) 1. - offset_zro[H + d]);
//                    d_ph[d] = m_diff_t[d] * zro_t[H_ + d] + dh_t[d]*((Dtype)1. - zro_t[d]);
            d_ph[index] = m_diff[index] * offset_zro[H + d] + dh[index] * offset_zro[d];

        }
    }

    template<typename Dtype>
    __global__ void GRUForwardOH(const int nthreads, const int H,
                                 const Dtype *o_hat_t, const Dtype *h_t_1, Dtype *gate, Dtype *h) {
        CUDA_KERNEL_LOOP(index, nthreads) {
            const int n = index / H;
            const int d = index % H;
            Dtype *offset_gate = gate + 3 * H * n;
            offset_gate[2 * H + d] = tanh(o_hat_t[index]);
            h[index] = (Dtype(1.0) - offset_gate[d]) * offset_gate[2 * H + d] + offset_gate[d] * h_t_1[index];
        }
    }

    template<typename Dtype>
    void GRULayer<Dtype>::Forward_gpu(const vector<Blob<Dtype> *> &bottom,
                                      const vector<Blob<Dtype> *> &top) {

        CHECK_EQ(top[0]->gpu_data(), top_.gpu_data());
        Dtype *top_data = top_.mutable_cpu_data();
        const Dtype *bottom_data = bottom[0]->gpu_data();

        const Dtype *weight_zr_x = this->blobs_[0]->gpu_data();//3
        const Dtype *weight_h = this->blobs_[1]->gpu_data();//2
        const Dtype *weight_om = this->blobs_[2]->gpu_data();//1
        const Dtype *weight_ox = this->blobs_[3]->gpu_data();//1
        const Dtype *bias1 = this->blobs_[4]->gpu_data();
        const Dtype *bias2 = this->blobs_[5]->gpu_data();
        Dtype *zr_hat_data = zr_hat_.mutable_gpu_data();
        Dtype *gate_data = zro_.mutable_gpu_data();
        Dtype *m_data = m_.mutable_gpu_data();
        Dtype *h_to_gate_zr = h_to_gate_zr_.mutable_gpu_data(); // 2
        Dtype *h_to_gate_m = h_to_gate_m_.mutable_gpu_data(); // 1
        Dtype *o_hat_data = o_hat_.mutable_gpu_data();

        caffe_gpu_set(h_0_.count(), (Dtype) 0., h_0_.mutable_gpu_data());

        //Wzx * xt + b
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, T_ * N_, 2 * H_, I_, (Dtype) 1.,
                              bottom_data, weight_zr_x, (Dtype) 0., zr_hat_data);
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, T_ * N_, 2 * H_, 1, (Dtype) 1.,
                              bias_multiplier_.gpu_data(), bias1, (Dtype) 1., zr_hat_data);

        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, T_ * N_, H_, I_, (Dtype) 1.,
                              bottom_data, weight_ox, (Dtype) 0., o_hat_data);
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, T_ * N_, H_, 1, (Dtype) 1.,
                              bias_multiplier_.gpu_data(), bias2, (Dtype) 1., o_hat_data);
        // Compute recurrent forward propagation

        for (int t = 0; t < T_; ++t) {
            Dtype *h_t = top_data + top_.offset(t);
            Dtype *zr_hat_t = zr_hat_data + zr_hat_.offset(t);
            Dtype *gate_t = gate_data + zro_.offset(t);
            Dtype *m_t = m_data + m_.offset(t);
            Dtype *o_hat_t = o_hat_data + o_hat_.offset(t);
            const Dtype *h_t_1 = t > 0 ? (top_data + top_.offset(t - 1)) : h_0_.gpu_data();

            // Hidden-to-hidden propagation
            // h_to_gate_zr := h_t_1 * weight_h^T
            // MxN := MxK * KxN = N_ x 2*H_
            caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, N_, 2 * H_, H_, (Dtype) 1.,
                                  h_t_1, weight_h, (Dtype) 0., h_to_gate_zr);

            ClipAdd<Dtype><<<CAFFE_GET_BLOCKS(2 * N_ * H_), CAFFE_CUDA_NUM_THREADS>>>(
                    2 * N_ * H_, t, h_to_gate_zr_.gpu_data(), zr_hat_t);
            CUDA_POST_KERNEL_CHECK;
            GRUForwardZRM<Dtype><<<CAFFE_GET_BLOCKS(N_ * H_), CAFFE_CUDA_NUM_THREADS>>>(
                    N_ * H_, H_, zr_hat_t, h_t_1, gate_t, m_t);
            CUDA_POST_KERNEL_CHECK;
            caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, N_, H_, H_, (Dtype) 1.,
                                  m_t, weight_om, (Dtype) 0., h_to_gate_m);

            ClipAdd<Dtype><<<CAFFE_GET_BLOCKS(N_ * H_), CAFFE_CUDA_NUM_THREADS>>>(
                    N_ * H_, t, h_to_gate_m_.gpu_data(), o_hat_t);
            CUDA_POST_KERNEL_CHECK;
            GRUForwardOH<Dtype><<<CAFFE_GET_BLOCKS(N_ * H_), CAFFE_CUDA_NUM_THREADS>>>(
                    N_ * H_, H_, o_hat_t, h_t_1, gate_t, h_t);
            CUDA_POST_KERNEL_CHECK;

        }
    }

    template<typename Dtype>
    void GRULayer<Dtype>::Backward_gpu(const vector<Blob<Dtype> *> &top,
                                       const vector<bool> &propagate_down,
                                       const vector<Blob<Dtype> *> &bottom) {
        const Dtype *top_data = top_.gpu_data();
        const Dtype *bottom_data = bottom[0]->gpu_data();
        const Dtype *weight_zr_x = this->blobs_[0]->gpu_data();
        const Dtype *weight_zr_h = this->blobs_[1]->gpu_data();
        const Dtype *weight_om = this->blobs_[2]->gpu_data();
        const Dtype *weight_ox = this->blobs_[3]->gpu_data();
        const Dtype *zro_data = zro_.gpu_data();
        const Dtype *zr_hat_data = zr_hat_.gpu_data();

        Dtype *top_diff = top_.mutable_gpu_diff();
        Dtype *zr_hat_diff = zr_hat_.mutable_gpu_diff();
        Dtype *zro_diff = zro_.mutable_gpu_diff();
        Dtype *m_diff = m_.mutable_gpu_diff();
        Dtype *o_hat_diff = o_hat_.mutable_gpu_diff();




        for (int t = T_ - 1; t >= 0; --t) {
            Dtype *dh_t = top_diff + top_.offset(t);// dL/dh_t
            Dtype *zro_diff_t = zro_diff + zro_.offset(t);// dL/dz_t, dL/dr_t, dL/do_t
            Dtype *zr_hat_diff_t = zr_hat_diff + zr_hat_.offset(t); // dL/dz_t_hat, dL/dr_t_hat, dL/do_t_hat

            Dtype *m_diff_t = m_diff + m_.offset(t);
            Dtype *o_hat_diff_t = o_hat_diff + o_hat_.offset(t);
            Dtype *d_ph = d_ph_.mutable_cpu_data();
            Dtype *dh_t_1 = t > 0 ? top_diff + top_.offset(t - 1) : h_0_.mutable_gpu_diff();

            const Dtype *h_t = top_data + top_.offset(t);
            const Dtype *zro_t = zro_data + zro_.offset(t);
            const Dtype *zr_hat_t = zr_hat_data + zr_hat_.offset(t);
            const Dtype *h_t_1 = t > 0 ? (h_t - top_.offset(1)) : h_0_.gpu_data();

//            GRUBackwardZO_Ohat(const int nthreads, const int H,
//            const Dtype *zro, const Dtype *dh, const Dtype *h_t_1, Dtype *zro_diff, Dtype *o_hat_diff)

            GRUBackwardZO_Ohat<Dtype><<<CAFFE_GET_BLOCKS(N_ * H_), CAFFE_CUDA_NUM_THREADS>>>(
                    N_ * H_, H_, zro_t, dh_t, h_t_1, zro_diff_t, o_hat_diff_t);
            CUDA_POST_KERNEL_CHECK;
            caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, N_, H_, H_, (Dtype) 1.,
                                  o_hat_diff_t, weight_om, (Dtype) 0., m_diff_t);
//            GRUBackwardR_ZRhat_h1(const int nthreads, const int H,
//            const Dtype *zro, const Dtype *dh, const Dtype *m_diff, const Dtype *h_t_1, Dtype *zro_diff, Dtype *zr_hat_diff, Dtype *d_ph)

            GRUBackwardR_ZRhat_h1<Dtype><<<CAFFE_GET_BLOCKS(N_ * H_), CAFFE_CUDA_NUM_THREADS>>>(
                    N_ * H_, H_, zro_t, dh_t, m_.gpu_diff() + m_.offset(t), h_t_1, weight_zr_h, zro_diff_t, zr_hat_diff_t, d_ph);
            CUDA_POST_KERNEL_CHECK;
//            caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_, H_, 2 * H_,
//                                  (Dtype) 1., zr_hat_diff_t,
//                                  weight_zr_h, (Dtype) 1., d_ph_.mutable_gpu_data());
//
//            ClipAdd<Dtype><<<CAFFE_GET_BLOCKS(N_ * H_), CAFFE_CUDA_NUM_THREADS>>>(
//                    N_ * H_, H_, t, clip_t, d_ph_.gpu_data(), dh_t_1);
//            CUDA_POST_KERNEL_CHECK;


        } //for (int t = T_-1; t >= 0; --t)
//        std::cout<<"test0 ";

        if (this->param_propagate_down_[0]) {
            // Gradient w.r.t. input-to-hidden weight矩阵与向量的导数
            // dL/dW_zx = dL/dz_t_hat * dz_t_hat/dW_xi
            //          = x_t^T * dL/dz_t_hat
            // dL/dW_zx dL/dW_rx

            caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
                                  2 * H_, I_, T_ * N_,
                                  (Dtype) 1., zr_hat_diff, bottom_data,
                                  (Dtype) 1., this->blobs_[0]->mutable_gpu_diff());

        }

        if (this->param_propagate_down_[1]) {
            // Gradient w.r.t. hidden-to-hidden weight

            caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
                                  2 * H_, H_, (T_ - 1) * N_,
                                  (Dtype) 1., zr_hat_diff + zr_hat_.offset(1), top_data,
                                  (Dtype) 1., this->blobs_[1]->mutable_gpu_diff());

            // Add Gradient from previous time-step
            // h0
            caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
                                  2 * H_, H_, N_,
                                  (Dtype) 1., zr_hat_diff, h_0_.gpu_data(),
                                  (Dtype) 1., this->blobs_[1]->mutable_gpu_diff());

        }
        if (this->param_propagate_down_[2]) {
            // dL/dW_om = dL/do_t_hat * do_t_hat/dW_om
            caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, H_, H_, T_ * N_, (Dtype) 1.,
                                  o_hat_diff, m_.gpu_data(),
                                  (Dtype) 1., this->blobs_[2]->mutable_gpu_diff());
        }
        if (this->param_propagate_down_[3]) {
            //dL/dW_ox
            caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, H_, I_, T_ * N_, (Dtype) 1.,
                                  o_hat_diff, bottom_data, (Dtype) 1.,
                                  this->blobs_[3]->mutable_gpu_diff());
        }
        if (this->param_propagate_down_[4]) {
            // Gradient w.r.t. bias1
            caffe_gpu_gemv<Dtype>(CblasTrans,
                                  T_ * N_, 2 * H_,
                                  (Dtype) 1., zr_hat_diff, bias_multiplier_.gpu_data(),
                                  (Dtype) 1., this->blobs_[4]->mutable_gpu_diff());
        }

        if (this->param_propagate_down_[5]) {
            // Gradient w.r.t. bias2
            caffe_gpu_gemv<Dtype>(CblasTrans, T_ * N_, H_, (Dtype) 1., o_hat_diff,
                                  bias_multiplier_.gpu_data(), (Dtype) 1.,
                                  this->blobs_[5]->mutable_gpu_diff());
        }
        if (propagate_down[0]) {
            // dL/dx_t

            caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, T_ * N_, I_, 2 * H_, (Dtype) 1.,
                                  zr_hat_diff, weight_zr_x, (Dtype) 0., bottom[0]->mutable_gpu_diff());

            caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, T_ * N_, I_, H_, (Dtype) 1.,
                                  o_hat_diff, weight_ox, (Dtype) 1., bottom[0]->mutable_gpu_diff());

        }
//        std::cout<<"test7 ";
    }

    INSTANTIATE_LAYER_GPU_FUNCS(GRULayer);

}  // namespace caffe