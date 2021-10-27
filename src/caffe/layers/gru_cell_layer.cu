#include <vector>
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/gru_cell_layer.hpp"

namespace caffe {

    template<typename Dtype>
    __device__ Dtype cuda_sigmoid(Dtype x) {
        return 1. / (1. + exp(-x));
    }

    template<typename Dtype>
    __device__ Dtype cuda_sigmoid_diff(Dtype x) {
        return x * (1. - x);
    }

    template<typename Dtype>
    __device__ Dtype cuda_tanh(Dtype x) {
        return 2. * cuda_sigmoid(2. * x) - 1.;
    }

    template<typename Dtype>
    __device__ Dtype cuda_tanh_diff(Dtype x) {
        return (1. - x * x);
    }

    template<typename Dtype>
    __global__ void ForwardSigmoid(
            int n,
            Dtype *z,
            Dtype *r) {
        CUDA_KERNEL_LOOP(idx, n) {
            z[idx] = cuda_sigmoid(z[idx]);
            r[idx] = cuda_sigmoid(r[idx]);
        }
    }

    template<typename Dtype>
    __global__ void ForwardTanH(
            int n,
            Dtype *o) {
        CUDA_KERNEL_LOOP(idx, n) {
            o[idx] = cuda_tanh(o[idx]);
        }
    }

    template<typename Dtype>
    __global__ void ForwardNextState(
            int n,
            const Dtype *z,
            const Dtype *o,
            const Dtype *ht_1,
            Dtype *ht) {
        CUDA_KERNEL_LOOP(idx, n) {
            ht[idx] = ((Dtype) 1. - z[idx]) * o[idx] + z[idx] * ht_1[idx];
        }
    }

    template<typename Dtype>
    __global__ void BackwardZODiff(
            int n,
            const Dtype *top_diff,
            const Dtype *ht_1,
            const Dtype *z,
            const Dtype *o,
            Dtype *dz_hat,
            Dtype *do_hat) {
        CUDA_KERNEL_LOOP(idx, n) {
            dz_hat[idx] = top_diff[idx] * (ht_1[idx] - o[idx]) * cuda_sigmoid_diff(z[idx]);
            do_hat[idx] = top_diff[idx] * (Dtype(1.0) - z[idx]) * cuda_tanh_diff(o[idx]);
        }
    }

    template<typename Dtype>
    __global__ void BackwardRDiff(
            int n,
            const Dtype *z,
            const Dtype *r,
            const Dtype *ht_1,
            const Dtype *dm,
            const Dtype *top_diff,
            Dtype *dh_1,
            Dtype *dr_hat) {
        CUDA_KERNEL_LOOP(idx, n) {
            dr_hat[idx] = dm[idx] * ht_1[idx] * cuda_sigmoid_diff(r[idx]);
            dh_1[idx] += dm[idx] * r[idx] + top_diff[idx] * z[idx];
        }
    }

    template<typename Dtype>
    __global__ void BackwardRDiff(
            int n,
            const Dtype *dm,
            const Dtype *ht_1,
            const Dtype *r,
            Dtype *dr_hat) {
        CUDA_KERNEL_LOOP(idx, n) {
            dr_hat[idx] = dm[idx] * ht_1[idx] * sigmoid_diff(r[idx]);
        }
    }

    template<typename Dtype>
    void GRUCellLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype> *> &bottom,
                                          const vector<Blob<Dtype> *> &top) {
        const Dtype *x = bottom[0]->gpu_data();
        const Dtype *ht_1 = bottom[1]->gpu_data();

        const Dtype *Wzx = this->blobs_[0]->gpu_data();
        const Dtype *Wrx = this->blobs_[1]->gpu_data();
        const Dtype *Wox = this->blobs_[2]->gpu_data();

        const Dtype *Wzh = this->blobs_[3]->gpu_data();
        const Dtype *Wrh = this->blobs_[4]->gpu_data();
        const Dtype *Wom = this->blobs_[5]->gpu_data();

        const Dtype *Bz = this->blobs_[6]->gpu_data();
        const Dtype *Br = this->blobs_[7]->gpu_data();
        const Dtype *Bo = this->blobs_[8]->gpu_data();

        Dtype *ht = top[0]->mutable_gpu_data();

        Dtype *z = z_gates_data_buffer_->mutable_gpu_data();
        Dtype *r = r_gates_data_buffer_->mutable_gpu_data();
        Dtype *o = o_gates_data_buffer_->mutable_gpu_data();
        Dtype *m = m_data_buffer_->mutable_gpu_data();
        const int num_elem = M_ * N_;

        // pre_gate
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_,
                              (Dtype) 1., x, Wzx,
                              (Dtype) 0., z);
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, N_,
                              (Dtype) 1., ht_1, Wzh,
                              (Dtype) 1., z);

        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_,
                              (Dtype) 1., x, Wrx,
                              (Dtype) 0., r);
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, N_,
                              (Dtype) 1., ht_1, Wrh,
                              (Dtype) 1., r);

        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_,
                              (Dtype) 1., x, Wox,
                              (Dtype) 0., o);

        if (has_bias) {
            caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype) 1.,
                                  bias_multiplier_.gpu_data(), Bz, (Dtype) 1., z);
            caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype) 1.,
                                  bias_multiplier_.gpu_data(), Br, (Dtype) 1., r);
            caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype) 1.,
                                  bias_multiplier_.gpu_data(), Bo, (Dtype) 1., o);
        
        }
        
        ForwardSigmoid<Dtype><<<CAFFE_GET_BLOCKS(num_elem),
        CAFFE_CUDA_NUM_THREADS>>>(
                num_elem, z, r);
        CUDA_POST_KERNEL_CHECK;

        caffe_gpu_mul(num_elem, r, ht_1, m);//m=r*ht-1

        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, N_,
                              (Dtype) 1., m, Wom,
                              (Dtype) 1., o);
        ForwardTanH<Dtype><<<CAFFE_GET_BLOCKS(num_elem),
        CAFFE_CUDA_NUM_THREADS>>>(
                num_elem, o);
        CUDA_POST_KERNEL_CHECK;

        ForwardNextState<Dtype><<<CAFFE_GET_BLOCKS(num_elem),
        CAFFE_CUDA_NUM_THREADS>>>(
                num_elem, z, o, ht_1, ht);
        CUDA_POST_KERNEL_CHECK;
    }

    template<typename Dtype>
    void GRUCellLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype> *> &top,
                                           const vector<bool> &propagate_down,
                                           const vector<Blob<Dtype> *> &bottom) {
        for (int i = 0; i < 2; ++i) {
            caffe_gpu_set(bottom[i]->count(), Dtype(0), bottom[i]->mutable_gpu_diff());
        }
        const Dtype *x = bottom[0]->gpu_data();
        const Dtype *ht_1 = bottom[1]->gpu_data();

        const Dtype *Wzx = this->blobs_[0]->gpu_data();
        const Dtype *Wrx = this->blobs_[1]->gpu_data();
        const Dtype *Wox = this->blobs_[2]->gpu_data();

        const Dtype *Wzh = this->blobs_[3]->gpu_data();
        const Dtype *Wrh = this->blobs_[4]->gpu_data();
        const Dtype *Wom = this->blobs_[5]->gpu_data();

        const Dtype *z = z_gates_data_buffer_->gpu_data();
        const Dtype *r = r_gates_data_buffer_->gpu_data();
        const Dtype *o = o_gates_data_buffer_->gpu_data();
        const Dtype *m = m_data_buffer_->gpu_data();

        Dtype *dz_hat = z_gates_data_buffer_->mutable_gpu_diff();
        Dtype *dr_hat = r_gates_data_buffer_->mutable_gpu_diff();
        Dtype *do_hat = o_gates_data_buffer_->mutable_gpu_diff();
        Dtype *dm = m_data_buffer_->mutable_gpu_diff();

        Dtype *dx = bottom[0]->mutable_gpu_diff();
        Dtype *dh_1 = bottom[1]->mutable_gpu_diff();
        const Dtype *top_diff = top[0]->cpu_diff();
        const int num_elem = M_ * N_;

        // pre-gate diff:dL/dz_har ...
        BackwardZODiff<Dtype><<<CAFFE_GET_BLOCKS(num_elem),
        CAFFE_CUDA_NUM_THREADS>>>(
                num_elem, top_diff, ht_1, z, o, dz_hat, do_hat);
        CUDA_POST_KERNEL_CHECK;
        //dm
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, N_, (Dtype) 1.,
                              do_hat, Wom, (Dtype) 0., dm);

        BackwardRDiff<Dtype><<<CAFFE_GET_BLOCKS(num_elem),
        CAFFE_CUDA_NUM_THREADS>>>(
                num_elem, z, r, ht_1, dm, top_diff, dh_1, dr_hat);
        CUDA_POST_KERNEL_CHECK;

        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, N_, (Dtype) 1.,
                              dz_hat, Wzh, (Dtype) 1., dh_1);//dht-1
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, N_, (Dtype) 1.,
                              dr_hat, Wrh, (Dtype) 1., dh_1);//dht-1

        if (propagate_down[0]) {
            // dx
            caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, K_, N_, (Dtype) 1.,
                                  dz_hat, Wzx, (Dtype) 1., dx);
            caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, K_, N_, (Dtype) 1.,
                                  dr_hat, Wrx, (Dtype) 1., dx);
            caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, K_, N_, (Dtype) 1.,
                                  do_hat, Wox, (Dtype) 1., dx);
        }

        Dtype *Wzx_diff = this->blobs_[0]->mutable_gpu_diff();
        Dtype *Wrx_diff = this->blobs_[1]->mutable_gpu_diff();
        Dtype *Wox_diff = this->blobs_[2]->mutable_gpu_diff();
        Dtype *Wzh_diff = this->blobs_[3]->mutable_gpu_diff();
        Dtype *Wrh_diff = this->blobs_[4]->mutable_gpu_diff();
        Dtype *Wom_diff = this->blobs_[5]->mutable_gpu_diff();

        Dtype *Bz_diff = this->blobs_[6]->mutable_gpu_diff();
        Dtype *Br_diff = this->blobs_[7]->mutable_gpu_diff();
        Dtype *Bo_diff = this->blobs_[8]->mutable_gpu_diff();

        if (this->param_propagate_down_[0])
            caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
                                  feature_dim_, input_data_dim_, batch_size_,
                                  (Dtype) 1., dz_hat, x,
                                  (Dtype) 1., Wzx_diff);
        if (this->param_propagate_down_[1])
            caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
                                  feature_dim_, input_data_dim_, batch_size_,
                                  (Dtype) 1., dr_hat, x,
                                  (Dtype) 1., Wrx_diff);
        if (this->param_propagate_down_[2])
            caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
                                  feature_dim_, input_data_dim_, batch_size_,
                                  (Dtype) 1., do_hat, x,
                                  (Dtype) 1., Wox_diff);
        if (this->param_propagate_down_[3])
            caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
                                  feature_dim_, feature_dim_, batch_size_,
                                  (Dtype) 1., dz_hat, ht_1,
                                  (Dtype) 1., Wzh_diff);
        if (this->param_propagate_down_[4])
            caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
                                  feature_dim_, feature_dim_, batch_size_,
                                  (Dtype) 1., dr_hat, ht_1,
                                  (Dtype) 1., Wrh_diff);
        if (this->param_propagate_down_[5])
            caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
                                  feature_dim_, feature_dim_, batch_size_,
                                  (Dtype) 1., do_hat, m,
                                  (Dtype) 1., Wom_diff);
        if (this->param_propagate_down_[6])
            caffe_gpu_gemv<Dtype>(CblasTrans,
                                  M_, N_,
                                  (Dtype) 1., dz_hat, bias_multiplier_.gpu_data(),
                                  (Dtype) 1., Bz_diff);
        if (this->param_propagate_down_[7])
            caffe_gpu_gemv<Dtype>(CblasTrans,
                                  M_, N_,
                                  (Dtype) 1., dr_hat, bias_multiplier_.gpu_data(),
                                  (Dtype) 1., Br_diff);
        if (this->param_propagate_down_[8])
            caffe_gpu_gemv<Dtype>(CblasTrans,
                                  M_, N_,
                                  (Dtype) 1., do_hat, bias_multiplier_.gpu_data(),
                                  (Dtype) 1., Bo_diff);
    }

    INSTANTIATE_LAYER_GPU_FUNCS(GRUCellLayer);

}  // namespace caffe