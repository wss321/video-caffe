/*
 * tensor_transpose_layer.hpp
 *
 *  Created on: Nov 10, 2017
 *      Author: cfeng
 */

#ifndef INCLUDE_CAFFE_LAYERS_TENSOR_TRANSPOSE_LAYER_HPP_
#define INCLUDE_CAFFE_LAYERS_TENSOR_TRANSPOSE_LAYER_HPP_


#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief transpose input tensor X and output $Y=X(o)$ where o is the axes order
 *
 * Input:
 * X: <B1xB2...xBn>
 * Output:
 * Y: <Bo(1)xBo(2)...xBo(n)>
 */
template <typename Dtype>
class TensorTransposeLayer : public Layer<Dtype> {
 public:
  explicit TensorTransposeLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "TensorTranspose"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
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

  Blob<unsigned int> order_; //set from caffe.proto
  Blob<unsigned int> Xstride_;
  Blob<unsigned int> Ystride_;
};

}  // namespace caffe


#endif /* INCLUDE_CAFFE_LAYERS_TENSOR_TRANSPOSE_LAYER_HPP_ */
