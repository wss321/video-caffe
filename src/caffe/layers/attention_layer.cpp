#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/attention_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
    template<class Dtype>
    void AttentionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
//        const int num_input_blobs = bottom.size();
        LOG(INFO)<<"top size:"<<top.size();
        CHECK(bottom.size()==3||bottom.size()==1)
        << "bottom must have 1 or 3 blobs-- (#K, #Q, #V)";
        for (int i = 0; i < bottom.size(); ++i) {
            CHECK_EQ(bottom[i]->num_axes(), 3)
                << "bottom[" << i << "] must have 3 axes -- (#batch, #feature num, #feature dim)";
        }

        B_ = bottom[0]->shape(0);//batch size
        N_ = bottom[0]->shape(1);//feature num
        D_ = bottom[0]->shape(2);//feature dim
        LOG(INFO) << "Initializing attention layer: assuming input batch contains "
                  << N_ << " features of " << D_ << " feature dim.";

        // If expose_hidden is set, we take as input and produce as output
        // the hidden state blobs at the first and last timesteps.
        has_bias_ = this->layer_param_.attention_param().bias_term();

        // Get (recurrent) input/output names.
        vector<string> output_names;
        AttentionOutputBlobNames(&output_names);

        // Create a NetParameter; setup the inputs that aren't unique to particular
        // recurrent architectures.
        NetParameter net_param;

        LayerParameter *input_layer_param = net_param.add_layer();
        input_layer_param->set_type("Input");
        InputParameter *input_param = input_layer_param->mutable_input_param();
        input_layer_param->add_top("k");
        input_layer_param->add_top("q");
        input_layer_param->add_top("v");

        for (int i = 0; i < bottom.size(); ++i) {
            BlobShape input_shape;
            for (int j = 0; j < bottom[i]->num_axes(); ++j) {
                input_shape.add_dim(bottom[i]->shape(j));
            }
            input_param->add_shape()->CopyFrom(input_shape);
        }

        // Call the child's FillUnrolledNet implementation to specify the unrolled
        // recurrent architecture.
        this->FillUnrolledNet(&net_param);

        // Prepend this layer's name to the names of each layer in the unrolled net.
        const string &layer_name = this->layer_param_.name();
        if (layer_name.size()) {
            for (int i = 0; i < net_param.layer_size(); ++i) {
                LayerParameter *layer = net_param.mutable_layer(i);
                layer->set_name(layer_name + "_" + layer->name());
            }
        }

        // Add "pseudo-losses" to all outputs to force backpropagation.
        // (Setting force_backward is too aggressive as we may not need to backprop to
        // all inputs, e.g., the sequence continuation indicators.)
        vector<string> pseudo_losses(output_names.size());
        for (int i = 0; i < output_names.size(); ++i) {
            LayerParameter* layer = net_param.add_layer();
            pseudo_losses[i] = output_names[i] + "_pseudoloss";
            layer->set_name(pseudo_losses[i]);
            layer->set_type("Reduction");
            layer->add_bottom(output_names[i]);
            layer->add_top(pseudo_losses[i]);
            layer->add_loss_weight(1);
        }

        // Create the unrolled net.
        unrolled_net_.reset(new Net<Dtype>(net_param));

        // Setup pointers to paired recurrent inputs/outputs.
        att_input_blobs_.push_back(CHECK_NOTNULL(unrolled_net_->blob_by_name("k").get()));
        att_input_blobs_.push_back(CHECK_NOTNULL(unrolled_net_->blob_by_name("q").get()));
        att_input_blobs_.push_back(CHECK_NOTNULL(unrolled_net_->blob_by_name("v").get()));


        att_output_blobs_.resize(output_names.size());
        for (int i = 0; i < output_names.size(); ++i) {
            att_output_blobs_[i] =
                    CHECK_NOTNULL(unrolled_net_->blob_by_name(output_names[i]).get());
        }

        // Setup pointers to outputs.
        CHECK_EQ(top.size(), output_names.size())
            << "OutputBlobNames must provide an output blob name for each top.";

        // This layer's parameters are any parameters in the layers of the unrolled
        // net. We only want one copy of each parameter, so check that the parameter
        // is "owned" by the layer, rather than shared with another.
        this->blobs_.clear();
        for (int i = 0; i < unrolled_net_->params().size(); ++i) {
            if (unrolled_net_->param_owners()[i] == -1) {
                LOG(INFO) << "Adding parameter " << i << ": "
                          << unrolled_net_->param_display_names()[i];
                this->blobs_.push_back(unrolled_net_->params()[i]);
            }
        }
        // Check that param_propagate_down is set for all of the parameters in the
        // unrolled net; set param_propagate_down to true in this layer.
        for (int i = 0; i < unrolled_net_->layers().size(); ++i) {
            for (int j = 0; j < unrolled_net_->layers()[i]->blobs().size(); ++j) {
                CHECK(unrolled_net_->layers()[i]->param_propagate_down(j))
                << "param_propagate_down not set for layer " << i << ", param " << j;
            }
        }
        this->param_propagate_down_.clear();
        this->param_propagate_down_.resize(this->blobs_.size(), true);

//         Set the diffs of attention score to 0 -- we can't backpropagate across
//         batches.
//        caffe_set(att_output_blobs_[1]->count(), Dtype(0),
//                  att_output_blobs_[1]->mutable_cpu_diff());
        last_layer_index_ = unrolled_net_->layer_names().size() - 1- pseudo_losses.size();

    }

    template<typename Dtype>
    void AttentionLayer<Dtype>::AttentionOutputBlobNames(vector<string> *names) const {
        names->resize(1);
        (*names)[0] = "attention";
//        (*names)[1] = "score";
    }


    template <typename Dtype>
    void AttentionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top) {
        CHECK(bottom.size()==3||bottom.size()==1)
            << "bottom must have 1 or 3 blobs-- (#K, #Q, #V)";
        CHECK_GE(bottom[0]->num_axes(), 2)
            << "bottom[0] must have at least 2 axes -- (#timesteps, #streams, ...)";

        if (bottom.size()==3){
            CHECK(bottom[0]->num_axes()==bottom[1]->num_axes()&&bottom[2]->num_axes()== bottom[1]->num_axes())
            << "bottoms must have same dimensions-- (#K, #Q, #V)";

        CHECK(bottom[2]->shape(0)==bottom[1]->shape(0)&&bottom[0]->shape(0)==bottom[1]->shape(0))
            << "bottoms must have same batch size-- (#K, #Q, #V)";
            }

        CHECK(att_output_blobs_.size()==top.size())
            << "number output blobs must be same as top size";

        B_ = bottom[0]->shape(0);
        unrolled_net_->Reshape();

        //复制
        if (bottom.size()==3){
            for (int i = 0; i < att_input_blobs_.size(); ++i) {
                att_input_blobs_[i]->ReshapeLike(*bottom[i]);
                att_input_blobs_[i]->ShareData(*bottom[i]);
                att_input_blobs_[i]->ShareDiff(*bottom[i]);
            }
        }else{
            for (int i = 0; i < att_input_blobs_.size(); ++i) {
                att_input_blobs_[i]->ReshapeLike(*bottom[0]);
                att_input_blobs_[i]->ShareData(*bottom[0]);
                att_input_blobs_[i]->ShareDiff(*bottom[0]);
            }
        }

        for (int i = 0; i < att_output_blobs_.size(); ++i) {
            top[i]->ReshapeLike(*att_output_blobs_[i]);
            top[i]->ShareData(*att_output_blobs_[i]);
            top[i]->ShareDiff(*att_output_blobs_[i]);
        }
    }

    template <typename Dtype>
    void AttentionLayer<Dtype>::Reset() {
    }

    template <typename Dtype>
    void AttentionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                            const vector<Blob<Dtype>*>& top) {
        if (this->phase_ == TEST) {
            unrolled_net_->ShareWeights();
        }
        unrolled_net_->ForwardTo(last_layer_index_);
    }

    template <typename Dtype>
    void AttentionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                             const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
//        CHECK(!propagate_down[1]) << "Cannot backpropagate to sequence indicators.";
//
//        // TODO: skip backpropagation to inputs and parameters inside the unrolled
        // net according to propagate_down[0] and propagate_down[2]. For now just
        // backprop to inputs and parameters unconditionally, as either the inputs or
        // the parameters do need backward (or Net would have set
        // layer_needs_backward_[i] == false for this layer).
        unrolled_net_->BackwardFrom(last_layer_index_);
    }


    template<typename Dtype>
    void AttentionLayer<Dtype>::FillUnrolledNet(NetParameter *net_param) const {
        const int num_output = this->layer_param_.attention_param().num_output();
        CHECK_GT(num_output, 0) << "num_output must be positive";
        const FillerParameter &weight_filler =
                this->layer_param_.attention_param().weight_filler();
        const FillerParameter& bias_filler =
                this->layer_param_.attention_param().bias_filler();
//        const int n_head = this->layer_param_.attention_param().n_head();
//
//        CHECK_EQ(num_output%n_head, 0) << "num_output must be divisible by n_head";
//        int d_k = num_output/n_head;

        // Add generic LayerParameter's (without bottoms/tops) of layer types we'll
        // use to save redundant code.
        LayerParameter W_param;//Wx
        W_param.set_type("InnerProduct");
        W_param.mutable_inner_product_param()->set_num_output(num_output);
        if (has_bias_){
            W_param.mutable_inner_product_param()->set_bias_term(true);
            W_param.mutable_inner_product_param()->
                    mutable_bias_filler()->CopyFrom(bias_filler);

        }
        W_param.mutable_inner_product_param()->set_axis(2);
        W_param.mutable_inner_product_param()->
                mutable_weight_filler()->CopyFrom(weight_filler);
//        W_param.add_propagate_down(true);


        {
            // 3 WXt
            LayerParameter *Wk_param = net_param->add_layer();
            Wk_param->CopyFrom(W_param);
            Wk_param->set_name("wk");
            Wk_param->add_param()->set_name("Wk");
            Wk_param->add_param()->set_name("bk");
            Wk_param->add_bottom("k");
            Wk_param->add_top("wk");
            Wk_param->add_propagate_down(true);

            LayerParameter *Wq_param = net_param->add_layer();
            Wq_param->CopyFrom(W_param);
            Wq_param->set_name("wq");
            Wq_param->add_param()->set_name("Wq");
            Wq_param->add_param()->set_name("bq");
            Wq_param->add_bottom("q");
            Wq_param->add_top("wq");
            Wq_param->add_propagate_down(true);

            LayerParameter *Wv_param = net_param->add_layer();
            Wv_param->CopyFrom(W_param);
            Wv_param->set_name("wv");
            Wv_param->add_param()->set_name("Wv");
            Wv_param->add_param()->set_name("bv");
            Wv_param->add_bottom("v");
            Wv_param->add_top("wv");
            Wv_param->add_propagate_down(true);

            LayerParameter *transpose_param = net_param->add_layer();
            transpose_param->set_type("TensorTranspose");
            transpose_param->mutable_tensor_transpose_param()->add_order(0);
            transpose_param->mutable_tensor_transpose_param()->add_order(2);
            transpose_param->mutable_tensor_transpose_param()->add_order(1);
            transpose_param->add_bottom("wk");
            transpose_param->add_top("kT");
//            transpose_param->add_propagate_down(true);

            LayerParameter *matmul_param = net_param->add_layer();
            matmul_param->set_type("MatrixMultiplication");
            matmul_param->set_name("qkT");
            matmul_param->add_bottom("wq");
            matmul_param->add_bottom("kT");
            matmul_param->add_top("qkT");
//            matmul_param->add_propagate_down(true);
//            matmul_param->add_propagate_down(true);


            LayerParameter* scale_shift=net_param->add_layer();
            scale_shift->set_type("Power");
            scale_shift->set_name("scale");
            scale_shift->mutable_power_param()->set_shift(0.0);
            scale_shift->mutable_power_param()->set_scale(sqrt(D_));
            scale_shift->mutable_power_param()->set_power(1.0);
            scale_shift->add_bottom("qkT");
            scale_shift->add_top("sqkT");
//            scale_shift->add_propagate_down(true);

            LayerParameter *softmax_param = net_param->add_layer();
            softmax_param->set_type("Softmax");
            softmax_param->set_name("softmax");
            softmax_param->mutable_softmax_param()->set_axis(2);
            softmax_param->add_bottom("sqkT");
            softmax_param->add_top("score");
//            softmax_param->add_propagate_down(true);

            LayerParameter *matmul_param2 = net_param->add_layer();
            matmul_param2->set_type("MatrixMultiplication");
            matmul_param2->set_name("attention");
            matmul_param2->add_bottom("score");
            matmul_param2->add_bottom("wv");
            matmul_param2->add_top("attention");
//            matmul_param2->add_propagate_down(true);
//            matmul_param2->add_propagate_down(true);
        }

    }
#ifdef CPU_ONLY
    STUB_GPU(AttentionLayer);
#endif
    INSTANTIATE_CLASS(AttentionLayer);
    REGISTER_LAYER_CLASS(Attention);

}  // namespace caffe
