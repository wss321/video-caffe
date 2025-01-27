#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/grucf_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

    template <typename Dtype>
    void GRUCFLayer<Dtype>::RecurrentInputBlobNames(vector<string>* names) const {
        names->resize(1);
        (*names)[0] = "h_0";
    }

    template <typename Dtype>
    void GRUCFLayer<Dtype>::RecurrentOutputBlobNames(vector<string>* names) const {
        names->resize(1);
        (*names)[0] = "h_" + format_int(this->T_);
    }

    template <typename Dtype>
    void GRUCFLayer<Dtype>::RecurrentInputShapes(vector<BlobShape>* shapes) const {
        const int num_output = this->layer_param_.recurrent_param().num_output();
        const int num_blobs = 1;
        shapes->resize(num_blobs);
        for (int i = 0; i < num_blobs; ++i) {
            (*shapes)[i].Clear();
            (*shapes)[i].add_dim(1);  // a single timestep
            (*shapes)[i].add_dim(this->N_);
            (*shapes)[i].add_dim(num_output);
        }
    }

    template <typename Dtype>
    void GRUCFLayer<Dtype>::OutputBlobNames(vector<string>* names) const {
        names->resize(1);
        (*names)[0] = "h";
    }

    template <typename Dtype>
    void GRUCFLayer<Dtype>::FillUnrolledNet(NetParameter* net_param) const {
        const int num_output = this->layer_param_.recurrent_param().num_output();
        CHECK_GT(num_output, 0) << "num_output must be positive";
        const FillerParameter& weight_filler =
                this->layer_param_.recurrent_param().weight_filler();
//        const FillerParameter& bias_filler =
//                this->layer_param_.recurrent_param().bias_filler();

        // Add generic LayerParameter's (without bottoms/tops) of layer types we'll
        // use to save redundant code.
        LayerParameter Wx_hidden_param;//Wx
        Wx_hidden_param.set_type("InnerProduct");
        Wx_hidden_param.mutable_inner_product_param()->set_num_output(num_output*3);
        Wx_hidden_param.mutable_inner_product_param()->set_bias_term(false);
        Wx_hidden_param.mutable_inner_product_param()->set_axis(2);
        Wx_hidden_param.mutable_inner_product_param()->
                mutable_weight_filler()->CopyFrom(weight_filler);

//        LayerParameter biased_hidden_param(Wx_hidden_param);
//        biased_hidden_param.mutable_inner_product_param()->set_bias_term(true);
//        biased_hidden_param.mutable_inner_product_param()->
//                mutable_bias_filler()->CopyFrom(bias_filler);

        LayerParameter Wh_hidden_param;//Wh
        Wh_hidden_param.set_type("InnerProduct");
        Wh_hidden_param.mutable_inner_product_param()->set_num_output(num_output);
        Wh_hidden_param.mutable_inner_product_param()->set_bias_term(false);
        Wh_hidden_param.mutable_inner_product_param()->set_axis(2);
        Wh_hidden_param.mutable_inner_product_param()->
                mutable_weight_filler()->CopyFrom(weight_filler);

//        LayerParameter biased_hidden_param_h(Wh_hidden_param);
//        biased_hidden_param_h.mutable_inner_product_param()->set_bias_term(true);
//        biased_hidden_param_h.mutable_inner_product_param()->
//                mutable_bias_filler()->CopyFrom(bias_filler);

        LayerParameter sum_param;
        sum_param.set_type("Eltwise");
        sum_param.mutable_eltwise_param()->set_operation(
                EltwiseParameter_EltwiseOp_SUM);

        LayerParameter prod_param;
        prod_param.set_type("Eltwise");
        prod_param.mutable_eltwise_param()->set_operation(
                EltwiseParameter_EltwiseOp_PROD);

        LayerParameter scale_shift;
        scale_shift.set_type("Scale");
        scale_shift.mutable_power_param()->set_shift(1.0);
        scale_shift.mutable_power_param()->set_scale(-1.0);
        scale_shift.mutable_power_param()->set_power(1.0);

        LayerParameter slice_param_axis0;
        slice_param_axis0.set_type("Slice");
        slice_param_axis0.mutable_slice_param()->set_axis(0);

        vector<BlobShape> input_shapes;
        RecurrentInputShapes(&input_shapes);
        CHECK_EQ(1, input_shapes.size());

        LayerParameter* input_layer_param = net_param->add_layer();
        input_layer_param->set_type("Input");
        InputParameter* input_param = input_layer_param->mutable_input_param();

        input_layer_param->add_top("h_0");
        input_param->add_shape()->CopyFrom(input_shapes[0]);


        // Add layer to transform all timesteps of x to the hidden state dimension.
        //     W_xc_x = W_xc * x + b_c
//        {
//            LayerParameter* x_transform_param = net_param->add_layer();
//            x_transform_param->CopyFrom(biased_hidden_param);
//            x_transform_param->set_name("x_transform");
//            x_transform_param->add_param()->set_name("W_xc");
//            x_transform_param->add_param()->set_name("b_c");
//            x_transform_param->add_bottom("x");
//            x_transform_param->add_top("W_xc_x");
//            x_transform_param->add_propagate_down(true);
//        }

//        if (this->static_input_) {
//            // Add layer to transform x_static to the gate dimension.
//            //     W_xc_x_static = W_xc_static * x_static
//            LayerParameter* x_static_transform_param = net_param->add_layer();
//            x_static_transform_param->CopyFrom(hidden_param);
//            x_static_transform_param->mutable_inner_product_param()->set_axis(1);
//            x_static_transform_param->set_name("W_xc_x_static");
//            x_static_transform_param->add_param()->set_name("W_xc_static");
//            x_static_transform_param->add_bottom("x_static");
//            x_static_transform_param->add_top("W_xc_x_static_preshape");
//            x_static_transform_param->add_propagate_down(true);
//
//            LayerParameter* reshape_param = net_param->add_layer();
//            reshape_param->set_type("Reshape");
//            BlobShape* new_shape =
//                    reshape_param->mutable_reshape_param()->mutable_shape();
//            new_shape->add_dim(1);  // One timestep.
//            // Should infer this->N as the dimension so we can reshape on batch size.
//            new_shape->add_dim(-1);
//            new_shape->add_dim(
//                    x_static_transform_param->inner_product_param().num_output());
//            reshape_param->set_name("W_xc_x_static_reshape");
//            reshape_param->add_bottom("W_xc_x_static_preshape");
//            reshape_param->add_top("W_xc_x_static");
//        }

        LayerParameter output_concat_layer;
        output_concat_layer.set_name("h_concat");
        output_concat_layer.set_type("Concat");
        output_concat_layer.add_top("h");
        output_concat_layer.mutable_concat_param()->set_axis(0);

        LayerParameter sigmoid_param;
        sigmoid_param.set_type("Sigmoid");

        LayerParameter tanh_param;
        tanh_param.set_type("TanH");

        {
            // 3 WXt
            LayerParameter* Wr_param = net_param->add_layer();
            Wr_param->CopyFrom(Wx_hidden_param);
            Wr_param->set_name("wx");
            Wr_param->add_param()->set_name("Wx");
            Wr_param->add_bottom("x");
            Wr_param->add_top("Wx");
            Wr_param->mutable_inner_product_param()->set_axis(2);

            LayerParameter* slice_Wx = net_param->add_layer();
            slice_Wx->set_type("Slice");
            slice_Wx->mutable_slice_param()->set_axis(2);
            slice_Wx->mutable_slice_param()->add_slice_point(1*num_output);
            slice_Wx->mutable_slice_param()->add_slice_point(2*num_output);
            slice_Wx->add_bottom("Wx");
            slice_Wx->add_top("Wr_x");
            slice_Wx->add_top("Wh_x");
            slice_Wx->add_top("Wz_x");

        }

        LayerParameter* wrx_slice_param = net_param->add_layer();// slice x_t from timestamps
        wrx_slice_param->CopyFrom(slice_param_axis0);
        wrx_slice_param->add_bottom("Wr_x");
        wrx_slice_param->set_name("Wr_x_slice");

        LayerParameter* whx_slice_param = net_param->add_layer();// slice x_t from timestamps
        whx_slice_param->CopyFrom(slice_param_axis0);
        whx_slice_param->add_bottom("Wh_x");
        whx_slice_param->set_name("Wh_x_slice");

        LayerParameter* wzx_slice_param = net_param->add_layer();// slice x_t from timestamps
        wzx_slice_param->CopyFrom(slice_param_axis0);
        wzx_slice_param->add_bottom("Wz_x");
        wzx_slice_param->set_name("Wz_x_slice");


        for (int t = 1; t <= this->T_; ++t) {
            string tm1s = format_int(t - 1);//t-1
            string ts = format_int(t);//t

            wrx_slice_param->add_top("Wrx_" + ts);
            whx_slice_param->add_top("Whx_" + ts);
            wzx_slice_param->add_top("Wzx_" + ts);

            {
                // rt=sigmoid(Wr * xt + Ur * ht-1)

                LayerParameter* Ur_param = net_param->add_layer();
                Ur_param->CopyFrom(Wh_hidden_param);
                Ur_param->set_name("ur_h_" + ts);
                Ur_param->add_param()->set_name("U_rh");
                Ur_param->add_bottom("h_" + tm1s);
                Ur_param->add_top("ur_h_" + ts);
                Ur_param->mutable_inner_product_param()->set_axis(2);

                LayerParameter* Sum_param = net_param->add_layer();
                Sum_param->CopyFrom(sum_param);
                Sum_param->set_name("sum_r_" + ts);
                Sum_param->add_bottom("ur_h_" + ts);
                Sum_param->add_bottom("Wrx_" + ts);
                Sum_param->add_top("sum_r_" + ts);

                LayerParameter* Sr_param = net_param->add_layer();
                Sr_param->CopyFrom(sigmoid_param);
                Sr_param->set_name("r_" + ts);
                Sr_param->add_bottom("sum_r_" + ts);
                Sr_param->add_top("r_" + ts);
            }

            {
                // h_hat_t = tanh(W*xt + U*(rt*h_t-1)

                LayerParameter* Prod_param = net_param->add_layer();
                Prod_param->CopyFrom(prod_param);
                Prod_param->set_name("prod_rh_" + ts);
                Prod_param->add_bottom("r_" + ts);
                Prod_param->add_bottom("h_" + tm1s);
                Prod_param->add_top("prod_rh_" + ts);

                LayerParameter* U_param = net_param->add_layer();
                U_param->CopyFrom(Wh_hidden_param);
                U_param->set_name("u_rh_" + ts);
                U_param->add_param()->set_name("U_rh");
                U_param->add_bottom("prod_rh_" + ts);
                U_param->add_top("u_rh_" + ts);
                U_param->mutable_inner_product_param()->set_axis(2);

                LayerParameter* Sum_param = net_param->add_layer();
                Sum_param->CopyFrom(sum_param);
                Sum_param->set_name("sum_wu_" + ts);
                Sum_param->add_bottom("Whx_" + ts);
                Sum_param->add_bottom("u_rh_" + ts);
                Sum_param->add_top("sum_wu_" + ts);

                LayerParameter* Th_param = net_param->add_layer();
                Th_param->CopyFrom(tanh_param);
                Th_param->set_name("h_hat_" + ts);
                Th_param->add_bottom("sum_wu_" + ts);
                Th_param->add_top("h_hat_" + ts);

            }

            {
                // zt=sigmoid(Wz * xt + Uz * ht-1)
                LayerParameter* Ur_param = net_param->add_layer();
                Ur_param->CopyFrom(Wh_hidden_param);
                Ur_param->set_name("uz_h_" + ts);
                Ur_param->add_param()->set_name("U_zh");
                Ur_param->add_bottom("h_" + tm1s);
                Ur_param->add_top("uz_h_" + ts);
                Ur_param->mutable_inner_product_param()->set_axis(2);

                LayerParameter* Sum_param = net_param->add_layer();
                Sum_param->CopyFrom(sum_param);
                Sum_param->set_name("sum_z_" + ts);
                Sum_param->add_bottom("uz_h_" + ts);
                Sum_param->add_bottom("Wzx_" + ts);
                Sum_param->add_top("sum_z_" + ts);

                LayerParameter* Sr_param = net_param->add_layer();
                Sr_param->CopyFrom(sigmoid_param);
                Sr_param->set_name("z_" + ts);
                Sr_param->add_bottom("sum_z_" + ts);
                Sr_param->add_top("z_" + ts);
            }
            {
                // h_t =(1-z_t) .* h_{t-1} + z_t .* h_hat_t
                LayerParameter* scale_shift_param = net_param->add_layer();
                scale_shift_param->CopyFrom(scale_shift);
                scale_shift_param->set_name("ss_" + ts);
                scale_shift_param->add_bottom("z_" + ts);
                scale_shift_param->add_top("ss_" + ts);

                LayerParameter* Prod_param0 = net_param->add_layer();
                Prod_param0->CopyFrom(prod_param);
                Prod_param0->set_name("prod_z0h_" + ts);
                Prod_param0->add_bottom("ss_" + ts);
                Prod_param0->add_bottom("h_" + tm1s);
                Prod_param0->add_top("prod_z0h_" + ts);


                LayerParameter* Prod_param1 = net_param->add_layer();
                Prod_param1->CopyFrom(prod_param);
                Prod_param1->set_name("prod_z1h_" + ts);
                Prod_param1->add_bottom("z_" + ts);
                Prod_param1->add_bottom("h_hat_" + ts);
                Prod_param1->add_top("prod_z1h_" + ts);

                LayerParameter* Sum_param = net_param->add_layer();
                Sum_param->CopyFrom(sum_param);
                Sum_param->set_name("h_" + ts);
                Sum_param->add_bottom("prod_z0h_" + ts);
                Sum_param->add_bottom("prod_z1h_" + ts);
                Sum_param->add_top("h_" + ts);


            }

            output_concat_layer.add_bottom("h_" + ts);
        }  // for (int t = 1; t <= this->T_; ++t)

        net_param->add_layer()->CopyFrom(output_concat_layer);
    }

    INSTANTIATE_CLASS(GRUCFLayer);
    REGISTER_LAYER_CLASS(GRUCF);

}  // namespace caffe