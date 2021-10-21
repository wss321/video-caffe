#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>


#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_multi_data_layer.hpp"
#include "caffe/layers/video_gyro_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

    template <typename Dtype>
    VideoGyroDataLayer<Dtype>::~VideoGyroDataLayer<Dtype>() {
        this->StopInternalThread();
    }

    template <typename Dtype>
    void VideoGyroDataLayer<Dtype>::MultiDataLayerSetUp(const vector<Blob<Dtype>*>&
    bottom, const vector<Blob<Dtype>*>& top) {
        const int new_length = this->layer_param_.video_gyro_data_param().new_length();
        const int new_height = this->layer_param_.video_gyro_data_param().new_height();
        const int new_width  = this->layer_param_.video_gyro_data_param().new_width();
        const bool is_color  = this->layer_param_.video_gyro_data_param().is_color();
        string root_folder = this->layer_param_.video_gyro_data_param().root_folder();
        const int num_label  = this->layer_param_.video_gyro_data_param().num_label();

        CHECK((new_height == 0 && new_width == 0) ||
              (new_height > 0 && new_width > 0)) << "Current implementation requires "
                                                    "new_height and new_width to be set at the same time.";
        // Read the file with filenames and labels
        const string& source = this->layer_param_.video_gyro_data_param().source();
        LOG(INFO) << "Opening file " << source;
        std::ifstream infile(source.c_str());
        string videofilename, gyrofilename, labelfilename;
        int frame_num;
        while (infile >> videofilename >>gyrofilename>>labelfilename>> frame_num) {
            quadruple video_gyro_label;
            video_gyro_label.first = videofilename;
            video_gyro_label.second = gyrofilename;
            video_gyro_label.third = labelfilename;
            video_gyro_label.fourth = frame_num;
            lines_.push_back(video_gyro_label);
        }

        CHECK(!lines_.empty()) << "File is empty";

        if (this->layer_param_.video_gyro_data_param().shuffle()) {
            // randomly shuffle data
            LOG(INFO) << "Shuffling data";
            const unsigned int prefetch_rng_seed = caffe_rng_rand();
            prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
            ShuffleVideos();
        } else {
            if (this->phase_ == TRAIN && Caffe::solver_rank() > 0 &&
                this->layer_param_.video_gyro_data_param().rand_skip() == 0) {
                LOG(WARNING) << "Shuffling or skipping recommended for multi-GPU";
            }
        }
        LOG(INFO) << "A total of " << lines_.size() << " video chunks.";

        lines_id_ = 0;
        // Check if we would need to randomly skip a few data points
        if (this->layer_param_.video_gyro_data_param().rand_skip()) {
            unsigned int skip = caffe_rng_rand() %
                                this->layer_param_.video_gyro_data_param().rand_skip();
            LOG(INFO) << "Skipping first " << skip << " data points.";
            CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
            lines_id_ = skip;
        }
        // Read a video clip, and use it to initialize the top blob.
        std::vector<cv::Mat> cv_imgs;
        bool read_video_result = ReadVideoToCVMat(root_folder +
                                                  lines_[lines_id_].first,
                                                  lines_[lines_id_].fourth,
                                                  new_length, new_height, new_width,
                                                  is_color,
                                                  &cv_imgs);
        CHECK(read_video_result) << "Could not load " << lines_[lines_id_].first <<
                                 " at frame " << lines_[lines_id_].fourth << ".";
        CHECK_EQ(cv_imgs.size(), new_length) << "Could not load " <<
                                             lines_[lines_id_].first <<
                                             " at frame " <<
                                             lines_[lines_id_].fourth <<
                                             " correctly.";


        // Read a gyro file, and use it to initialize the top blob.
        cv::Mat gyro;
        bool read_gyro_result = ReadCSVToCVMat(root_folder +
                                               lines_[lines_id_].second,
                                               gyro,',');
        CHECK(read_gyro_result) << "Could not load " << lines_[lines_id_].second << ".";

        // Use data_transformer to infer the expected blob shape from a cv_image.
        const bool is_video = true;
        vector<int> video_shape = this->data_transformer_->InferBlobShape(cv_imgs,
                                                                          is_video);
        this->transformed_data_.Reshape(video_shape);
        // Reshape prefetch_data and top[0] according to the batch_size.
        const int batch_size = this->layer_param_.video_gyro_data_param().batch_size();
        CHECK_GT(batch_size, 0) << "Positive batch size required";
        video_shape[0] = batch_size;

        top[0]->Reshape(video_shape);
        LOG(INFO) << "video data size: " << top[0]->shape(0) << ","
                  << top[0]->shape(1) << "," << top[0]->shape(2) << ","
                  << top[0]->shape(3) << "," << top[0]->shape(4);

        vector<int> gyro_shape{batch_size, gyro.rows,  gyro.cols};
        top[1]->Reshape(gyro_shape);
        LOG(INFO) << "gyro data size: " << top[1]->shape(0) << ","
                  << top[1]->shape(1) << "," << top[1]->shape(2);


        vector<int> label_shape{batch_size, num_label};
        top[2]->Reshape(label_shape);
        LOG(INFO) << "label size: " << top[2]->shape(0) << ","
                  << top[2]->shape(1);

        for (int i = 0; i < this->prefetch_.size(); ++i) {
            this->prefetch_[i]->data_.resize(3);
//            this->prefetch_[i]->data_[0].reset(new Blob<Dtype>());
            this->prefetch_[i]->data_[0] = new Blob<Dtype>();
            this->prefetch_[i]->data_[0]->Reshape(video_shape);

//            this->prefetch_[i]->data_[1].reset(new Blob<Dtype>());
            this->prefetch_[i]->data_[1] = new Blob<Dtype>();
            this->prefetch_[i]->data_[1]->Reshape(gyro_shape);

//            this->prefetch_[i]->data_[2].reset(new Blob<Dtype>());
            this->prefetch_[i]->data_[2] = new Blob<Dtype>();
            this->prefetch_[i]->data_[2]->Reshape(label_shape);
        }
    }

    template <typename Dtype>
    void VideoGyroDataLayer<Dtype>::ShuffleVideos() {
        caffe::rng_t* prefetch_rng =
                static_cast<caffe::rng_t*>(prefetch_rng_->generator());
        shuffle(lines_.begin(), lines_.end(), prefetch_rng);
    }

// This function is called on prefetch thread
    template <typename Dtype>
    void VideoGyroDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
        CPUTimer batch_timer;
        batch_timer.Start();
        double read_time = 0;
        double trans_time = 0;
        CPUTimer timer;
        CHECK(batch->data_[0]->count());
        CHECK(this->transformed_data_.count());
        VideoGyroDataParameter video_gyro_data_param = this->layer_param_.video_gyro_data_param();
        const int batch_size = video_gyro_data_param.batch_size();
        const int new_length = video_gyro_data_param.new_length();
        const int new_height = video_gyro_data_param.new_height();
        const int new_width = video_gyro_data_param.new_width();
        const bool is_color = video_gyro_data_param.is_color();
        const string root_folder = video_gyro_data_param.root_folder();
        const int interval  = this->layer_param_.video_gyro_data_param().interval();

        Dtype* prefetch_video_data = batch->data_[0]->mutable_cpu_data();
        Dtype* prefetch_gyro_data = batch->data_[1]->mutable_cpu_data();
        Dtype* prefetch_label = batch->data_[2]->mutable_cpu_data();

        // datum scales
        const int lines_size = lines_.size();
        for (int item_id = 0; item_id < batch_size; ++item_id) {
            // get a blob
            timer.Start();
            CHECK_GT(lines_size, lines_id_);
            std::vector<cv::Mat> cv_imgs;
            bool read_video_result = ReadVideoToCVMat(root_folder +
                                                      lines_[lines_id_].first,
                                                      lines_[lines_id_].fourth,
                                                      new_length, new_height,
                                                      new_width, interval,
                                                      is_color, &cv_imgs);
//            LOG(INFO) << cv_imgs[0];
            CHECK(read_video_result) << "Could not load " << lines_[lines_id_].first <<
                                     " at frame " << lines_[lines_id_].second << ".";
            CHECK_EQ(cv_imgs.size(), new_length) << "Could not load " <<
                                                 lines_[lines_id_].first <<
                                                 " at frame " <<
                                                 lines_[lines_id_].second <<
                                                 " correctly.";
            cv::Mat gyro;
            bool read_gyro_result = ReadCSVToCVMat(root_folder +
                                                   lines_[lines_id_].second,
                                                   gyro,',');
            CHECK(read_gyro_result) << "Could not load " << lines_[lines_id_].second << ".";
//            LOG(INFO) << gyro;


//        // Read a label file, and use it to initialize the top blob.
        cv::Mat label;
        bool read_label_result = ReadCSVToCVMat(root_folder +
                                               lines_[lines_id_].third,
                                               label,',');
        CHECK(read_label_result) << "Could not load " << lines_[lines_id_].third << ".";
            CHECK_EQ(batch->data_[2]->count()/batch_size, label.total())<<"label size mismatch.";
//            LOG(INFO) << label;

            read_time += timer.MicroSeconds();
            timer.Start();
            // Apply transformations (mirror, crop...) to the image
            int offset_v = batch->data_[0]->offset(item_id);
            int offset_g = batch->data_[1]->offset(item_id);
            int offset_l = batch->data_[2]->offset(item_id);
            this->transformed_data_.set_cpu_data(prefetch_video_data + offset_v);
            const bool is_video = true;
            this->data_transformer_->Transform(cv_imgs, &(this->transformed_data_),
                                               is_video);
            trans_time += timer.MicroSeconds();

            int n_c = gyro.cols;

//            LOG(INFO)<<gyro;
            for (int i = 0; i < gyro.rows; ++i) {
                for (int j = 0; j < n_c; ++j) {
                    prefetch_gyro_data[offset_g + i * n_c + j] = gyro.at<Dtype>(i, j);
//                    std::cout<<gyro.at<Dtype>(i, j)<<" ";
                }
            }
            n_c = label.cols;
            for (int i = 0; i < label.rows; ++i) {
                for (int j = 0; j < n_c; ++j) {
                    prefetch_label[offset_l + i * n_c + j] = label.at<Dtype>(i, j);
                }
            }

            // go to the next iter
            lines_id_++;
            if (lines_id_ >= lines_size) {
                // We have reached the end. Restart from the first.
                DLOG(INFO) << "Restarting data prefetching from start.";
                lines_id_ = 0;
                if (this->layer_param_.video_gyro_data_param().shuffle()) {
                    ShuffleVideos();
                }
            }
        }
        batch_timer.Stop();
        DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
        DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
        DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
    }

    INSTANTIATE_CLASS(VideoGyroDataLayer);
    REGISTER_LAYER_CLASS(VideoGyroData);

}  // namespace caffe
#endif  // USE_OPENCV
