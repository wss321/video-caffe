#include <fcntl.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/core/version.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#if CV_MAJOR_VERSION == 3
#include <opencv2/videoio/videoio.hpp>
#endif
#endif  // USE_OPENCV
#include <stdint.h>

#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

// Check if a given path is a regular file or a path
void check_path(const std::string& path, bool* is_file, bool* is_dir) {
  struct stat path_stat;
  stat(path.c_str(), &path_stat);
  *is_file = S_ISREG(path_stat.st_mode);
  *is_dir  = S_ISDIR(path_stat.st_mode);
}

const int kProtoReadBytesLimit = INT_MAX;  // Max size of 2 GB minus 1 byte.

namespace caffe {

using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::ZeroCopyOutputStream;
using google::protobuf::io::CodedOutputStream;
using google::protobuf::Message;

bool ReadProtoFromTextFile(const char* filename, Message* proto) {
  int fd = open(filename, O_RDONLY);
  CHECK_NE(fd, -1) << "File not found: " << filename;
  FileInputStream* input = new FileInputStream(fd);
  bool success = google::protobuf::TextFormat::Parse(input, proto);
  delete input;
  close(fd);
  return success;
}

void WriteProtoToTextFile(const Message& proto, const char* filename) {
  int fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0644);
  FileOutputStream* output = new FileOutputStream(fd);
  CHECK(google::protobuf::TextFormat::Print(proto, output));
  delete output;
  close(fd);
}

bool ReadProtoFromBinaryFile(const char* filename, Message* proto) {
  int fd = open(filename, O_RDONLY);
  CHECK_NE(fd, -1) << "File not found: " << filename;
  ZeroCopyInputStream* raw_input = new FileInputStream(fd);
  CodedInputStream* coded_input = new CodedInputStream(raw_input);
  coded_input->SetTotalBytesLimit(kProtoReadBytesLimit, 536870912);

  bool success = proto->ParseFromCodedStream(coded_input);

  delete coded_input;
  delete raw_input;
  close(fd);
  return success;
}

void WriteProtoToBinaryFile(const Message& proto, const char* filename) {
  fstream output(filename, ios::out | ios::trunc | ios::binary);
  CHECK(proto.SerializeToOstream(&output));
}

#ifdef USE_OPENCV
cv::Mat ReadImageToCVMat(const string& filename,
    const int height, const int width, const bool is_color) {
  cv::Mat cv_img;
  int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :
    CV_LOAD_IMAGE_GRAYSCALE);
  cv::Mat cv_img_origin = cv::imread(filename, cv_read_flag);
  if (!cv_img_origin.data) {
    LOG(ERROR) << "Could not open or find file " << filename;
    return cv_img_origin;
  }
  if (height > 0 && width > 0) {
    cv::resize(cv_img_origin, cv_img, cv::Size(width, height));
  } else {
    cv_img = cv_img_origin;
  }
  return cv_img;
}

cv::Mat ReadImageToCVMat(const string& filename,
    const int height, const int width) {
  return ReadImageToCVMat(filename, height, width, true);
}

cv::Mat ReadImageToCVMat(const string& filename,
    const bool is_color) {
  return ReadImageToCVMat(filename, 0, 0, is_color);
}

cv::Mat ReadImageToCVMat(const string& filename) {
  return ReadImageToCVMat(filename, 0, 0, true);
}

// Do the file extension and encoding match?
static bool matchExt(const std::string & fn,
                     std::string en) {
  size_t p = fn.rfind('.');
  std::string ext = p != fn.npos ? fn.substr(p+1) : fn;
  std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
  std::transform(en.begin(), en.end(), en.begin(), ::tolower);
  if ( ext == en )
    return true;
  if ( en == "jpg" && ext == "jpeg" )
    return true;
  return false;
}

bool ReadImageToDatum(const string& filename, const int label,
    const int height, const int width, const bool is_color,
    const std::string & encoding, Datum* datum) {
  cv::Mat cv_img = ReadImageToCVMat(filename, height, width, is_color);
  if (cv_img.data) {
    if (encoding.size()) {
      if ( (cv_img.channels() == 3) == is_color && !height && !width &&
          matchExt(filename, encoding) )
        return ReadFileToDatum(filename, label, datum);
      std::vector<uchar> buf;
      cv::imencode("."+encoding, cv_img, buf);
      datum->set_data(std::string(reinterpret_cast<char*>(&buf[0]),
                      buf.size()));
      datum->set_label(label);
      datum->set_encoded(true);
      return true;
    }
    CVMatToDatum(cv_img, datum);
    datum->set_label(label);
    return true;
  } else {
    return false;
  }
}
//template <typename Dtype>
bool ReadCSVToCVMat(const string& filename, cv::Mat& mat, const char delim){
    std::ifstream file(filename.c_str(), std::ifstream::in);

    if(! file)
    {
        LOG(ERROR) << "Could not open or find file " << filename;
        return false;
    }
    string line;
    int num_line = 0;
    int num_col = 1;
    if (getline(file, line)) {
        num_line++;
        for (int i = 0; i < line.size(); ++i) {
            if (line[i] == delim) num_col++;
        }
    }

    while (getline(file, line)) {
        num_line++;
    };
    file.clear();
    file.seekg(0, file.beg);
    string data;
    vector<float> out(num_line * num_col, 0);
    int index = 0;
//    if (typeid(Dtype(0))==typeid(double(0)))
//        mat = cv::Mat::zeros(num_line,  num_col, CV_64FC1);
//    else
    mat = cv::Mat::zeros(num_line,  num_col, CV_32FC1);
    int n_r = 0, n_c = 0;
    while (getline(file, line)) {
        std::istringstream sin(line);
        n_c = 0;
        while (getline(sin, data, ',')) {
            out[index] = stof(data);
            index++;
            mat.at<float>(n_r,n_c) = stof(data);
            n_c++;
        }
        n_r ++;

    };

    file.close();
    return true;

}
bool ReadVideoToCVMat(const string& path,
                      const int start_frame, const int length, const int height, const int width, const int interval,
                      const bool is_color, std::vector<cv::Mat>* cv_imgs){
    // Check if path is a directory that holds extracted images from a video,
    // or a regular video file.
    bool is_video_file, is_path;
    check_path(path, &is_video_file, &is_path);
    if (!is_video_file && !is_path) {
        LOG(ERROR) << "Could not open or find file " << path;
        return false;
    }

    cv::Mat cv_img, cv_img_origin;

    // In case of a video file
    if (is_video_file) {
        cv::VideoCapture cap;
        cap.open(path);

        if (!cap.isOpened()) {
            LOG(ERROR) << "Cannot open a video file=" << path;
            return false;
        }

        int num_frames = cap.get(CV_CAP_PROP_FRAME_COUNT) + 1;
        int end_frame = start_frame + (length- 1) * (interval + 1);
        if (num_frames < end_frame) {
            LOG(ERROR) << "not enough frames; num_frames=" << num_frames <<
                       ", start_frame=" << start_frame <<
                       ", length=" << length<<
                       ", interval=" << interval<<
                       ", end_frame=" << end_frame;
            return false;
        }

        // CV_CAP_PROP_POS_FRAMES is 0-based whereas start_frame is 1-based

        for (int i = start_frame; i <= end_frame; i+=(interval+1)) {
            cap.set(CV_CAP_PROP_POS_FRAMES, i - 2);
            cap.read(cv_img_origin);
            if (!cv_img_origin.data) {
                LOG(INFO) << "Could not read frame=" << i <<
                          " from a video file=" << path <<
                          ", where num of frames=" << num_frames <<
                          ". Use previous frame.";
                cv_imgs->push_back(cv_img.clone());
                cv_img_origin.release();
                continue;
            }

            // Force color
            if (is_color && cv_img_origin.channels() == 1) {
                cv::cvtColor(cv_img_origin, cv_img_origin, CV_GRAY2BGR);
                // Force grayscale
            } else if (!is_color && cv_img_origin.channels() == 3) {
                cv::cvtColor(cv_img_origin, cv_img_origin, CV_BGR2GRAY);
            }

            if (height > 0 && width > 0) {
                cv::resize(cv_img_origin, cv_img, cv::Size(width, height));
            } else {
                cv_img = cv_img_origin;
            }
            cv_imgs->push_back(cv_img.clone());
            cv_img_origin.release();
        }
        cap.release();

        // In case of a directory with extracted frames within
    } else {
        int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :
                            CV_LOAD_IMAGE_GRAYSCALE);

        // Filename: 6-digit zero-padded
        int end_frame = start_frame + (length- 1) * (interval + 1);
        char image_filename[256];

        for (int i = start_frame; i <= end_frame;  i+=(interval+1)) {
            snprintf(image_filename, sizeof(image_filename), "%s/image_%04d.jpg",
                     path.c_str(), i);
            cv_img_origin = cv::imread(image_filename, cv_read_flag);
            if (!cv_img_origin.data) {
                LOG(ERROR) << "Could not read frame=" << i <<
                           " from an image file=" << image_filename;
                cv_imgs->clear();
                return false;
            }
            if (height > 0 && width > 0) {
                cv::resize(cv_img_origin, cv_img, cv::Size(width, height));
            } else {
                cv_img = cv_img_origin;
            }
            cv_imgs->push_back(cv_img.clone());
            cv_img_origin.release();
        }
    }
    return true;
};
bool ReadVideoToCVMat(const string& path,
    const int start_frame, const int length, const int height, const int width,
    const bool is_color, std::vector<cv::Mat>* cv_imgs) {
    return ReadVideoToCVMat(path, start_frame,length,height,width,0,is_color,cv_imgs);
}

#endif  // USE_OPENCV

bool ReadFileToDatum(const string& filename, const int label,
    Datum* datum) {
  std::streampos size;

  fstream file(filename.c_str(), ios::in|ios::binary|ios::ate);
  if (file.is_open()) {
    size = file.tellg();
    std::string buffer(size, ' ');
    file.seekg(0, ios::beg);
    file.read(&buffer[0], size);
    file.close();
    datum->set_data(buffer);
    datum->set_label(label);
    datum->set_encoded(true);
    return true;
  } else {
    return false;
  }
}

#ifdef USE_OPENCV
cv::Mat DecodeDatumToCVMatNative(const Datum& datum) {
  cv::Mat cv_img;
  CHECK(datum.encoded()) << "Datum not encoded";
  const string& data = datum.data();
  std::vector<char> vec_data(data.c_str(), data.c_str() + data.size());
  cv_img = cv::imdecode(vec_data, -1);
  if (!cv_img.data) {
    LOG(ERROR) << "Could not decode datum ";
  }
  return cv_img;
}
cv::Mat DecodeDatumToCVMat(const Datum& datum, bool is_color) {
  cv::Mat cv_img;
  CHECK(datum.encoded()) << "Datum not encoded";
  const string& data = datum.data();
  std::vector<char> vec_data(data.c_str(), data.c_str() + data.size());
  int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :
    CV_LOAD_IMAGE_GRAYSCALE);
  cv_img = cv::imdecode(vec_data, cv_read_flag);
  if (!cv_img.data) {
    LOG(ERROR) << "Could not decode datum ";
  }
  return cv_img;
}
void DecodeDatumToBlob(const Datum& datum, float* out_data) {
    const string& data = datum.data();
    std::vector<char> vec_data(data.c_str(), data.c_str() + data.size());
    const int datum_channels = datum.channels();
    const int datum_height = datum.height();
    const int datum_width = datum.width();
    const bool has_uint8 = data.size() > 0;

    float datum_element;
    int top_index=0, data_index;
    for (int c = 0; c < datum_channels; ++c) {
        for (int h = 0; h < datum_height; ++h) {
            for (int w = 0; w < datum_width; ++w) {
                data_index = (c * datum_height + h) * datum_width + w;

                if (has_uint8) {
                    datum_element =
                            static_cast<float>(static_cast<uint8_t>(data[data_index]));
                } else {
                    datum_element = datum.float_data(data_index);
                }
                out_data[top_index] = datum_element;
                top_index++;
            }
        }
    }
}
void DecodeDatumToBlob(const Datum& datum, double * out_data) {
    CHECK(datum.encoded()) << "Datum not encoded";
    const string& data = datum.data();
    std::vector<char> vec_data(data.c_str(), data.c_str() + data.size());
    const int datum_channels = datum.channels();
    const int datum_height = datum.height();
    const int datum_width = datum.width();
    const bool has_uint8 = data.size() > 0;

    double datum_element;
    int top_index=0, data_index;
    for (int c = 0; c < datum_channels; ++c) {
        for (int h = 0; h < datum_height; ++h) {
            for (int w = 0; w < datum_width; ++w) {
                data_index = (c * datum_height + h) * datum_width + w;
                if (has_uint8) {
                    datum_element =
                            static_cast<double >(static_cast<uint8_t>(data[data_index]));
                } else {
                    datum_element = (double)datum.float_data(data_index);
                }
                out_data[top_index] = datum_element;
                top_index++;
            }
        }
    }
}

// If Datum is encoded will decoded using DecodeDatumToCVMat and CVMatToDatum
// If Datum is not encoded will do nothing
bool DecodeDatumNative(Datum* datum) {
  if (datum->encoded()) {
    cv::Mat cv_img = DecodeDatumToCVMatNative((*datum));
    CVMatToDatum(cv_img, datum);
    return true;
  } else {
    return false;
  }
}
bool DecodeDatum(Datum* datum, bool is_color) {
  if (datum->encoded()) {
    cv::Mat cv_img = DecodeDatumToCVMat((*datum), is_color);
    CVMatToDatum(cv_img, datum);
    return true;
  } else {
    return false;
  }
}

void CVMatToDatum(const cv::Mat& cv_img, Datum* datum) {
  CHECK(cv_img.depth() == CV_8U) << "Image data type must be unsigned byte";
  datum->set_channels(cv_img.channels());
  datum->set_height(cv_img.rows);
  datum->set_width(cv_img.cols);
  datum->clear_data();
  datum->clear_float_data();
  datum->set_encoded(false);
  int datum_channels = datum->channels();
  int datum_height = datum->height();
  int datum_width = datum->width();
  int datum_size = datum_channels * datum_height * datum_width;
  std::string buffer(datum_size, ' ');
  for (int h = 0; h < datum_height; ++h) {
    const uchar* ptr = cv_img.ptr<uchar>(h);
    int img_index = 0;
    for (int w = 0; w < datum_width; ++w) {
      for (int c = 0; c < datum_channels; ++c) {
        int datum_index = (c * datum_height + h) * datum_width + w;
        buffer[datum_index] = static_cast<char>(ptr[img_index++]);
      }
    }
  }
  datum->set_data(buffer);
}
#endif  // USE_OPENCV
}  // namespace caffe
