#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <queue>
#include <fstream>
#include <thread>
#include <future>
#include <atomic>
#include <mutex>         // std::mutex, std::unique_lock
#include <cmath>


// It makes sense only for video-Camera (not for video-File)
// To use - uncomment the following line. Optical-flow is supported only by OpenCV 3.x - 4.x
//#define TRACK_OPTFLOW
//#define GPU

// To use 3D-stereo camera ZED - uncomment the following line. ZED_SDK should be installed.
//#define ZED_STEREO


#include "yolo_v2_class.hpp"    // imported functions from DLL

#ifdef OPENCV
#ifdef ZED_STEREO
#include <sl_zed/Camera.hpp>
#pragma comment(lib, "sl_core64.lib")
#pragma comment(lib, "sl_input64.lib")
#pragma comment(lib, "sl_zed64.lib")

float getMedian(std::vector<float> &v) {
    size_t n = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + n, v.end());
    return v[n];
}

std::vector<bbox_t> get_3d_coordinates(std::vector<bbox_t> bbox_vect, cv::Mat xyzrgba)
{
    bool valid_measure;
    int i, j;
    const unsigned int R_max_global = 10;

    std::vector<bbox_t> bbox3d_vect;

    for (auto &cur_box : bbox_vect) {

        const unsigned int obj_size = std::min(cur_box.w, cur_box.h);
        const unsigned int R_max = std::min(R_max_global, obj_size / 2);
        int center_i = cur_box.x + cur_box.w * 0.5f, center_j = cur_box.y + cur_box.h * 0.5f;

        std::vector<float> x_vect, y_vect, z_vect;
        for (int R = 0; R < R_max; R++) {
            for (int y = -R; y <= R; y++) {
                for (int x = -R; x <= R; x++) {
                    i = center_i + x;
                    j = center_j + y;
                    sl::float4 out(NAN, NAN, NAN, NAN);
                    if (i >= 0 && i < xyzrgba.cols && j >= 0 && j < xyzrgba.rows) {
                        cv::Vec4f &elem = xyzrgba.at<cv::Vec4f>(j, i);  // x,y,z,w
                        out.x = elem[0];
                        out.y = elem[1];
                        out.z = elem[2];
                        out.w = elem[3];
                    }
                    valid_measure = std::isfinite(out.z);
                    if (valid_measure)
                    {
                        x_vect.push_back(out.x);
                        y_vect.push_back(out.y);
                        z_vect.push_back(out.z);
                    }
                }
            }
        }

        if (x_vect.size() * y_vect.size() * z_vect.size() > 0)
        {
            cur_box.x_3d = getMedian(x_vect);
            cur_box.y_3d = getMedian(y_vect);
            cur_box.z_3d = getMedian(z_vect);
        }
        else {
            cur_box.x_3d = NAN;
            cur_box.y_3d = NAN;
            cur_box.z_3d = NAN;
        }

        bbox3d_vect.emplace_back(cur_box);
    }

    return bbox3d_vect;
}

cv::Mat slMat2cvMat(sl::Mat &input) {
    // Mapping between MAT_TYPE and CV_TYPE
    int cv_type = -1;
    switch (input.getDataType()) {
    case sl::MAT_TYPE_32F_C1:
        cv_type = CV_32FC1;
        break;
    case sl::MAT_TYPE_32F_C2:
        cv_type = CV_32FC2;
        break;
    case sl::MAT_TYPE_32F_C3:
        cv_type = CV_32FC3;
        break;
    case sl::MAT_TYPE_32F_C4:
        cv_type = CV_32FC4;
        break;
    case sl::MAT_TYPE_8U_C1:
        cv_type = CV_8UC1;
        break;
    case sl::MAT_TYPE_8U_C2:
        cv_type = CV_8UC2;
        break;
    case sl::MAT_TYPE_8U_C3:
        cv_type = CV_8UC3;
        break;
    case sl::MAT_TYPE_8U_C4:
        cv_type = CV_8UC4;
        break;
    default:
        break;
    }
    return cv::Mat(input.getHeight(), input.getWidth(), cv_type, input.getPtr<sl::uchar1>(sl::MEM_CPU));
}

cv::Mat zed_capture_rgb(sl::Camera &zed) {
    sl::Mat left;
    zed.retrieveImage(left);
    return slMat2cvMat(left).clone();
}

cv::Mat zed_capture_3d(sl::Camera &zed) {
    sl::Mat cur_cloud;
    zed.retrieveMeasure(cur_cloud, sl::MEASURE_XYZ);
    return slMat2cvMat(cur_cloud).clone();
}

static sl::Camera zed; // ZED-camera

#else   // ZED_STEREO
std::vector<bbox_t> get_3d_coordinates(std::vector<bbox_t> bbox_vect, cv::Mat xyzrgba) {
    return bbox_vect;
}
#endif  // ZED_STEREO


#include <opencv2/opencv.hpp>            // C++
#include <opencv2/core/version.hpp>
#ifndef CV_VERSION_EPOCH
#include <opencv2/videoio/videoio.hpp>
#define OPENCV_VERSION CVAUX_STR(CV_VERSION_MAJOR)"" CVAUX_STR(CV_VERSION_MINOR)"" CVAUX_STR(CV_VERSION_REVISION)
#ifndef USE_CMAKE_LIBS
#pragma comment(lib, "opencv_world" OPENCV_VERSION ".lib")
#ifdef TRACK_OPTFLOW
#pragma comment(lib, "opencv_cudaoptflow" OPENCV_VERSION ".lib")
#pragma comment(lib, "opencv_cudaimgproc" OPENCV_VERSION ".lib")
#pragma comment(lib, "opencv_core" OPENCV_VERSION ".lib")
#pragma comment(lib, "opencv_imgproc" OPENCV_VERSION ".lib")
#pragma comment(lib, "opencv_highgui" OPENCV_VERSION ".lib")
#endif    // TRACK_OPTFLOW
#endif    // USE_CMAKE_LIBS
#else
#define OPENCV_VERSION CVAUX_STR(CV_VERSION_EPOCH)"" CVAUX_STR(CV_VERSION_MAJOR)"" CVAUX_STR(CV_VERSION_MINOR)
#ifndef USE_CMAKE_LIBS
#pragma comment(lib, "opencv_core" OPENCV_VERSION ".lib")
#pragma comment(lib, "opencv_imgproc" OPENCV_VERSION ".lib")
#pragma comment(lib, "opencv_highgui" OPENCV_VERSION ".lib")
#endif    // USE_CMAKE_LIBS
#endif    // CV_VERSION_EPOCH


void draw_boxes(cv::Mat mat_img, std::vector<bbox_t> result_vec, std::vector<std::string> obj_names,
    int current_det_fps = -1, int current_cap_fps = -1)
{
    int const colors[6][3] = { { 1,0,1 },{ 0,0,1 },{ 0,1,1 },{ 0,1,0 },{ 1,1,0 },{ 1,0,0 } };

    for (auto &i : result_vec) {
        cv::Scalar color = obj_id_to_color(i.obj_id);
        cv::rectangle(mat_img, cv::Rect(i.x, i.y, i.w, i.h), color, 2);
        if (obj_names.size() > i.obj_id) {
            std::string obj_name = obj_names[i.obj_id];
            if (i.track_id > 0) obj_name += " - " + std::to_string(i.track_id);
            cv::Size const text_size = getTextSize(obj_name, cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, 2, 0);
            int max_width = (text_size.width > i.w + 2) ? text_size.width : (i.w + 2);
            max_width = std::max(max_width, (int)i.w + 2);
            //max_width = std::max(max_width, 283);
            std::string coords_3d;
            if (!std::isnan(i.z_3d)) {
                std::stringstream ss;
                ss << std::fixed << std::setprecision(2) << "x:" << i.x_3d << "m y:" << i.y_3d << "m z:" << i.z_3d << "m ";
                coords_3d = ss.str();
                cv::Size const text_size_3d = getTextSize(ss.str(), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, 1, 0);
                int const max_width_3d = (text_size_3d.width > i.w + 2) ? text_size_3d.width : (i.w + 2);
                if (max_width_3d > max_width) max_width = max_width_3d;
            }

            cv::rectangle(mat_img, cv::Point2f(std::max((int)i.x - 1, 0), std::max((int)i.y - 35, 0)),
                cv::Point2f(std::min((int)i.x + max_width, mat_img.cols - 1), std::min((int)i.y, mat_img.rows - 1)),
                color, CV_FILLED, 8, 0);
            putText(mat_img, obj_name, cv::Point2f(i.x, i.y - 16), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, cv::Scalar(0, 0, 0), 2);
            if(!coords_3d.empty()) putText(mat_img, coords_3d, cv::Point2f(i.x, i.y-1), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cv::Scalar(0, 0, 0), 1);
        }
    }
    if (current_det_fps >= 0 && current_cap_fps >= 0) {
        std::string fps_str = "FPS detection: " + std::to_string(current_det_fps) + "   FPS capture: " + std::to_string(current_cap_fps);
        putText(mat_img, fps_str, cv::Point2f(10, 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, cv::Scalar(50, 255, 0), 2);
    }
}
#endif    // OPENCV


void show_console_result(std::vector<bbox_t> const result_vec, std::vector<std::string> const obj_names, int frame_id = -1) {
    if (frame_id >= 0) std::cout << " Frame: " << frame_id << std::endl;
    for (auto &i : result_vec) {
        if (obj_names.size() > i.obj_id) std::cout << obj_names[i.obj_id] << " - ";
        std::cout << "obj_id = " << i.obj_id << ",  x = " << i.x << ", y = " << i.y
            << ", w = " << i.w << ", h = " << i.h
            << std::setprecision(3) << ", prob = " << i.prob << std::endl;
    }
}

std::vector<std::string> objects_names_from_file(std::string const filename) {
    std::ifstream file(filename);
    std::vector<std::string> file_lines;
    if (!file.is_open()) return file_lines;
    for(std::string line; getline(file, line);) file_lines.push_back(line);
    std::cout << "object names loaded \n";
    return file_lines;
}

template<typename T>
class send_one_replaceable_object_t {
    const bool sync;
    std::atomic<T *> a_ptr;
public:

    void send(T const& _obj) {
        T *new_ptr = new T;
        *new_ptr = _obj;
        if (sync) {
            while (a_ptr.load()) std::this_thread::sleep_for(std::chrono::milliseconds(3));
        }
        std::unique_ptr<T> old_ptr(a_ptr.exchange(new_ptr));
    }

    T receive() {
        std::unique_ptr<T> ptr;
        do {
            while(!a_ptr.load()) std::this_thread::sleep_for(std::chrono::milliseconds(3));
            ptr.reset(a_ptr.exchange(NULL));
        } while (!ptr);
        T obj = *ptr;
        return obj;
    }

    bool is_object_present() {
        return (a_ptr.load() != NULL);
    }

    send_one_replaceable_object_t(bool _sync) : sync(_sync), a_ptr(NULL)
    {}
};

int main(int argc, char *argv[])
{
    std::string  namefile = "data/coco.names";
    std::string  cfg= "cfg/yolov3.cfg";
    std::string  weights= "yolov3.weights";
    std::string filename;

    if (argc > 3) {    //voc.names yolo-voc.cfg yolo-voc.weights test.mp4
        namefile = argv[1];
        cfg= argv[2];
        weights= argv[3];
        //filename = argv[4];
    }
    //else if (argc > 1) filename = argv[1];

    float const thresh = (argc > 5) ? std::stof(argv[5]) : 0.2;

    std::vector<std::string> img_paths = {
        "/home/camsys/data/calib/18181923/im000.jpg",
        "/home/camsys/data/calib/18181923/im001.jpg",
        "/home/camsys/data/calib/18181923/im002.jpg",
        "/home/camsys/data/calib/18181923/im003.jpg",
        "/home/camsys/data/calib/18181923/im004.jpg",
        "/home/camsys/data/calib/18181923/im005.jpg",
        "/home/camsys/data/calib/18181923/im006.jpg",
        "/home/camsys/data/calib/18181923/im007.jpg"} ;

    // int batch = img_paths.size();
    int batch = 4;
    std::cout << "batch size: " << batch << std::endl;
    std::vector<cv::Mat> mats;
    for(int i=0;i<batch;i++)
    {
        cv::Mat img = cv::imread(img_paths[i]);

        if(img.empty())
        {
            std::cout << "img empty" << std::endl;
            exit(-1);
        }
        cv::Mat small;
        cv::resize(img, small,cv::Size(416,416));
        mats.push_back(small);
    }
    YOLO_Batch detector(cfg, weights, batch, 0);
    detector.m_detectType = 1;
    auto names = detector.load_object_names(namefile);

    // test bathch
    std::cout << "debug: detector initialization success" << std::endl;

    struct timeval tp;
    gettimeofday(&tp, NULL);
    long int ms = tp.tv_sec * 1000 + tp.tv_usec / 1000;

    BoxesVec bbox_vec_batch = detector.detect_mat_batch(mats);

    struct timeval tp1;
    gettimeofday(&tp1, NULL);
    long int ms1 = tp1.tv_sec * 1000 + tp1.tv_usec / 1000;
    long int elapse_detection = ms1 - ms;

    std::cout << "batch detection time:" << elapse_detection << " ms" << std::endl;
    for(int i=0;i<batch;i++)
    {
        cv::Mat visual_box;


        visual_box = detector.draw_boxes(mats[i], bbox_vec_batch[i], names);
        // visualize
        cv::imshow("detection", visual_box);
        cv::waitKey();
    }

/**********************************************************/
    // test single
    // std::cout << "TEST SINGLE" << std::endl;

    // YOLO_Batch detector_single(cfg, weights, 1, 1);
    // detector_single.m_detectType = 1;

    // struct timeval tp2;
    // gettimeofday(&tp2, NULL);
    // long int ms2 = tp2.tv_sec * 1000 + tp2.tv_usec / 1000;

    // std::vector< std::vector<bbox_t> > bbox_vec_batch_2;
    // for(int i=0;i<batch; i++)
    // {
    //     std::vector<bbox_t> bbox_vec = detector_single.detect_mat_si(mats[i]);
    //     bbox_vec_batch_2.push_back(bbox_vec);
    // }

    // struct timeval tp3;
    // gettimeofday(&tp3, NULL);
    // long int ms3 = tp3.tv_sec * 1000 + tp3.tv_usec / 1000;
    // long int elapse_detection_2 = ms3 - ms2;
    // std::cout << "single: " << elapse_detection_2 << " ms" << std::endl;

    // for(int i=0;i<batch;i++)
    // {
    //     cv::Mat visual_box;
    //     visual_box = detector.draw_boxes(mats[i], bbox_vec_batch_2[i], names);
    //     // visualize
    //     cv::imshow("detection", visual_box);
    //     cv::waitKey();
    // }


    return 0;
}
