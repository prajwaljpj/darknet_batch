#ifndef _YOLO_BATCH_HPP_ 
#define _YOLO_BATCH_HPP_


#ifdef YOLODLL_EXPORTS
#if defined(_MSC_VER)
#define YOLODLL_API __declspec(dllexport)
#else
#define YOLODLL_API __attribute__((visibility("default")))
#endif
#else
#if defined(_MSC_VER)
#define YOLODLL_API __declspec(dllimport)
#else
#define YOLODLL_API
#endif
#endif

//#define DISPLAY_OUT

#ifdef __cplusplus
#include <memory>
#include <vector>
#include <deque>
#include <algorithm>

#ifdef OPENCV
#include <opencv2/opencv.hpp>            // C++
#include "opencv2/highgui/highgui_c.h"    // C
#include "opencv2/imgproc/imgproc_c.h"    // C
#endif    // OPENCV

#endif // __cplusplus

// extern "C"
// {
//     #include "box.h" 
// }
struct bbox_t {
    unsigned int x, y, w, h;    // (x,y) - top-left corner, (w, h) - width & height of bounded box
    float prob;                    // confidence - probability that the object was found correctly
    unsigned int obj_id;        // class of object - from range [0, classes-1]
    unsigned int track_id;        // tracking id for video (0 - untracked, 1 - inf - tracked object)
    unsigned int frames_counter;// counter of frames on which the object was detected
};

typedef  std::vector<bbox_t>  Boxes; 
typedef  std::vector< std::vector<bbox_t> > BoxesVec; 

struct image_t {
    int h;                        // height
    int w;                        // width
    int c;                        // number of chanels (3 - for RGB)
    float *data;                // pointer to the image data
};

typedef  std::vector< std::shared_ptr<image_t> > ImagePtrList;


class YOLODLL_API YOLO_Batch 
{
private: 
    std::shared_ptr<void> m_detector_gpu_ptr; 
    bool m_isInit; 

public: 
    int m_currDeviceId;    // gpu id 
    int m_nms;             // threshold for nms 
    bool m_waitStream; 
    int m_batchsize;       // batch size to be detected
    // 0 -- region 1 -- YOLO 
    int m_detectType;
    std::vector<std::string> names;
    int batch_size = 5;  
  
public: 
    YOLO_Batch(); 
    YOLO_Batch(std::string cfgfile, std::string weightfile, int batch=1, int deviceId=0); 
    ~YOLO_Batch(); 

    int init(std::string cfgfile, std::string weightfile, int batch = 1,int deviceId=0); 
    void release();  // you should call it explicitly at the end of program.

    /* RBC added functions */ 
    int init_batch(std::string cfgfile, std::string weightfile, std::string namesfile, int batch_size = 1,int deviceId=0);
    // void run_detector_batch(std::vector<std::string> img_paths);
    BoxesVec run_detector_batch(std::vector<image_t> img_paths);

    /**
     @brief  detect on single image
     @param  image_t img   : input image with yolo image type
     @param  float thresh  : threshold for report a detection
     @param  bool use_mean : 
     @return std::vector<bbox_t> : bboxes of detected objects 
     */
    std::vector<bbox_t> detect_si(image_t img, float thresh = 0.6, bool use_mean = false);

    /**
     @brief This solution comes from jstumpin at https://github.com/AlexeyAB/darknet/issues/878
     *      But it seems not work for yolov3 detection.
     *      It works for **yolov2** detection 
     */
    std::vector<bbox_t> detect_si_region(image_t img, float thresh=0.6, bool use_mean =false); 
    /**
     @brief  detect objects on resized images 
     @param  image_t img_resized  : resized input image 
     @param  init_w               : original image width 
     @param  init_h               : original image height 
     */
    std::vector<bbox_t> detect_resized_si(image_t img_resized, int init_w, int init_h, float thresh = 0.6, bool use_mean = false); 

    std::vector<bbox_t> detect_mat_si(cv::Mat img, float thresh = 0.6); 

    /**
     @brief  detect batch images, and get bboxes using get_region_boxes 
     */
    BoxesVec detect_batch_region(ImagePtrList imgs, float thresh = 0.6); 

    /**
     @brief  detect batch images with yolo layer functions 
     */
    BoxesVec detect_batch_yolo(ImagePtrList imgs, float thresh=0.6);

    BoxesVec detect_resized_batch(ImagePtrList imgs_resized, int init_w, int init_h, float thresh=0.6); 

    BoxesVec detect_mat_batch(std::vector<cv::Mat>& mats, float thresh=0.6); 

    /**************************compatable with opencv***************/
    std::shared_ptr<image_t> mat_to_image_resize(cv::Mat mat);
    std::shared_ptr<image_t> mat_to_image(cv::Mat img_src);
    image_t ipl_to_image(IplImage* src);
    image_t make_empty_image(int w, int h, int c);
    image_t make_image_custom(int w, int h, int c);


    /***********************utils**********************************/
    int get_net_width()const; 
    int get_net_height()const; 
    void free_image(image_t m); 
    void save_image_t_(image_t& m, const char* name); 

    /***********************post process****************************/
    std::vector<std::string> load_object_names(std::string filename) ;
    cv::Mat draw_boxes(cv::Mat mat_img,	std::vector<bbox_t> result_vec, std::vector<std::string> obj_names); 

}; 


#endif // _YOLO_BATCH_HPP_