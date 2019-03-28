#include <iostream> 
#include "yolo_batch.hpp" 
#include <sys/time.h> 
#include <opencv2/opencv.hpp> 
#include <string> 
#include <thread> 

std::vector<cv::Mat> mats; 
std::vector< std::vector<bbox_t> > bbox_vec_batch_2(4);


void detect(YOLO_Batch* detector_ptr, int id)
{
    bbox_vec_batch_2[id] = detector_ptr->detect_mat_si(mats[id]); 
}
int main(int argc, char**argv) 
{
    std::vector<std::string> img_paths = {
        "/home/camsys/data/calib/18181923/im000.jpg",
        "/home/camsys/data/calib/18181923/im001.jpg",
        "/home/camsys/data/calib/18181923/im002.jpg",
        "/home/camsys/data/calib/18181923/im003.jpg",
        "/home/camsys/data/calib/18181923/im004.jpg",
        "/home/camsys/data/calib/18181923/im005.jpg",
        "/home/camsys/data/calib/18181923/im006.jpg",
        "/home/camsys/data/calib/18181923/im007.jpg"} ; 
    std::string cfg = "./cfg/yolov3.cfg";
    std::string weights = "./models/yolov3.weights"; 
    std::string namefile = "./data/coco.names"; 
    // std::string cfg = "./cfg/yolov2-voc.cfg"; 
    // std::string weights = "./models/yolov2-voc.weights"; 
    // std::string namefile = "./data/voc.names"; 
    // std::string cfg = "./cfg/yolov3-tiny.cfg"; 
    // std::string weights = "./models/yolov3-tiny.weights";
    // std::string namefile = "./data/coco.names";
    // std::string cfg = "./cfg/yolov2-tiny-voc.cfg";
    // std::string weights = "./models/yolov2-tiny-voc.weights"; 
    // std::string namefile = "./data/voc.names";
    // std::string cfg = "./cfg/yolov3-spp.cfg";
    // std::string weights = "./models/yolov3-spp.weights"; 
    // std::string namefile = "./data/coco.names"; 

    // int batch = img_paths.size(); 
    int batch = 2;
    std::cout << "batch size: " << batch << std::endl; 
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


/**********************************************************/
    // test single 
    std::cout << "TEST SINGLE" << std::endl;

    std::vector<YOLO_Batch*> detector_ptrs(4);
    for(int i=0;i<batch;i++)
    {
        int gpu_id = i / 2;
        detector_ptrs[i] = new YOLO_Batch(cfg, weights, 1,gpu_id); 
        detector_ptrs[i]->m_detectType = 1;
    }
    // YOLO_Batch detector_single(cfg, weights, 1, 0); 
    auto names = detector_ptrs[0]->load_object_names(namefile); 


    struct timeval tp2;
    gettimeofday(&tp2, NULL);
    long int ms2 = tp2.tv_sec * 1000 + tp2.tv_usec / 1000;

    // for(int i=0;i<batch; i++)
    // {
    //     std::vector<bbox_t> bbox_vec = detector_ptrs[i]->detect_mat_si(mats[i]);
    //     // bbox_vec_batch_2.push_back(bbox_vec); 
    //     bbox_vec_batch_2[i] = bbox_vec; 
    // }

    std::thread ths[2];
    for(int i=0;i<2;i++) ths[i] = std::thread(detect, detector_ptrs[i], i); 

    for(int i=0;i<2;i++) ths[i].join(); 

    struct timeval tp3;
    gettimeofday(&tp3, NULL);
    long int ms3 = tp3.tv_sec * 1000 + tp3.tv_usec / 1000;
    long int elapse_detection_2 = ms3 - ms2; 
    std::cout << "single: " << elapse_detection_2 << " ms" << std::endl; 

    for(int i=0;i<batch;i++)
    {
        cv::Mat visual_box; 
        visual_box = detector_ptrs[0]->draw_boxes(mats[i], bbox_vec_batch_2[i], names); 
        // visualize 
        cv::imshow("detection", visual_box);
        cv::waitKey(); 
    }

    cv::destroyAllWindows(); 

    return 0; 
}