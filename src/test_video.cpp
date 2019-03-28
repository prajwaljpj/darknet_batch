#include <iostream> 
#include "yolo_batch.hpp" 
#include <sys/time.h> 
#include <opencv2/opencv.hpp> 
#include <string> 
#include <thread> 
#include <sstream> 

std::vector<cv::VideoCapture> caps(2); 
std::vector< std::vector<bbox_t> > bbox_vecs(2); 
std::vector< std::string > names; 
std::vector< cv::Mat > mats(2); 

void detect(YOLO_Batch* detector_ptr, int id)
{
    bbox_vecs[id] = detector_ptr->detect_mat_si(mats[id]);     
}

int main(int argc, char** argv) 
{
    std::string cfg = "./cfg/yolov3.cfg";
    std::string weights = "./models/yolov3.weights"; 
    std::string namefile = "./data/coco.names"; 
    // std::string cfg = "./cfg/yolov3-tiny.cfg"; 
    // std::string weights = "./models/yolov3-tiny.weights";
    // std::string namefile = "./data/coco.names";
    // std::string cfg = "./cfg/yolov2-voc.cfg"; 
    // std::string weights = "./models/yolov2-voc.weights"; 
    // std::string namefile = "./data/voc.names"; 

    std::vector<YOLO_Batch*> detectors_ptr(2); 
    for(int i=0;i<2;i++)
    {
        int gpu_id = i; 
        detectors_ptr[i] = new YOLO_Batch(cfg, weights, 1, gpu_id);
        detectors_ptr[i]->m_detectType = 1; 
    }
    names = detectors_ptr[0]->load_object_names(namefile); 
    caps[0].open("/home/camsys/data/data00/cam_00.avi");
    caps[1].open("/home/camsys/data/data00/cam_01.avi"); 
    if(!caps[0].isOpened() || !caps[0].isOpened())
    {
        std::cout << "opencv video failed" << std::endl; 
        caps[0].release(); 
        caps[1].release(); 
        exit(-1); 
    }
    // start thread 
    std::cout << "preparing to start thread" << std::endl; 

    while(true)
    {
        cv::Mat frame; 
        caps[0]>>frame;         
        if(frame.empty()) break; 
        cv::resize(frame, mats[0], cv::Size(512,512));
        caps[1]>>frame; 
        if(frame.empty()) break; 
        cv::resize(frame, mats[1], cv::Size(512,512)); 

        std::thread ths[2]; 

        for(int i=0;i<2;i++) ths[i] = std::thread(detect, detectors_ptr[i], i); 
        for(int i=0;i<2;i++) ths[i].join(); 

        for(int i=0;i<2;i++)
        {
            cv::Mat visual_box; 
            visual_box = detectors_ptr[i]->draw_boxes(mats[i],bbox_vecs[i],names);
            std::stringstream ss; 
            ss << i; 
            cv::imshow(ss.str(), visual_box); 
        }
        cv::waitKey(1); 
    }

    caps[0].release();
    caps[1].release(); 

    cv::destroyAllWindows(); 

    delete detectors_ptr[0];
    delete detectors_ptr[1]; 
    return 0; 
}