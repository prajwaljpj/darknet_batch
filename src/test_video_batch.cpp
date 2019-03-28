#include <iostream> 
#include "yolo_batch.hpp" 
#include <sys/time.h> 
#include <opencv2/opencv.hpp> 
#include <string> 
#include <thread> 
#include <sstream> 

std::vector<cv::VideoCapture> caps(2); 
std::vector< std::string > names; 

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

   int batch = 2 ; 
    // std::vector<YOLO_Batch*> detectors_ptr(2); 
    // for(int i=0;i<2;i++)
    // {
    //     int gpu_id = i; 
    //     detectors_ptr[i] = new YOLO_Batch(cfg, weights, 1, gpu_id);
    //     detectors_ptr[i]->m_detectType = 1; 
    // }
    YOLO_Batch detector(cfg,weights,batch,0);
    detector.m_detectType = 1;
    names = detector.load_object_names(namefile); 
    caps[0].open("/home/camsys/data/data00/cam_00.avi");
    caps[1].open("/home/camsys/data/data00/cam_01.avi"); 
    if(!caps[0].isOpened() || !caps[0].isOpened())
    {
        std::cout << "opencv video failed" << std::endl; 
        caps[0].release(); 
        caps[1].release(); 
        exit(-1); 
    }

    while(true)
    {
        std::vector<cv::Mat> mats; 
        for(int i=0;i<batch;i++)
        {
            cv::Mat frame; 
            caps[i]>>frame;         
            if(frame.empty()) break; 
            cv::Mat small; 
            cv::resize(frame, small, cv::Size(500,500));
            mats.push_back(small); 
        }
        if(mats.size() < batch) break; 

        auto bbox_vecs_batch = detector.detect_mat_batch(mats); 

        for(int i=0;i<batch;i++)
        {
            cv::Mat visual_img;
            visual_img = detector.draw_boxes(mats[i], bbox_vecs_batch[i],names); 
            
            std::stringstream ss; 
            ss << "cam " << i ;
            cv::imshow(ss.str(), visual_img); 
            
        }
        int code = cv::waitKey(1); 
        if (code == 27) break; 
    }

    // caps[0].release();
    // caps[1].release(); 


    return 0; 
}