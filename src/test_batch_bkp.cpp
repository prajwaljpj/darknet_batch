#include <iostream> 
#include "yolo_batch.hpp" 
#include <sys/time.h> 
#include <opencv2/opencv.hpp> 
#include <string> 

int main(int argc, char**argv) 
// void test_batch_yolo(int argc, char**argv)
{

    std::cout << "Hello \n";

#if 1

    std::vector<std::string> img_paths = {
        "/home/sadgun/Libraries/darknet_temp/darknet_bkp/data/1-19_197__14.jpg",
        "/home/sadgun/Libraries/darknet_temp/darknet_bkp/data/1-19_197__170.jpg",
        "/home/sadgun/Libraries/darknet_temp/darknet_bkp/data/1-19_197__179.jpg",
        "/home/sadgun/Libraries/darknet_temp/darknet_bkp/data/1-19_197__17.jpg",
        "/home/sadgun/Libraries/darknet_temp/darknet_bkp/data/1-19_197__182.jpg",
        "/home/sadgun/Libraries/darknet_temp/darknet_bkp/data/1-19_197__186.jpg",
        "/home/sadgun/Libraries/darknet_temp/darknet_bkp/data/1-19_197__189.jpg"} ; 
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
    // std::string cfg = "./cfg/yolov2-tiny.cfg";
    // std::string weights = "./models/yolov2-tiny.weights";
    // std::string namefile = "./data/coco.names"; 
    // std::string cfg = "./cfg/yolov3-spp.cfg"; 
    // std::string weights = "./models/yolov3-spp.weights"; 
    // std::string namefile = "./data/coco.names"; 
    // std::string cfg = "./cfg/yolov3-608.cfg";
    // std::string weights = "./models/yolov3-608.weights"; 
    // std::string namefile = "./data/coco.names"; 
    // std::string cfg = "./cfg/yolo9000.cfg";
    // std::string weights = "./models/yolo9000.weights"; 
    // std::string namefile = "./cfg/9k.names"; 


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

#endif
    return 0; 
    // return;
}