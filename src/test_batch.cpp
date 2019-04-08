#include <iostream> 
#include <sys/time.h> 
#include <string> 
#include <algorithm>

#include "yolo_batch.hpp" 

#include <opencv2/opencv.hpp> 

int main(int argc, char** argv)
{
    std::cout << "running test_batch \n";

    // run_detector_batch(argc, argv);

    std::string cfg = "./cfg/yolov3.cfg";
    std::string weights = "./models/yolov3.weights"; 
    std::string namefile = "./data/coco.names"; 
    
    std::vector<std::string> img_paths = {
        "/home/sadgun/Libraries/darknet_temp/darknet_bkp/data/1-19_197__14.jpg",
        "/home/sadgun/Libraries/darknet_temp/darknet_bkp/data/1-19_197__170.jpg",
        "/home/sadgun/Libraries/darknet_temp/darknet_bkp/data/1-19_197__179.jpg",
        "/home/sadgun/Libraries/darknet_temp/darknet_bkp/data/1-19_197__17.jpg",
        "/home/sadgun/Libraries/darknet_temp/darknet_bkp/data/1-19_197__182.jpg",
        "/home/sadgun/Libraries/darknet_temp/darknet_bkp/data/1-19_197__186.jpg",
        "/home/sadgun/Libraries/darknet_temp/darknet_bkp/data/1-19_197__189.jpg"} ;


    int batch_size = 5;
    std::cout << "[INFO] YOLO BATCH INIT \n";
    YOLO_Batch detector;
    detector.init_batch(cfg, weights, namefile, batch_size, 0);



    // std::cout << "[INFO] TEST 1 \n";
    // detector.run_detector_batch(img_paths);
    // std::cout << "[INFO] TEST 2 \n";
    // std::random_shuffle(img_paths.begin(), img_paths.end());
    // detector.run_detector_batch(img_paths);
    // std::cout << "[INFO] TEST 3 \n";
    // std::random_shuffle(img_paths.begin(), img_paths.end());
    // detector.run_detector_batch(img_paths);


    return 0;
}