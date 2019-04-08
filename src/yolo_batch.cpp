#include "yolo_batch.hpp"
#include "network.h" 

extern "C" {
#include "detection_layer.h" 
#include "region_layer.h" 
#include "cost_layer.h" 
#include "utils.h" 
#include "parser.h" 
#include "box.h" 
#include "image.h" 
#include "demo.h" 
#include "option_list.h" 
#include "stb_image.h" 
}
#include <sys/time.h> 
#include <vector> 
#include <iostream> 
#include <algorithm> 
#include <iomanip> 
#include <exception>

#define BATCH 3

/*************************YOLO Batch***************************/
YOLO_Batch::YOLO_Batch()
{
    // 
}

YOLO_Batch::YOLO_Batch(std::string cfgfile, std::string weightfile, int batch, int deviceId)
{
    init(cfgfile, weightfile, batch, deviceId); 
}

YOLO_Batch::~YOLO_Batch()
{
    release(); 
}

struct detector_gpu_t
{
    network net; 
    // image images[BATCH]; 
    // float* avg; 
    // float* predictions[BATCH]; 
    // int demo_index; // used for temporal meanning 
    // unsigned int* track_id; 
}; 


int YOLO_Batch::init(std::string cfgfile,
                     std::string weightfile,
                     int batch, 
                     int deviceId)
{
    m_currDeviceId = deviceId; 

    if(m_currDeviceId >=0 )
        cuda_set_device(m_currDeviceId); 

    // init shared pointer 
    m_detector_gpu_ptr = std::make_shared<detector_gpu_t>(); 
    detector_gpu_t &detector_gpu = *static_cast<detector_gpu_t*>(m_detector_gpu_ptr.get()); 
    
    network &net = detector_gpu.net; 
    net.gpu_index = m_currDeviceId; 
    std::cout << "Used GPU: " << m_currDeviceId << std::endl; 
    char* c_cfg = const_cast<char*>(cfgfile.data()); 
    char* c_weight = const_cast<char*>(weightfile.data()); 

    net = parse_network_cfg_custom(c_cfg,batch, 0);  
    if(c_weight)
    {
        load_weights(&net, c_weight); 
    }
    set_batch_network(&net, batch);  // set batch  
    net.gpu_index = m_currDeviceId; 
    fuse_conv_batchnorm(net); 

    // layer l = net.layers[net.n-1]; //get last layer
    // int j;

    // detector_gpu.avg = (float*)calloc(l.outputs, sizeof(float)); 
    // if(detector_gpu.avg == NULL)std::cout << "detector_gpu.avg calloc failed!" << std::endl; 
    // for(j=0;j<BATCH;++j) detector_gpu.predictions[j] = (float*)calloc(l.outputs, sizeof(float)); 
    // for(j=0;j<BATCH;++j) detector_gpu.images[j] = make_image(1,1,3); 

    // init other variables 
    m_nms = 0.4; 
    m_waitStream = false; 
    m_batchsize = batch; 
    m_detectType = 0;   // default region type
    return 0;
}

void YOLO_Batch::release()
{

}

std::vector<bbox_t> YOLO_Batch::detect_si(image_t img, float thresh, bool use_mean)
{
    detector_gpu_t &detector_gpu = *static_cast<detector_gpu_t*>(m_detector_gpu_ptr.get());
    network &net = detector_gpu.net; 
    int old_gpu_index = cuda_get_device();  // save old gpu index for restoration
    if(m_currDeviceId>=0 && m_currDeviceId != old_gpu_index) cuda_set_device(net.gpu_index); 

    net.wait_stream = m_waitStream;  // 1 - wait CUDA-stream, 0 - not to wait 
    
    image im; 
    im.c = img.c; 
    im.data = img.data; 
    im.h = img.h; 
    im.w = img.w; 

    image sized; 
    if(net.w == im.w && net.h == im.h)
    {
        sized = make_image(im.w, im.h, im.c); 
        memcpy(sized.data, im.data, im.w*im.h*im.c * sizeof(float)); 
    }
    else sized = resize_image(im, net.w, net.h); 
    layer l = net.layers[net.n-1]; 
    float* X = sized.data; 
    // float* prediction = network_predict(net, X); 
    network_predict(net, X); 

    // if (use_mean) {
    //     memcpy(detector_gpu.predictions[detector_gpu.demo_index], prediction, l.outputs * sizeof(float));
    //     mean_arrays(detector_gpu.predictions, FRAMES, l.outputs, detector_gpu.avg);
    //     l.output = detector_gpu.avg;
    //     detector_gpu.demo_index = (detector_gpu.demo_index + 1) % FRAMES;
    // }

    // interpret results
    int nboxes = 0; 
    int letterbox = 0; 
    float hier_thresh = 0.5; 
    detection* dets = get_network_boxes(&net, im.w, im.h, thresh, hier_thresh, 0,1,&nboxes, letterbox); 
    // std::cout << "before nms: " << nboxes << std::endl; 
    do_nms_sort(dets, nboxes, l.classes, m_nms); // clean non-object and overlap boxes; real objects num may smaller than nboxes 

    std::vector<bbox_t> bbox_vec; 

    for (int i=0;i<nboxes;++i)
    {
        box b = dets[i].bbox;  // TODO: what's the dimension here? 
        const int obj_id = max_index(dets[i].prob, l.classes); 
        const float prob = dets[i].prob[obj_id];
        // std::cout << "prob " << std::setw(2) << i << " :" << prob << std::endl; 
        if(prob > thresh)
        {
            bbox_t bbox; 

            bbox.x = std::max((double)0, (b.x - b.w/2.)*im.w); 
            bbox.y = std::max((double)0, (b.y - b.h/2.)*im.h); 
            bbox.w = b.w*im.w; 
            bbox.h = b.h*im.h;
            bbox.obj_id = obj_id; 
            bbox.prob = prob; 
            bbox.track_id = 0; 
            bbox_vec.push_back(bbox); 
        }
    }

    free_detections(dets, nboxes); 
    if(sized.data) free(sized.data); 

    if(m_currDeviceId != old_gpu_index)cuda_set_device(old_gpu_index); 

    return bbox_vec; 
}



std::vector<bbox_t> YOLO_Batch::detect_si_region(image_t img, float thresh, bool use_mean)
{
    detector_gpu_t &detector_gpu = *static_cast<detector_gpu_t*>(m_detector_gpu_ptr.get());
    network &net = detector_gpu.net; 
    int old_gpu_index = cuda_get_device();  // save old gpu index for restoration
    if(m_currDeviceId >=0 && m_currDeviceId != old_gpu_index) cuda_set_device(net.gpu_index); 

    net.wait_stream = m_waitStream;  // 1 - wait CUDA-stream, 0 - not to wait 
    
    image im; 
    im.c = img.c; 
    im.data = img.data; 
    im.h = img.h; 
    im.w = img.w; 

    image sized; 
    if(net.w == im.w && net.h == im.h)
    {
        sized = make_image(im.w, im.h, im.c); 
        memcpy(sized.data, im.data, im.w*im.h*im.c * sizeof(float)); 
    }
    else sized = resize_image(im, net.w, net.h); 

    layer l = net.layers[net.n-1]; 
    float* X = sized.data; 
    // float* prediction = network_predict(net, X); 
    network_predict(net, X); 

    // interpret results
    // detection* dets = get_network_boxes(&net, im.w, im.h, thresh, hier_thresh, 0,1,&nboxes, letterbox); 
    box* boxes = (box*)calloc(l.w * l.h * l.n, sizeof(box)); 
    float **probs = (float**)calloc(l.w*l.h*l.h, sizeof(float*)); 
    for(int k=0;k<l.w*l.h*l.n;k++)probs[k]=(float*)calloc(l.classes, sizeof(float*)); 
    
    std::vector<bbox_t> bbox_vec; 

    get_region_boxes(l, 1,1,thresh, probs, boxes, 0,0); 
    do_nms(boxes, probs,l.w*l.h*l.n, l.classes, m_nms); 

    for (int k=0;k<l.w*l.h*l.n;++k)
    {
        box b = boxes[k]; 
        const int obj_id = max_index(probs[k], l.classes); 
        const float prob = probs[k][obj_id]; 
        if(prob > thresh)
        {
            bbox_t bbox; 
            bbox.x = std::max((double)0, (b.x - b.w/2.)*im.w); 
            bbox.y = std::max((double)0, (b.y - b.h/2.)*im.h); 
            bbox.w = b.w*im.w; 
            bbox.h = b.h*im.h;
            bbox.obj_id = obj_id; 
            bbox.prob = prob; 
            bbox.track_id = 0; 

            bbox_vec.push_back(bbox); 
        }
    }
    free(boxes); 
    free_ptrs((void**)probs, l.w*l.h*l.n); 
    if(sized.data) free(sized.data); 


    if(m_currDeviceId != old_gpu_index)cuda_set_device(old_gpu_index); 

    return bbox_vec; 
}


std::shared_ptr<image_t> YOLO_Batch::mat_to_image_resize(cv::Mat mat) 
{
    if (mat.data == NULL) return std::shared_ptr<image_t>(NULL);

    cv::Size network_size = cv::Size(get_net_width(), get_net_height());
    cv::Mat det_mat;
    if (mat.size() != network_size)
        cv::resize(mat, det_mat, network_size);
    else
        det_mat = mat;  // only reference is copied

    auto img = mat_to_image(det_mat); 
    return img; 
}

std::shared_ptr<image_t> YOLO_Batch::mat_to_image(cv::Mat img_src)
{
    cv::Mat img;
    cv::cvtColor(img_src, img, cv::COLOR_RGB2BGR);
    std::shared_ptr<image_t> image_ptr(new image_t, [this](image_t *im) { free_image(*im); delete im; });
    std::shared_ptr<IplImage> ipl_small = std::make_shared<IplImage>(img);
    *image_ptr = ipl_to_image(ipl_small.get());
    return image_ptr;
}

image_t YOLO_Batch::ipl_to_image(IplImage* src)
{
    unsigned char *data = (unsigned char *)src->imageData;
    int h = src->height;
    int w = src->width;
    int c = src->nChannels;
    int step = src->widthStep;
    image_t out = make_image_custom(w, h, c);
    int count = 0;

    for (int k = 0; k < c; ++k) {
        for (int i = 0; i < h; ++i) {
            int i_step = i*step;
            for (int j = 0; j < w; ++j) {
                out.data[count++] = data[i_step + j*c + k] / 255.;
            }
        }
    }

    return out;
}

image_t YOLO_Batch::make_empty_image(int w, int h, int c)
{
    image_t out;
    out.data = 0;
    out.h = h;
    out.w = w;
    out.c = c;
    return out;
}

image_t YOLO_Batch::make_image_custom(int w, int h, int c)
{
    image_t out = make_empty_image(w, h, c);
    out.data = (float *)calloc(h*w*c, sizeof(float));
    return out;
}

std::vector<bbox_t> YOLO_Batch::detect_resized_si(
    image_t img_resized, 
    int init_w, 
    int init_h, 
    float thresh, 
    bool use_mean)
{
    if(img_resized.data == NULL)throw std::runtime_error("Image is empty"); 
    std::vector<bbox_t> bboxes; 
    if(m_detectType == 1)
        bboxes = detect_si(img_resized, thresh, use_mean); 
    else bboxes = detect_si_region(img_resized, thresh, use_mean); 
    float wk = (float)init_w / img_resized.w;
    float hk = (float)init_h / img_resized.h;
    for(auto &bbox: bboxes) {
        bbox.x *= wk; 
        bbox.w *= wk; 
        bbox.y *= hk;
        bbox.h *= hk;
    }
    return bboxes; 
}

int YOLO_Batch::get_net_height() const 
{
    detector_gpu_t &detector_gpu = *static_cast<detector_gpu_t*>(m_detector_gpu_ptr.get()); 
    return detector_gpu.net.h; 
}

int YOLO_Batch::get_net_width() const 
{
    detector_gpu_t &detector_gpu = *static_cast<detector_gpu_t*>(m_detector_gpu_ptr.get()); 
    return detector_gpu.net.w; 
}

void YOLO_Batch::free_image(image_t m)
{
    if(m.data) free(m.data); 
}

std::vector<std::string> YOLO_Batch::load_object_names(std::string filename)
{
    std::ifstream namefile(filename); 
    std::vector<std::string> names; 
    if(!namefile.is_open()) 
    {
        std::cout << "cannot open " << filename << std::endl; 
        return names; 
    }

    for(std::string line; getline(namefile, line);)names.push_back(line); 

    return names; 
}

cv::Mat YOLO_Batch::draw_boxes(
    cv::Mat mat_img,
	std::vector<bbox_t> result_vec, 
    std::vector<std::string> obj_names) 
{
	cv::Mat visualImg = mat_img.clone();
	int const colors[6][3] = { { 1,0,1 },{ 0,0,1 },{ 0,1,1 },{ 0,1,0 },{ 1,1,0 },{ 1,0,0 } };
	for (auto &i : result_vec) {
		if (i.prob < 0.5 || obj_names[i.obj_id]!="person") // just show person 
			continue;
		int const offset = i.obj_id * 123457 % 6;
		int const color_scale = 150 + (i.obj_id * 123457) % 100;
		cv::Scalar color(colors[offset][0], colors[offset][1], colors[offset][2]);
		color *= color_scale;
		cv::rectangle(visualImg, cv::Rect(i.x, i.y, i.w, i.h), color, 3);
		if (obj_names.size() > i.obj_id) {
			std::string obj_name = obj_names[i.obj_id];
			// if (i.track_id > 0) obj_name += " - " + std::to_string(i.track_id); // no track now
			const cv::Size text_size = getTextSize(obj_name, cv::FONT_HERSHEY_COMPLEX_SMALL, 0.6, 1, 0);
			int const max_width = (text_size.width > (int)i.w) ? text_size.width : (i.w);
			cv::rectangle(visualImg, cv::Point2f(std::max<int>((int)i.x -3 , 0), // 3 is linewidth of bbox
                          std::max<int>((int)i.y - 17, 0)),cv::Point2f(std::min<int>((int)i.x + max_width, mat_img.cols - 1),
					      std::min<int>((int)i.y, mat_img.rows - 1)),
				          color, CV_FILLED, 1, 0);
			putText(visualImg, obj_name, cv::Point2f(i.x, i.y - 8),
				    cv::FONT_HERSHEY_COMPLEX_SMALL, 0.6, cv::Scalar(255, 255, 255), 1);
		}
	}
	return visualImg;
}

void YOLO_Batch::save_image_t_(image_t& img, const char* name )
{
    image temp;
    temp.c = img.c;temp.w = img.w;temp.h = img.h;temp.data=img.data;
    save_image(temp, name); 
}

std::vector<bbox_t> YOLO_Batch::detect_mat_si(cv::Mat img, float thresh)
{
    if(img.data == NULL)throw std::runtime_error("Image is empty"); 
    auto image_ptr = mat_to_image_resize(img); 
    return detect_resized_si(*image_ptr, img.cols, img.rows, thresh, false); 
}


/*****************batch detection*******************/
// Thanks to jstumpin at https://github.com/AlexeyAB/darknet/issues/878 
BoxesVec YOLO_Batch::detect_batch_region(ImagePtrList img_ptrs, float thresh)
{
    detector_gpu_t& detector_gpu = *static_cast<detector_gpu_t*>(m_detector_gpu_ptr.get());
    network &net = detector_gpu.net; 
    int old_gpu_index = cuda_get_device(); 
    if(m_currDeviceId >=0 && m_currDeviceId != old_gpu_index) cuda_set_device(net.gpu_index); 
    net.wait_stream = m_waitStream; 


    // assume channel 3 
    float *X = (float*)calloc(net.batch*net.w*net.h*3,sizeof(float));
    for(int i=0;i<net.batch;i++)
    {
        image im; 
        im.c = img_ptrs[i]->c;
        im.w = img_ptrs[i]->w; 
        im.h = img_ptrs[i]->h; 
        im.data = img_ptrs[i]->data; 
        image sized; 
        if(net.w==im.w && net.h==im.h)
        {
            sized = make_image(im.w,im.h,im.c); 
            memcpy(sized.data, im.data, im.w*im.h*im.c*sizeof(float)); 
        }
        else sized = resize_image(im, net.w, net.h); 
        memcpy(X+i*net.h*net.w*3, sized.data, net.h*net.w*3*sizeof(float)); 
        free(sized.data); 
    } 

    // predict 
    network_predict(net, X); 
    layer l = net.layers[net.n-1]; 



    // get bbox
    BoxesVec bbox_vec_batch; 
    for(int j=0;j<net.batch;j++)
    {
        box* boxes = (box*)calloc(l.w*l.h*l.n, sizeof(box)); // so what is layer.n
        float** probs = (float**)calloc(l.w*l.h*l.n, sizeof(float*)); 
        for(int j=0;j<l.w*l.h*l.n;j++)
        {
            probs[j] = (float*)calloc(l.classes, sizeof(float)); 
        }
        Boxes bbox_vec; 
        get_region_boxes(l,1,1,thresh,probs,boxes, 0, 0); 
        do_nms(boxes, probs, l.w*l.h*l.n, l.classes, m_nms);
        for(int k=0;k<l.w*l.h*l.n;k++)
        {
            // box b = boxes[k]; 
            const int obj_id = max_index(probs[k], l.classes); 
            const float prob = probs[k][obj_id]; 
            if(prob > thresh)
            {
                // bbox_t bbox;
                // bbox.x = std::max((double)0, (b.x - b.w/2.)*imgs[j].w); 
                // bbox.y = std::max((double)0, (b.y - b.h/2.)*imgs[j].h); 
                // bbox.w = b.w*imgs[j].w; 
                // bbox.h = b.h*imgs[j].h;
                // bbox.obj_id = obj_id;
                // bbox.track_id = 0;  
                // bbox.prob = prob; 

                bbox_t bbox;
                if (boxes[k].w > 1) {
                    bbox.x = 0;
                    bbox.w = img_ptrs[j]->w;
                }
                else {
                    float w = boxes[k].w * img_ptrs[j]->w;
                    bbox.x = round(boxes[k].x * img_ptrs[j]->w - w / 2);
                    bbox.w = w;
                }
                if (boxes[k].h > 1) {
                    bbox.y = 0;
                    bbox.h = img_ptrs[j]->h;
                }
                else {
                    float h = boxes[k].h * img_ptrs[j]->h;
                    bbox.y = round(boxes[k].y * img_ptrs[j]->h - h / 2);
                    bbox.h = h;
                }
                bbox.obj_id = obj_id;
                bbox.track_id = 0;  
                bbox.prob = prob; 

                bbox_vec.push_back(bbox); 
            }
        }
        bbox_vec_batch.push_back(bbox_vec); 

        l.output += l.h*l.w*l.n*(l.classes + l.coords + 1); // stepping
        // l.output = l.output + l.outputs; 
        free(boxes); 
        free_ptrs((void**)probs, l.w*l.h*l.n);
    }


    free(X); 

    if(m_currDeviceId != old_gpu_index) cuda_set_device(old_gpu_index); 
    return bbox_vec_batch;
}

BoxesVec YOLO_Batch::detect_resized_batch(
    ImagePtrList imgs_resized, 
    int init_w, 
    int init_h, 
    float thresh)
{
    BoxesVec bbox_vec_batch;
    if(m_detectType == 1)
        bbox_vec_batch = detect_batch_yolo(imgs_resized, thresh); 
    else bbox_vec_batch = detect_batch_region(imgs_resized, thresh); 

    float wk = (float)init_w / imgs_resized[0]->w;
    float hk = (float)init_h / imgs_resized[0]->h; 
    for(int i=0;i<m_batchsize;i++)
    {
        std::vector<bbox_t>& bbox_vec = bbox_vec_batch[i]; 
        for(auto &bbox:bbox_vec)
        {
            bbox.x *= wk; 
            bbox.w *= wk; 
            bbox.y *= hk; 
            bbox.h *= hk; 
        }
    }
    return bbox_vec_batch; 
}

BoxesVec YOLO_Batch::detect_mat_batch(
    std::vector<cv::Mat>& mats, 
    float thresh)
{
    ImagePtrList img_ptrs; 
    for(int i=0;i<m_batchsize;i++)
    {
        auto image_ptr = mat_to_image_resize(mats[i]); 
        img_ptrs.push_back(image_ptr);
    }
    return detect_resized_batch(img_ptrs, mats[0].cols, mats[0].rows, thresh ); 
}


BoxesVec YOLO_Batch::detect_batch_yolo(ImagePtrList img_ptrs, float thresh)
{
    detector_gpu_t& detector_gpu = *static_cast<detector_gpu_t*>(m_detector_gpu_ptr.get());
    network &net = detector_gpu.net; 
    int old_gpu_index = cuda_get_device(); 
    if(m_currDeviceId >=0 && m_currDeviceId != old_gpu_index) cuda_set_device(net.gpu_index); 
    net.wait_stream = m_waitStream; 

    // assume channel 3 

    // std::cout << "debug: new X" << std::endl; 
    float *X = (float*)calloc(net.batch*net.w*net.h*3,sizeof(float));
    for(int i=0;i<net.batch;i++)
    {
        image im; 
        im.c = img_ptrs[i]->c;
        im.w = img_ptrs[i]->w; 
        im.h = img_ptrs[i]->h; 
        im.data = img_ptrs[i]->data; 
        image sized; 
        if(net.w==im.w && net.h==im.h)
        {
            sized = make_image(im.w,im.h,im.c); 
            memcpy(sized.data, im.data, im.w*im.h*im.c*sizeof(float)); 
        }
        else sized = resize_image(im, net.w, net.h); 
        memcpy(X+i*net.h*net.w*3, sized.data, net.h*net.w*3*sizeof(float)); 

        free(sized.data); 
    } 
    // std::cout << "debug: predict  "  << std::endl; 
    // predict 
    network_predict(net, X); 
    layer l = net.layers[net.n-1]; 
   // get bbox
    BoxesVec bbox_vec_batch; 

    for(int j=0;j<net.batch;j++)
    {
        // std::cout << "get box " << j << std::endl; 
        int nboxes = 0; 
        int letterbox = 0; 
        float hier_thresh = 0.5; 
        detection* dets = get_network_boxes(&net,img_ptrs[j]->w,img_ptrs[j]->h,
                                             thresh, hier_thresh,
                                             0,1,&nboxes, letterbox);
        do_nms_sort(dets, nboxes, l.classes,m_nms); 

        std::vector<bbox_t> bbox_vec; 
        for(int i=0;i<nboxes;++i)
        {
            box b = dets[i].bbox;  
            const int obj_id = max_index(dets[i].prob, l.classes); 
            const float prob = dets[i].prob[obj_id];
            if(prob>thresh)
            {
                bbox_t bbox; 
                bbox.x = std::max((double)0,(b.x-b.w/2.)*img_ptrs[j]->w);
                bbox.y = std::max((double)0,(b.y-b.h/2.)*img_ptrs[j]->h);
                bbox.w = b.w*img_ptrs[j]->w;
                bbox.h = b.h*img_ptrs[j]->h; 
                bbox.obj_id = obj_id;
                bbox.prob = prob;
                bbox.track_id = 0;

                bbox_vec.push_back(bbox);
            }
        }

        // std::cout << "debug: free detection " << std::endl; 
        bbox_vec_batch.push_back(bbox_vec); 
        free_detections(dets, nboxes); 
        // stepping 
        // std::cout << "debug: stepping" << std::endl; 
        for(int j=0;j<net.n;j++)
        {
            layer& temp_l = net.layers[j];
            if(temp_l.type==YOLO || temp_l.type==REGION || temp_l.type==DETECTION)
            {
                temp_l.output = temp_l.output + temp_l.outputs; 
            }
        }
    }
    for(int j=0;j<net.n;j++)
    {
        layer& temp_l = net.layers[j];
        if(temp_l.type==YOLO || temp_l.type==REGION || temp_l.type==DETECTION)
        {
            for(int i=0;i<net.batch;i++)
            temp_l.output = temp_l.output - temp_l.outputs; 
        }
    }
    // std::cout << "debug: free x " << std::endl; 
    if(X)
    free(X); 

    if(m_currDeviceId != old_gpu_index) cuda_set_device(old_gpu_index); 

    return bbox_vec_batch;
}


/******************* YOLO BATCH START****************************/
// Wrapper written over the init function to initialize the yolo model
// input : <config file> <weights file> <names file> <batch size> <device ID>
int YOLO_Batch::init_batch(std::string cfgfile, std::string weightfile, std::string namesfile, int batch_size,int deviceId)
{
    init(cfgfile, weightfile, batch_size, deviceId);
    m_detectType = 1;
    names = load_object_names(namesfile);

    return 0;
}

// function for batch detection:
// input: std::vector<image_t> images_list - list of images in image_t format 
// output: 
// void YOLO_Batch::run_detector_batch(std::vector<std::string> img_paths)
BoxesVec YOLO_Batch::run_detector_batch(std::vector<image_t> images_list)
{
    std::cout << "[INFO] running batch detector \n";

    // int batch = img_paths.size(); 
    std::cout << "batch size: " << batch_size << std::endl; 
    std::vector<cv::Mat> mats; 
    for(int i=0;i<batch_size;i++) 
    {
        // cv::Mat img = cv::imread(img_paths[i]); 
        cv::Mat img(images_list[i].h, images_list[i].w, CV_8UC3, images_list[i].data);

        if(img.empty()) 
        {
            std::cout << "img empty" << std::endl; 
            exit(-1);
        }
        cv::Mat small; 
        cv::resize(img, small,cv::Size(416,416)); 
        mats.push_back(small); 
    }
#if DISPLAY_OUT
    // test bathch 
    std::cout << "debug: detector initialization success" << std::endl; 

    struct timeval tp;
    gettimeofday(&tp, NULL);
    long int ms = tp.tv_sec * 1000 + tp.tv_usec / 1000;
#endif

    BoxesVec bbox_vec_batch;
    try {
        //BoxesVec 
        bbox_vec_batch = detect_mat_batch(mats); 
    }
    catch (std::exception& e)
    {
        std::cout << e.what() << "\n";
    }

#if DISPLAY_OUT
    struct timeval tp1;
    gettimeofday(&tp1, NULL);
    long int ms1 = tp1.tv_sec * 1000 + tp1.tv_usec / 1000;
    long int elapse_detection = ms1 - ms; 
    
    std::cout << "batch detection time:" << elapse_detection << " ms" << std::endl; 
    for(int i=0;i<batch_size;i++)
    {
        cv::Mat visual_box; 


        visual_box = draw_boxes(mats[i], bbox_vec_batch[i], names); 
        // visualize 
        cv::imshow("detection", visual_box);
        cv::waitKey(); 
    }
#endif 

    return bbox_vec_batch;
}
/******************* YOLO BATCH END ****************************/