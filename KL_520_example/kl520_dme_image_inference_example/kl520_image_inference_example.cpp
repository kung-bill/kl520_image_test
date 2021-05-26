/**
 * @file      kl520_image_inference_example.cpp
 */

#include "errno.h"
#include "kdp_host.h"
#include "stdio.h"
#include <string.h>
#include <unistd.h>
#include "user_util.h"
#include "ipc.h"
#include "base.h"
#include "model_res.h"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <vector>


using namespace std;
#include "post_processing_ex.h"
#include "kdpio.h"

extern "C" {
uint32_t round_up(uint32_t num);
int post_processing_simple(int model_id, struct kdp_image_s *image_p, float* fres, int* fres_len);
}

#if defined(__cplusplus) || defined(c_plusplus)
extern "C" {
#endif

#define DME_MODEL_FILE      ("../../input_models/KL520/ssd_fd_lm/models_520.nef")
#define DME_MODEL_SIZE (20 * 1024 * 1024)

#define IMG_SOURCE_W 640
#define IMG_SOURCE_H 480
#define FD_IMG_SIZE (IMG_SOURCE_W * IMG_SOURCE_H * 2)
#define FD_MIN_SCORE        (0.5f)

//customized define
#define MY_DME_MODEL_FILE   ("../../input_models/KL520/my_test_model/models_520.nef")
#define MY_IMG_SOURCE_W 224
#define MY_IMG_SOURCE_H 224
#define MY_FD_IMG_SIZE (MY_IMG_SOURCE_W * MY_IMG_SOURCE_H * 2)
#define MY_FD_MIN_SCORE        (0.5f)
// To print the dection results
//#define LOG
std::vector<cv::Mat> frames;

/**
 * get the data for all output nodes and do postprocessing for YOLO in host side
 * dev_idx: connected device ID. A host can connect several devices
 * inf_size: the size of inference result
 * post_par: parameters for postprocessing in host side
 * return 0 on succeed, 1 on exit
 */
void get_detection_res(int dev_idx, char* inf_res, uint32_t inf_size, struct post_parameter_s post_par, \
                       float* fres, int* fres_len)
{
    // array byte size larger than the size of the data for all output nodes

    // Get the data for all output nodes: TOTAL_OUT_NUMBER + (H/C/W/RADIX/SCALE) + (H/C/W/RADIX/SCALE) + ...
    // + FP_DATA (size: H*C*16*CEIL(W/16) ) + FP_DATA + ...


    // Prepare for postprocessing      
    int output_num = inf_res[0];
    cout << "output_num=" << output_num << endl;
    struct output_node_params *p_node_info;
    int r_len, offset;
    struct yolo_result_s *det_res = (struct yolo_result_s *)calloc(1, sizeof(dme_yolo_res));
    struct kdp_image_s *image_p = (struct kdp_image_s *)calloc(1, sizeof(struct kdp_image_s));
    offset = sizeof(int) + output_num * sizeof(output_node_params);

    // save the parameters to struct kdp_image_s
    RAW_INPUT_COL(image_p) = post_par.raw_input_col;
    RAW_INPUT_ROW(image_p) = post_par.raw_input_row;
    DIM_INPUT_COL(image_p) = post_par.model_input_row;
    DIM_INPUT_ROW(image_p) = post_par.model_input_row;
    RAW_FORMAT(image_p) = post_par.image_format;
    POSTPROC_RESULT_MEM_ADDR(image_p) = (uint32_t *)det_res;
    POSTPROC_OUTPUT_NUM(image_p) = output_num;

    for (int i = 0; i < output_num; i++) {
        if (check_ctl_break())
            return;
        // parse the H/C/W/Radix/Scale info
        p_node_info = (struct output_node_params *)(inf_res + sizeof(int) + i * sizeof(output_node_params));
        r_len = p_node_info->channel * p_node_info->height * round_up(p_node_info->width);
        // save the parameters to struct kdp_image_s
        POSTPROC_OUT_NODE_ADDR(image_p, i) = inf_res + offset;
        POSTPROC_OUT_NODE_ROW(image_p, i) = p_node_info->height;
        POSTPROC_OUT_NODE_CH(image_p, i) = p_node_info->channel;
        POSTPROC_OUT_NODE_COL(image_p, i) = p_node_info->width;
        POSTPROC_OUT_NODE_RADIX(image_p, i) = p_node_info->radix;
        POSTPROC_OUT_NODE_SCALE(image_p, i) = p_node_info->scale;
        // offset for next H/C/W/Radix/Scale info
        offset = offset + r_len;
    }            
    
    // Do postprocessing
    //post_yolo_v3(0, image_p);
    post_processing_simple(0, image_p, fres, fres_len);

    free(image_p);
    free(det_res);
}

//my image (.bmp 24bit) inference test
int user_img_inf_test(int dev_idx, int user_id, int mode, struct kdp_dme_cfg_s dme_cfg, \
                          struct post_parameter_s post_par)
{
    int ret = 0;
    uint32_t ret_model_id = 0;

    if (1) {
        char* p_buf = NULL;
        uint32_t buf_len = 0;
        uint32_t ret_size = 0;

        // read model data
        p_buf = new char[DME_MODEL_SIZE];
        memset(p_buf, 0, DME_MODEL_SIZE);
        int n_len = read_file_to_buf(p_buf, MY_DME_MODEL_FILE, DME_MODEL_SIZE);
        if (n_len <= 0) {
            printf("reading model file failed:%d...\n", n_len);
            delete[] p_buf;
            return -1;
        }
        buf_len = n_len;

        printf("starting DME inference in [serial mode] ...\n");
        int ret = kdp_start_dme_ext(dev_idx, p_buf, buf_len, &ret_size);
        if (ret != 0) {
            printf("could not set to DME mode:%d..\n", ret_size);
            delete[] p_buf;
            return -1;
        }

        delete[] p_buf;
        printf("DME mode succeeded...\n");
        usleep(SLEEP_TIME);
    }

    if (1) {
        int dat_size = 0;

        dat_size = sizeof(struct kdp_dme_cfg_s);
        printf("starting DME configure ...\n");
	
        ret = kdp_dme_configure(dev_idx, (char *)&dme_cfg, dat_size, &ret_model_id);
        if (ret != 0) {
            printf("could not set to DME configure mode..\n");
            return -1;
        }
        printf("DME configure model[%d] succeeded...\n", ret_model_id);
        usleep(SLEEP_TIME);
    }

    if (1) {
        uint32_t inf_size = 0;
        bool res_flag = true;

        uint32_t buf_len = MY_FD_IMG_SIZE;
        char inf_res[2560];

        double start;
        float fps = 0;

        printf("starting DME inference ...\n");
        start = what_time_is_it_now();

        //load image
        cv::Mat frame_input; //frame_input need greater than 224*224, because KL520 not support upsampling
	

	//frame_input = cv::imread("../../input_images/one_bike_many_cars_224x224.bmp");
	//frame_input = cv::imread("../../images/testI.bmp");
        frame_input = cv::imread("../../images/man.bmp");	
					
        //image resize
	//cv::Size size(MY_IMG_SOURCE_W, MY_IMG_SOURCE_H);
        //cv::resize(frame_input, frame_input, size);
	
	// cv::imshow cant work at BGR565 format
        cvtColor(frame_input, frame_input, CV_BGR2BGR565); 

        frames.push_back(frame_input);
        IplImage ipl_img;

#if CV_MAJOR_VERSION > 3 || (CV_MAJOR_VERSION == 3 && CV_SUBMINOR_VERSION >= 9)
        ipl_img = cvIplImage(frame_input);
#else
        ipl_img = (IplImage)frame_input;
#endif
	//printf("buf_len size = %d \n", buf_len);
	
        ret = kdp_dme_inference(dev_idx, (char *)ipl_img.imageData, buf_len, &inf_size, &res_flag, (char*) &inf_res, 0, ret_model_id);
        if (ret != 0) {
            printf("Inference failed..\n");
            return -1;
        }
	

	
        kdp_dme_retrieve_res(dev_idx, 0, inf_size, (char*) &inf_res);
	
	
        int max_output_number = 10; //the maximum ouput float number in your network(in this case > 5)
        int fres_len[1] = {0};
        float fres[max_output_number] = {0};
        get_detection_res(dev_idx, inf_res, inf_size, post_par, fres, fres_len);


        //print network output
        cout << "your network output is : " << endl;
        for(int i=0; i<fres_len[0]; ++i){
            cout << fres[i] << ", ";	
	}
        cout << endl;
	fps = 1./(what_time_is_it_now() - start);
        printf("[INFO] time on 1 iamge: %f ms/frame, fps: %f\n", (1/fps)*1000, fps);
        

        cvtColor(frames.front(), frames.front(), CV_BGR5652BGR);
	
        cv::imshow("Display window", frames.front());
	while(!(cv::waitKey(10) > 0)){	
	}
       
        vector<cv::Mat>::iterator first_frame = frames.begin();
        frames.erase(first_frame);
        

        printf("DME inference succeeded...\n");
        kdp_end_dme(dev_idx);
        usleep(SLEEP_TIME);
    }
    return 0;
}

int user_test(int dev_idx, int user_id)
{
    uint16_t mode = 0;
    struct post_parameter_s post_par;
    struct kdp_dme_cfg_s dme_cfg = create_dme_cfg_struct();

    // parameters for postprocessing in host side
    post_par.raw_input_row   = MY_IMG_SOURCE_H;
    post_par.raw_input_col   = MY_IMG_SOURCE_W;
    post_par.model_input_row = MY_IMG_SOURCE_H;
    post_par.model_input_col = MY_IMG_SOURCE_W;
    post_par.image_format    = IMAGE_FORMAT_SUB128 | NPU_FORMAT_RGB565 | IMAGE_FORMAT_RAW_OUTPUT;

    // dme configuration
    dme_cfg.model_id     = 57;
    dme_cfg.output_num   = 5;// the nerual in your .onnx output layer            
    dme_cfg.image_col    = post_par.raw_input_col;
    dme_cfg.image_row    = post_par.raw_input_row;
    dme_cfg.image_ch     = 3;
    dme_cfg.image_format = post_par.image_format;


    user_img_inf_test(dev_idx, user_id, mode, dme_cfg, post_par);
    
    return 0;
}

#if defined(__cplusplus) || defined(c_plusplus)
}
#endif
