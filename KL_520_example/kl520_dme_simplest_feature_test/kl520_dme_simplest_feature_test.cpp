/**
 * @file        kl520_dme_async_mobilenet_classification.cpp
 * @brief       kdp host lib user test examples
 * @version     0.1
 * @date        2019-08-01
 *
 * @copyright   Copyright (c) 2019-2021 Kneron Inc. All rights reserved.
 */


#include "errno.h"
#include "kdp_host.h"
#include "stdio.h"

#include <string.h>
#include <unistd.h>
#include <stdlib.h>
#include "user_util.h"
#include "post_processing_ex.h"
#include "kdpio.h"
#include "ipc.h"
#include "base.h"

#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>

extern "C" {
uint32_t round_up(uint32_t num);
int post_imgnet_classification(int model_id, struct kdp_image_s *image_p);
int post_processing_simplest(int model_id, struct kdp_image_s *image_p, float *res_float_array, int res_float_array_max, int *res_float_len);
float get_float(int h, int w, int c, int image_p_h, int image_p_w, int image_p_c, float *res_float_array);
}

#if defined(__cplusplus) || defined(c_plusplus)
extern "C" {
#endif

#define DME_MODEL_FILE      (HOST_LIB_DIR "/input_models/KL520/test_model/models_520.nef")

#define DME_IMG_SIZE (224 * 224 * 2)
#define DME_MODEL_SIZE (20 * 1024 * 1024)
#define DME_FWINFO_SIZE 128

#define MODEL_IMG_W 224
#define MODEL_IMG_H 224
#define INFERENCE_IMG_SIZE (MODEL_IMG_W * MODEL_IMG_H * 2)

void get_detection_res(int dev_idx, uint32_t inf_size, struct post_parameter_s post_par)
{
    char inf_res[256000];
    // Get the data for all output nodes: TOTAL_OUT_NUMBER + (H/C/W/RADIX/SCALE) + (H/C/W/RADIX/SCALE) + ...
    // + FP_DATA + FP_DATA + ...
    kdp_dme_retrieve_res(dev_idx, 0, inf_size, inf_res);

    // Prepare for postprocessing      
    int output_num = inf_res[0];
    struct output_node_params *p_node_info;
    int r_len, offset;
    struct imagenet_result_s *det_res = (struct imagenet_result_s *)calloc(IMAGENET_TOP_MAX, sizeof(imagenet_result_s));

    struct kdp_image_s *image_p = (struct kdp_image_s *)calloc(1, sizeof(struct kdp_image_s));
    offset = sizeof(int) + output_num * sizeof(output_node_params);

    // Struct to pass the parameters
    RAW_INPUT_COL(image_p) = post_par.raw_input_col;
    RAW_INPUT_ROW(image_p) = post_par.raw_input_row;
    DIM_INPUT_COL(image_p) = post_par.model_input_col;
    DIM_INPUT_ROW(image_p) = post_par.model_input_row;
    RAW_FORMAT(image_p) = post_par.image_format;
    POSTPROC_RESULT_MEM_ADDR(image_p) = (uint32_t *)det_res;
    POSTPROC_OUTPUT_NUM(image_p) = output_num;

    for (int i = 0; i < output_num; i++) {
        if (check_ctl_break())
            return;
        p_node_info = (struct output_node_params *)(inf_res + sizeof(int) + i * sizeof(output_node_params));
        r_len = p_node_info->channel * p_node_info->height * round_up(p_node_info->width);
                   
        POSTPROC_OUT_NODE_ADDR(image_p, i) = inf_res + offset;
        POSTPROC_OUT_NODE_ROW(image_p, i) = p_node_info->height;
        POSTPROC_OUT_NODE_CH(image_p, i) = p_node_info->channel;
        POSTPROC_OUT_NODE_COL(image_p, i) = p_node_info->width;
        POSTPROC_OUT_NODE_RADIX(image_p, i) = p_node_info->radix;
        POSTPROC_OUT_NODE_SCALE(image_p, i) = p_node_info->scale;

        offset = offset + r_len;
    }            
    
    // Do postprocessing
    int res_float_array_max = 30000;
    float *res_float_array = (float*)malloc(sizeof(float)*res_float_array_max);
    int *res_float_len = (int*)malloc(sizeof(int));

    post_processing_simplest(0, image_p, res_float_array, res_float_array_max, res_float_len);


    int image_p_h = p_node_info->height;
    int image_p_w = p_node_info->width;
    int image_p_c = p_node_info->channel;

 
    for (int c = 0; c < image_p_c; ++c){
        printf("(h,w,c)=(%d, %d, %d), %f\n", 0, 0, c, get_float(0, 0, c, image_p_h, image_p_w, image_p_c, res_float_array));

    }

    free(image_p);
    free(det_res);
}

int user_test_dme(int dev_idx, struct post_parameter_s post_par, \
                  struct kdp_dme_cfg_s dme_cfg)
{
    uint32_t model_id = 0;
    int ret = 0;
    if (1) {
        printf("reading model NEF file from '%s'\n", DME_MODEL_FILE);

        long model_size;
        char *model_buf = read_file_to_buffer_auto_malloc(DME_MODEL_FILE, &model_size);
        if(model_buf == NULL)
            return -1;

        printf("starting DME inference ...\n");
        uint32_t ret_size = 0;
        int ret = kdp_start_dme_ext(dev_idx, model_buf, model_size, &ret_size);
        if (ret != 0) {
            printf("could not set to DME mode:%d..\n", ret_size);
            free(model_buf);
            return -1;
        }

        free(model_buf);
        
        printf("DME mode succeeded...\n");
        usleep(SLEEP_TIME);
    }

    if (1) {
        int dat_size = 0;

        dat_size = sizeof(struct kdp_dme_cfg_s);
        printf("starting DME configure ...\n");
        int ret = kdp_dme_configure(dev_idx, (char *)&dme_cfg, dat_size, &model_id);
        if (ret != 0) {
            printf("could not set to DME configure mode..\n");
            return -1;
        }
        printf("DME configure model [%d] succeeded...\n", model_id);
        usleep(SLEEP_TIME);
    }

    if (1) {
        uint32_t inf_size = 0;
        bool res_flag = true;


        uint32_t buf_len = INFERENCE_IMG_SIZE;
        char *inf_res = (char *)malloc(256*1024);

        cv::Mat img, img_resized, img565;
        
        // img = cv::imread("../../images/birdman.bmp");
        img = cv::imread("../../images/img09.bmp");

        //image resize
	    // cv::Size size(MODEL_IMG_W, MODEL_IMG_H);
        

        cvtColor(img, img565, CV_BGR2BGR565);
        IplImage ipl_img;

#if CV_MAJOR_VERSION > 3 || (CV_MAJOR_VERSION == 3 && CV_SUBMINOR_VERSION >= 9)
        ipl_img = cvIplImage(img565);
#else
        ipl_img = (IplImage)img565;
#endif
	    printf("buf_len size = %d \n", buf_len);
	
        uint32_t ssid = 0;
        ret = kdp_dme_inference(dev_idx, (char *)ipl_img.imageData, buf_len, &inf_size, &res_flag, (char*) &inf_res, 0, model_id);
                  
        // Return if not succeed after retry for 2 times.
        if (ret == -1) {
            printf("could not set to DME inference mode..[error = %d]\n", ret);
            return -1;
        }

        printf("ssid = %d\n", ssid);
        
        get_detection_res(dev_idx, inf_size, post_par);
        

        printf("DME inference succeeded...\n");
        kdp_end_dme(dev_idx);


        cv::imshow("Display window", img);
        // press any key to leave 
	    while(!(cv::waitKey(10) > 0)){	
	    }

        free(inf_res);
    }
    return 0;
}

int user_test(int dev_idx, int user_id)
{
    struct post_parameter_s post_par;
    struct kdp_dme_cfg_s dme_cfg = create_dme_cfg_struct();
    
    // parameters for postprocessing
    post_par.raw_input_col   = 224;
    post_par.raw_input_row   = 224;
    post_par.model_input_col = post_par.raw_input_col;
    post_par.model_input_row = post_par.raw_input_row;
    post_par.image_format    = IMAGE_FORMAT_SUB128 | NPU_FORMAT_RGB565 | IMAGE_FORMAT_RAW_OUTPUT;

    // dme configuration
    dme_cfg.model_id     = 1;// model id when compiling in toolchain
    dme_cfg.output_num   = 1;                             // number of output node for the model
    dme_cfg.image_col    = post_par.raw_input_col;
    dme_cfg.image_row    = post_par.raw_input_row;
    dme_cfg.image_ch     = 3;
    dme_cfg.image_format = post_par.image_format;
    

    //dme test
    user_test_dme(dev_idx, post_par, dme_cfg);

    return 0;
}

#if defined(__cplusplus) || defined(c_plusplus)
}
#endif
