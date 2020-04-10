/*
 * Copyright (c) 2020 oscar
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

/**
 * @file 
 * Implements livepeer frame filter using deep convolutional networks. 
 */

#include "libavformat/avio.h"
#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"
#include "libavutil/avassert.h"
#include "libavutil/imgutils.h"
#include "libavutil/hwcontext.h"
#include "libavutil/hwcontext_internal.h"
#include "libswscale/swscale.h"
#include "avfilter.h"
#include "dnn_interface.h"
#include "formats.h"
#include "internal.h"

typedef enum {LVPDNN_CLASSIFY, LVPDNN_ODETECT} LVPDNNType;

#define MAX_DEVICE_SIZE 16
#define MAX_STRING_SIZE 256

typedef struct LVPDnnLoadData {
    DNNBackendType backend_type;
     //datas to be pre-loaded
    DNNModule   *dnn_module;
    DNNModel    *dnn_model;
    // input & output of the model at execution time
    DNNData     *dnn_input;
    DNNData     *dnn_output;

    char    *model_filename;
    char    *model_inputname;
    char    *model_outputname;

}LVPDnnLoadData;

LVPDnnLoadData** loadedmodels = NULL;

typedef struct LVPDnnContext {
    const AVClass *class;

    int     filter_type;
    char    *model_filename;
    DNNBackendType backend_type;
    char    *model_inputname;
    char    *model_outputname;
    int     sample_rate;
    char    *log_filename;
    int     device_id;

    struct SwsContext   *sws_rgb_scale;
    struct SwsContext   *sws_gray8_to_grayf32;

    struct AVFrame      *swscaleframe;
    struct AVFrame      *swframeforHW;

    FILE                *logfile;
    int                 framenum;

    struct SwsContext   *sws_grayf32_to_gray8;

    //pre-loaded datas poiter. don't free at uninit.
    LVPDnnLoadData      *dnndata;

} LVPDnnContext;

#define OFFSET(x) offsetof(LVPDnnContext, x)
#define FLAGS AV_OPT_FLAG_FILTERING_PARAM | AV_OPT_FLAG_VIDEO_PARAM
static const AVOption lvpdnn_options[] = {
    { "filter_type", "filter type(lvpclassify/lvpodetect)",   OFFSET(filter_type),    AV_OPT_TYPE_INT,    { .i64 = 0 },    0, 1, FLAGS, "type" },
    { "lvpclassify",    "classify filter flag",           0,                      AV_OPT_TYPE_CONST,      { .i64 = 0 },    0, 0, FLAGS, "type" },
    { "lvpodetect",     "detect filter flag",             0,                      AV_OPT_TYPE_CONST,      { .i64 = 1 },    0, 0, FLAGS, "type" },
    { "dnn_backend", "DNN backend",                OFFSET(backend_type),     AV_OPT_TYPE_INT,           { .i64 = 1 },    0, 1, FLAGS, "backend" },
    { "native",      "native backend flag",        0,                        AV_OPT_TYPE_CONST,         { .i64 = 0 },    0, 0, FLAGS, "backend" },
#if (CONFIG_LIBTENSORFLOW == 1)
    { "tensorflow",  "tensorflow backend flag",    0,                        AV_OPT_TYPE_CONST,         { .i64 = 1 },    0, 0, FLAGS, "backend" },
#endif
    { "device",     "GPU id for model loading",     OFFSET(device_id),    AV_OPT_TYPE_INT,              { .i64 = 0   },  0, 16, FLAGS },
    { "model",       "path to model file",          OFFSET(model_filename),   AV_OPT_TYPE_STRING,       { .str = NULL }, 0, 0, FLAGS },
    { "input",       "input name of the model",     OFFSET(model_inputname),  AV_OPT_TYPE_STRING,       { .str = NULL }, 0, 0, FLAGS },
    { "output",      "output name of the model",    OFFSET(model_outputname), AV_OPT_TYPE_STRING,       { .str = NULL }, 0, 0, FLAGS },
    { "sample","detector one every sample frames",  OFFSET(sample_rate),    AV_OPT_TYPE_INT,            { .i64 = 1   },  0, 200, FLAGS },
    { "log",         "path name of the log",        OFFSET(log_filename), AV_OPT_TYPE_STRING,           { .str = NULL }, 0, 0, FLAGS },    
    { NULL }
};

AVFILTER_DEFINE_CLASS(lvpdnn);

static av_cold int init(AVFilterContext *context)
{
    LVPDnnLoadData      *dnndata;
    LVPDnnContext *ctx = context->priv;
    if(ctx->filter_type == LVPDNN_ODETECT) {
        av_log(ctx, AV_LOG_ERROR, "Object detection filter will be implemented in the future.\n");
        return AVERROR(EINVAL);
    }
    if(ctx->backend_type == DNN_NATIVE) {
        av_log(ctx, AV_LOG_ERROR, "Native implementation is under testing.\n");
        return AVERROR(EINVAL);
    }
    if(ctx->device_id < 0 || ctx->device_id >= MAX_DEVICE_SIZE) {
        av_log(ctx, AV_LOG_ERROR, "invalid device id. should be between 0 and 15\n");
        return AVERROR(EINVAL);
    }
    if(loadedmodels == NULL || loadedmodels[ctx->device_id] == NULL) {
        av_log(ctx, AV_LOG_ERROR, "Didn't initialize dnn model for device id %d\n", ctx->device_id);
        return AVERROR(EINVAL);
    }

    dnndata = loadedmodels[ctx->device_id];
    if (strcmp(ctx->model_filename, dnndata->model_filename) != 0) {
        av_log(ctx, AV_LOG_ERROR, "model file for network is not matched with pre-loaded data\n");
        return AVERROR(EINVAL);
    }
    if (strcmp(ctx->model_inputname, dnndata->model_inputname) != 0) {
        av_log(ctx, AV_LOG_ERROR, "input name of the model network is not matched with pre-loaded data\n");
        return AVERROR(EINVAL);
    }
    if (strcmp(ctx->model_outputname, dnndata->model_outputname) != 0) {
        av_log(ctx, AV_LOG_ERROR, "output name of the model network is not matched with pre-loaded data\n");
        return AVERROR(EINVAL);
    }

    ctx->dnndata = dnndata;
    
    if(ctx->log_filename) {
        ctx->logfile = fopen(ctx->log_filename, "w");
    }
    else {
        ctx->logfile = NULL;
        av_log(ctx, AV_LOG_INFO, "output file for log is not specified\n");
    }
    
    ctx->framenum = 0;

    return 0;
}

static int query_formats(AVFilterContext *context)
{
    static const enum AVPixelFormat pix_fmts[] = {
        AV_PIX_FMT_RGB24, AV_PIX_FMT_BGR24, AV_PIX_FMT_GRAY8, AV_PIX_FMT_GRAYF32,
        AV_PIX_FMT_YUV420P, AV_PIX_FMT_YUV422P, AV_PIX_FMT_YUV444P,
        AV_PIX_FMT_YUV410P, AV_PIX_FMT_YUV411P,
        AV_PIX_FMT_CUDA, AV_PIX_FMT_NONE
    };
    AVFilterFormats *fmts_list = ff_make_format_list(pix_fmts);
    return ff_set_common_formats(context, fmts_list);
}

static int prepare_sws_context(AVFilterLink *inlink)
{
    int result = 0;
    enum AVPixelFormat fmt = inlink->format;
    AVFilterContext *context  = inlink->dst;
    LVPDnnContext *ctx = context->priv;
    DNNDataType input_dt  = ctx->dnndata->dnn_input->dt;

    //check hwframe 
    if (inlink->hw_frames_ctx)
    {
        enum AVPixelFormat *formats;

        result = av_hwframe_transfer_get_formats(inlink->hw_frames_ctx,
                                            AV_HWFRAME_TRANSFER_DIRECTION_FROM,
                                            &formats, 0);
        if(result < 0) {
            av_log(ctx, AV_LOG_ERROR, "could not find HW pixel format for scale\n");
            return result;
        }
        //pick the first supported one of possible formats usable in hardware surface for downloading to CPU
        fmt = formats[0];
        av_freep(&formats);
    }

    av_assert0(input_dt == DNN_FLOAT);

    ctx->sws_rgb_scale = sws_getContext(inlink->w, inlink->h, fmt,
                                            ctx->dnndata->dnn_input->width, ctx->dnndata->dnn_input->height, AV_PIX_FMT_RGB24,
                                            SWS_BILINEAR, NULL, NULL, NULL);

    ctx->sws_gray8_to_grayf32 = sws_getContext(ctx->dnndata->dnn_input->width*3,
                                                ctx->dnndata->dnn_input->height,
                                                AV_PIX_FMT_GRAY8,
                                                ctx->dnndata->dnn_input->width*3,
                                                ctx->dnndata->dnn_input->height,
                                                AV_PIX_FMT_GRAYF32,
                                                0, NULL, NULL, NULL);  

    if(ctx->sws_rgb_scale == 0 || ctx->sws_gray8_to_grayf32 == 0)
    {
        av_log(ctx, AV_LOG_ERROR, "could not create scale context\n");
        return AVERROR(ENOMEM);
    }

    ctx->swscaleframe = av_frame_alloc();
    if (!ctx->swscaleframe)
        return AVERROR(ENOMEM);

    ctx->swscaleframe->format = AV_PIX_FMT_RGB24;
    ctx->swscaleframe->width  = ctx->dnndata->dnn_input->width;
    ctx->swscaleframe->height = ctx->dnndata->dnn_input->height;

    result = av_frame_get_buffer(ctx->swscaleframe, 0);
    if (result < 0) {
        av_frame_free(&ctx->swscaleframe);
        return result;
    }

    if (inlink->hw_frames_ctx) {
        ctx->swframeforHW = av_frame_alloc();

        if (!ctx->swframeforHW)
            return AVERROR(ENOMEM);
    }

    return 0;
}
static int config_input(AVFilterLink *inlink)
{
    AVFilterContext *context = inlink->dst;
    LVPDnnContext *ctx = context->priv;
    int check;    
    check = prepare_sws_context(inlink);
    if (check != 0) {
        av_log(ctx, AV_LOG_ERROR, "could not create scale context for the model\n");
        return AVERROR(EIO);
    }
    return 0;
}

static int copy_from_frame_to_dnn(LVPDnnContext *ctx, const AVFrame *frame)
{
    int bytewidth;
    av_assert0(ctx->swscaleframe != 0);
    bytewidth = av_image_get_linesize(ctx->swscaleframe->format, ctx->swscaleframe->width, 0);
    DNNData *dnn_input = ctx->dnndata->dnn_input;

    if(ctx->swframeforHW)
    {
        if(av_hwframe_transfer_data(ctx->swframeforHW, frame, 0) != 0)
            return AVERROR(EIO);
        
        sws_scale(ctx->sws_rgb_scale, (const uint8_t **)ctx->swframeforHW->data, ctx->swframeforHW->linesize,
                  0, ctx->swframeforHW->height, (uint8_t * const*)(&ctx->swscaleframe->data),
                 ctx->swscaleframe->linesize);

    }
    else
    {
        sws_scale(ctx->sws_rgb_scale, (const uint8_t **)frame->data, frame->linesize,
                  0, frame->height, (uint8_t * const*)(&ctx->swscaleframe->data),
                  ctx->swscaleframe->linesize);
    }


    if (dnn_input->dt == DNN_FLOAT) {
        sws_scale(ctx->sws_gray8_to_grayf32, (const uint8_t **)ctx->swscaleframe->data, ctx->swscaleframe->linesize,
                    0, ctx->swscaleframe->height, (uint8_t * const*)(&ctx->dnndata->dnn_input->data),
                    (const int [4]){ctx->swscaleframe->width * 3 * sizeof(float), 0, 0, 0});
    } else {
        av_assert0(dnn_input->dt == DNN_UINT8);
        av_image_copy_plane(dnn_input->data, bytewidth,
                            ctx->swscaleframe->data[0], ctx->swscaleframe->linesize[0],
                            bytewidth, ctx->swscaleframe->height);
    }
    
    return 0;   
}
static int filter_frame(AVFilterLink *inlink, AVFrame *in)
{
    char slvpinfo[256] = {0,};
    char tokeninfo[64] = {0,};
    AVFilterContext *context  = inlink->dst;
    AVFilterLink *outlink = context->outputs[0];
    LVPDnnContext *ctx = context->priv;
    DNNReturnType dnn_result;
    AVDictionary **metadata = &in->metadata;
    DNNData *dnn_output;
    int i;

    ctx->framenum ++;

    if(ctx->sample_rate > 0 && ctx->framenum % ctx->sample_rate == 0) {
        copy_from_frame_to_dnn(ctx, in);

        dnn_result = (ctx->dnndata->dnn_module->execute_model)(ctx->dnndata->dnn_model, ctx->dnndata->dnn_output, 1);
        if (dnn_result != DNN_SUCCESS) {
            av_log(ctx, AV_LOG_ERROR, "failed to execute model\n");
            av_frame_free(&in);
            return AVERROR(EIO);
        }

        dnn_output = ctx->dnndata->dnn_output;
        float* pfdata = dnn_output->data;
        int lendata = ctx->dnndata->dnn_output->height;

        // need all inference probability as metadata
        for(i=0; i<lendata; i++) {
            snprintf(tokeninfo, sizeof(tokeninfo), "%.2f,", pfdata[i]);  
            strcat(slvpinfo,tokeninfo);
        }
        if(lendata > 0) {
            av_dict_set(metadata, "lavfi.lvpdnn.text", slvpinfo, 0);
            if(ctx->logfile != NULL) {
               	fprintf(ctx->logfile,"%s\n",slvpinfo);
            }
        }
    }

    return ff_filter_frame(outlink, in);
}
static av_cold void uninit(AVFilterContext *ctx)
{
    LVPDnnContext *context = ctx->priv;

    sws_freeContext(context->sws_rgb_scale);
    sws_freeContext(context->sws_gray8_to_grayf32);
    sws_freeContext(context->sws_grayf32_to_gray8);

    if(context->swscaleframe)
        av_frame_free(&context->swscaleframe);

    if(context->swframeforHW)
        av_frame_free(&context->swframeforHW);

    if(context->log_filename && context->logfile)
    {
        fclose(context->logfile);
    }
}
static int create_dnnmodel(LVPDnnLoadData** models, char *modelpath, char *input, char *output, int deviceid)
{
    int result = 0;
    LVPDnnLoadData* dnndata;

    if(models[deviceid]) {
        av_log(NULL, AV_LOG_WARNING, "model data already created before\n");
        return result;
    }

    dnndata = (LVPDnnLoadData*)av_mallocz(sizeof(LVPDnnLoadData));
    if(!dnndata) {
        av_log(NULL, AV_LOG_ERROR, "could not create LVPDnnLoadData buffer\n");
        return AVERROR(EIO);
    }

    models[deviceid] = dnndata;
    //copy parameters, will use to compare filter argument
    dnndata->model_filename = (char*)av_mallocz(MAX_STRING_SIZE);
    dnndata->model_inputname = (char*)av_mallocz(MAX_STRING_SIZE);
    dnndata->model_outputname = (char*)av_mallocz(MAX_STRING_SIZE);
    if (!dnndata->model_filename || !dnndata->model_inputname || !dnndata->model_outputname) {
        av_log(NULL, AV_LOG_ERROR, "could not create parameter string\n");
        return AVERROR(ENOMEM);
    }

    strcpy(dnndata->model_filename, modelpath);
    strcpy(dnndata->model_inputname, input);
    strcpy(dnndata->model_outputname, output);
    //init dnn model
    dnndata->backend_type = DNN_TF;
    dnndata->dnn_module = ff_get_dnn_module(dnndata->backend_type);
    if (!dnndata->dnn_module) {
        av_log(NULL, AV_LOG_ERROR, "could not create DNN module for requested backend\n");
        return AVERROR(ENOMEM);
    }
    //setting device id for model loading.
    if (dnndata->dnn_module->set_deviceid) {
        dnndata->dnn_module->set_deviceid(deviceid);
    }
    if (!dnndata->dnn_module->load_model) {
        av_log(NULL, AV_LOG_ERROR, "load_model for network is not specified\n");
        return AVERROR(EINVAL);
    }

    dnndata->dnn_model = (dnndata->dnn_module->load_model)(dnndata->model_filename);
    if (!dnndata->dnn_model) {
        av_log(NULL, AV_LOG_ERROR, "could not load DNN model\n");
        return AVERROR(EINVAL);
    }
    //init DNN input & output data
    dnndata->dnn_input = (DNNData*)av_mallocz(sizeof(DNNData));
    if (!dnndata->dnn_input) {
        av_log(NULL, AV_LOG_ERROR, "could not create DNN input buffer\n");
        return AVERROR(EINVAL);
    }
    dnndata->dnn_output = (DNNData*)av_mallocz(sizeof(DNNData));
    if (!dnndata->dnn_output) {
        av_log(NULL, AV_LOG_ERROR, "could not create DNN output buffer\n");
        return AVERROR(EINVAL);
    }
    result = dnndata->dnn_model->get_input(dnndata->dnn_model->model, dnndata->dnn_input, dnndata->model_inputname);
    if (result != DNN_SUCCESS) {
        av_log(NULL, AV_LOG_ERROR, "could not get input from the model\n");
        return AVERROR(EIO);
    }
    result = (dnndata->dnn_model->set_input_output)(dnndata->dnn_model->model,
                                        dnndata->dnn_input, dnndata->model_inputname,
                                        (const char **)&dnndata->model_outputname, 1);
    if (result != DNN_SUCCESS) {
        av_log(NULL, AV_LOG_ERROR, "could not set input and output for the model\n");
        return AVERROR(EIO);
    }
    // have a try run in case that the dnn model resize the frame
    result = (dnndata->dnn_module->execute_model)(dnndata->dnn_model, dnndata->dnn_output, 1);
    if (result != DNN_SUCCESS) {
        av_log(NULL, AV_LOG_ERROR, "failed to execute model\n");
        return AVERROR(EIO);
    }

    return result;
}
static void free_dnnmodel(LVPDnnLoadData** models, int deviceid)
{
    LVPDnnLoadData* dnndata;
    dnndata = models[deviceid];
    if(!dnndata) return;

    //free string buffer
    av_free(dnndata->model_filename);
    av_free(dnndata->model_inputname);
    av_free(dnndata->model_outputname);

    //free DNN input & output data
    if (dnndata->dnn_input)
        av_freep(&dnndata->dnn_input);
    if (dnndata->dnn_output)
        av_freep(&dnndata->dnn_output);

    if (dnndata->dnn_module)
        (dnndata->dnn_module->free_model)(&dnndata->dnn_model);

    av_freep(&dnndata->dnn_module);
}
int avfilter_register_lvpdnn(char *modelpath, char *input, char *output, char *deviceids)
{
    int ret = 0;
    char gpuids[64] = {0,};
    char *token = NULL;
    int index, i = 0;

    //check arguments
    if(strlen(modelpath) <= 0 || strlen(input) <= 0 ||
        strlen(output) <= 0 || strlen(deviceids) <= 0) {
        av_log(NULL, AV_LOG_ERROR, "include invalid parameter\n");
        return -1;
    }
    //for using strtok function safely
    strcpy(gpuids,deviceids);

    loadedmodels = (LVPDnnLoadData**)av_mallocz_array(MAX_DEVICE_SIZE, sizeof(LVPDnnLoadData*));
    if(loadedmodels == 0) {
        av_log(NULL, AV_LOG_ERROR, "could not create model data array\n");
        return AVERROR(EIO);
    }

    if(strlen(gpuids) > 0) {
        token = strtok(gpuids, ",");
        while( token != NULL && i < MAX_DEVICE_SIZE) {
            index = atoi(token);
            if(index < 0 || index >= MAX_DEVICE_SIZE) continue;
            ret = create_dnnmodel(loadedmodels, modelpath, input, output, index);
            if(ret != 0) {
                av_log(NULL, AV_LOG_ERROR, "could not create model\n");
                break;
            }
            token = strtok(NULL, ",");
            i++;
        }
    }
    //if fail to create & init the free all models
    if(ret != 0) {
        avfilter_remove_lvpdnn();
    }
    return ret;
}
void avfilter_remove_lvpdnn()
{
    int i;
    for(i=MAX_DEVICE_SIZE -1; i>=0; i--) {
        if(loadedmodels[i]) {
            free_dnnmodel(loadedmodels, i);
            av_free(loadedmodels[i]);
            loadedmodels[i] = 0;
        }
    }
    if(loadedmodels) {
        av_freep(&loadedmodels);
        loadedmodels = NULL;
    }
    return;
}

static const AVFilterPad lvpdnn_inputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_VIDEO,
        .config_props = config_input,
        .filter_frame = filter_frame,
    },
    { NULL }
};

static const AVFilterPad lvpdnn_outputs[] = {
    {
        .name = "default",
        .type = AVMEDIA_TYPE_VIDEO,
    },
    { NULL }
};

AVFilter ff_vf_lvpdnn = {
    .name          = "lvpdnn",
    .description   = NULL_IF_CONFIG_SMALL("Apply lvpdnn filter to the input."),
    .priv_size     = sizeof(LVPDnnContext),
    .init          = init,
    .uninit        = uninit,
    .query_formats = query_formats,
    .inputs        = lvpdnn_inputs,
    .outputs       = lvpdnn_outputs,
    .priv_class    = &lvpdnn_class,
};
