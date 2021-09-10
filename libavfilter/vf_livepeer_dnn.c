/*
 * Copyright (c) 2021 Livepeer Inc.
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
 * Filter implementing livepeer scene classification.
 */

#include "avfilter.h"
#include "formats.h"
#include "internal.h"
#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"
#include "libavutil/imgutils.h"
#include "libavformat/avio.h"
#include "libswscale/swscale.h"
#include "dnn_filter_common.h"

typedef struct LivepeerContext {
    const AVClass *class;

    DnnContext dnnctx;         ///< DNN model, backend, I/O layer names
    int input_width, input_height, output_width, output_height;      ///< Model input and output dimensions, initialized after the model is loaded
    struct SwsContext *sws_rgb_scale; ///< Used for scaling image to DNN input size and pixel format (RGB24)
    struct AVFrame *swscaleframe;  ///< Scaled image
    FILE *logfile;       ///< (Optional) Log classification probabilities in this file
    char *log_filename;  ///< File name
} LivepeerContext;

#define OFFSET(x) offsetof(LivepeerContext, x)
#define FLAGS AV_OPT_FLAG_FILTERING_PARAM | AV_OPT_FLAG_VIDEO_PARAM
static const AVOption livepeer_options[] = {
    {"dnn_backend", "DNN backend used for model execution", OFFSET(dnnctx.backend_type), AV_OPT_TYPE_INT,
     {.i64 = 1}, 0, 1, FLAGS, "backend"},
    {"native", "native backend flag", 0, AV_OPT_TYPE_CONST, {.i64 = 0}, 0, 0, FLAGS, "backend"},
#if (CONFIG_LIBTENSORFLOW == 1)
    {"tensorflow", "tensorflow backend flag", 0, AV_OPT_TYPE_CONST, {.i64 = 1}, 0, 0, FLAGS, "backend"},
#endif
    {"model", "path to model file specifying network architecture and its parameters",
     OFFSET(dnnctx.model_filename), AV_OPT_TYPE_STRING, {.str=NULL}, 0, 0, FLAGS},
    {"input", "input name of the model", OFFSET(dnnctx.model_inputname), AV_OPT_TYPE_STRING, {.str = "x"}, 0, 0,
     FLAGS},
    {"output", "output name of the model", OFFSET(dnnctx.model_outputname), AV_OPT_TYPE_STRING, {.str = "y"}, 0, 0,
     FLAGS},
    // default session_config = {allow_growth: true}
    {"backend_configs", "backend configs", OFFSET(dnnctx.backend_options), AV_OPT_TYPE_STRING,
     {.str = "sess_config=0x01200232"}, 0, 0, FLAGS},
    {"logfile", "path to logfile", OFFSET(log_filename), AV_OPT_TYPE_STRING, {.str = NULL}, 0, 0, FLAGS},
    {NULL}
};

AVFILTER_DEFINE_CLASS(livepeer);

static int post_proc(AVFrame *out, DNNData *dnn_output, AVFilterContext *context);

static av_cold int init(AVFilterContext *context)
{
    LivepeerContext *ctx = context->priv;
    DNNData input;
    int ret = 0;

    if (ctx->log_filename) {
        ctx->logfile = fopen(ctx->log_filename, "w");
    } else {
        ctx->logfile = NULL;
        av_log(ctx, AV_LOG_INFO, "output file for log is not specified\n");
    }

    ret = ff_dnn_init(&ctx->dnnctx, DFT_PROCESS_FRAME, context);

    ctx->dnnctx.model->get_input(ctx->dnnctx.model->model, &input, ctx->dnnctx.model_inputname);
    ctx->input_width = input.width;
    ctx->input_height = input.height;
    // pre-executes the model and gets output information
    if (DNN_SUCCESS != ctx->dnnctx.model->get_output(ctx->dnnctx.model->model, ctx->dnnctx.model_inputname,
                                                     input.width, input.height,
                                                     ctx->dnnctx.model_outputname,
                                                     &ctx->output_width,
                                                     &ctx->output_height)) {
        av_log(ctx, AV_LOG_ERROR, "failed to init model\n");
        return AVERROR(EIO);
    }

    ctx->dnnctx.model->post_proc = post_proc;

    return ret;
}

static int query_formats(AVFilterContext *context)
{
    const enum AVPixelFormat pixel_formats[] = {AV_PIX_FMT_RGB24, AV_PIX_FMT_BGR24, AV_PIX_FMT_NV12,
                                                AV_PIX_FMT_YUV420P, AV_PIX_FMT_YUV422P, AV_PIX_FMT_YUV444P,
                                                AV_PIX_FMT_YUV410P, AV_PIX_FMT_YUV411P, AV_PIX_FMT_GRAY8,
                                                AV_PIX_FMT_NONE};
    AVFilterFormats *formats_list;

    formats_list = ff_make_format_list(pixel_formats);
    if (!formats_list) {
        av_log(context, AV_LOG_ERROR, "could not create formats list\n");
        return AVERROR(ENOMEM);
    }

    return ff_set_common_formats(context, formats_list);
}

static int prepare_sws_context(AVFilterLink *inlink)
{
    int result = 0;
    enum AVPixelFormat fmt = inlink->format;
    AVFilterContext *context = inlink->dst;
    LivepeerContext *ctx = context->priv;
    DNNData input;

    ctx->dnnctx.model->get_input(ctx->dnnctx.model->model, &input, ctx->dnnctx.model_inputname);

    ctx->sws_rgb_scale = sws_getContext(inlink->w, inlink->h, fmt,
                                        input.width, input.height, AV_PIX_FMT_RGB24,
                                        SWS_BILINEAR, NULL, NULL, NULL);

    if (ctx->sws_rgb_scale == 0) {
        av_log(ctx, AV_LOG_ERROR, "could not create scale context\n");
        return AVERROR(ENOMEM);
    }

    ctx->swscaleframe = av_frame_alloc();
    if (!ctx->swscaleframe)
        return AVERROR(ENOMEM);

    ctx->swscaleframe->format = AV_PIX_FMT_RGB24;
    ctx->swscaleframe->width = input.width;
    ctx->swscaleframe->height = input.height;

    result = av_frame_get_buffer(ctx->swscaleframe, 0);
    if (result < 0) {
        av_frame_free(&ctx->swscaleframe);
        return result;
    }
    return 0;
}

static int config_input(AVFilterLink *inlink)
{
    AVFilterContext *context = inlink->dst;
    LivepeerContext *ctx = context->priv;
    int check;
    check = prepare_sws_context(inlink);
    if (check != 0) {
        av_log(ctx, AV_LOG_ERROR, "could not create scale context for the model\n");
        return AVERROR(EIO);
    }
    return 0;
}

static int filter_frame(AVFilterLink *inlink, AVFrame *in)
{
    AVFilterContext *context = inlink->dst;
    LivepeerContext *ctx = context->priv;
    AVFilterLink *outlink = context->outputs[0];
    DNNReturnType dnn_result;

    AVFrame *out = ff_get_video_buffer(outlink, 64, 64);
    if (!out) {
        av_log(context, AV_LOG_ERROR, "could not allocate memory for output frame\n");
        av_frame_free(&in);
        return AVERROR(ENOMEM);
    }
    av_frame_copy_props(out, in);

    // Scale frame and transform pixel format to what DNN expects.
    sws_scale(ctx->sws_rgb_scale, (const uint8_t **) in->data, in->linesize,
              0, in->height, (uint8_t *const *) (&ctx->swscaleframe->data),
              ctx->swscaleframe->linesize);

    // Execute model.
    dnn_result = ff_dnn_execute_model(&ctx->dnnctx, ctx->swscaleframe, out);

    // Copy classification metadata to input frame (check if we can use output frame)
    av_dict_copy(&in->metadata, out->metadata, 0);

    if (dnn_result != DNN_SUCCESS) {
        av_log(ctx, AV_LOG_ERROR, "failed to execute loaded model\n");
        av_frame_free(&in);
        av_frame_free(&out);
        return AVERROR(EIO);
    }

    // Don't use model output directly
    av_frame_free(&out);
    return ff_filter_frame(outlink, in);
}

static int post_proc(AVFrame *out, DNNData *dnn_output, AVFilterContext *context)
{
    LivepeerContext *ctx = context->priv;
    float *pfdata = dnn_output->data;
    int lendata = dnn_output->height;
    char slvpinfo[256] = {0,};
    char tokeninfo[64] = {0,};
    AVDictionary **metadata = &out->metadata;

    // need all inference probability as metadata
    for (int i = 0; i < lendata; i++) {
        snprintf(tokeninfo, sizeof(tokeninfo), "%.2f,", pfdata[i]);
        strcat(slvpinfo, tokeninfo);
    }
    if (lendata > 0) {
        av_dict_set(metadata, "lavfi.lvpdnn.text", slvpinfo, 0);
        if (ctx->logfile != NULL) {
            fprintf(ctx->logfile, "%s\n", slvpinfo);
        }
    }
    return DNN_SUCCESS;
}

static av_cold void uninit(AVFilterContext *context)
{
    LivepeerContext *ctx = context->priv;

    sws_freeContext(ctx->sws_rgb_scale);

    if (ctx->swscaleframe)
        av_frame_free(&ctx->swscaleframe);

    ff_dnn_uninit(&ctx->dnnctx);
    if (ctx->log_filename && ctx->logfile) {
        fclose(ctx->logfile);
    }
}

static const AVFilterPad livepeer_inputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_VIDEO,
        .config_props = config_input,
        .filter_frame = filter_frame,
    },
    {NULL}
};

static const AVFilterPad livepeer_outputs[] = {
    {
        .name = "default",
        .type = AVMEDIA_TYPE_VIDEO,
    },
    {NULL}
};

AVFilter ff_vf_livepeer_dnn = {
    .name          = "livepeer_dnn",
    .description   = NULL_IF_CONFIG_SMALL("Perform DNN-based scene classification on input."),
    .priv_size     = sizeof(LivepeerContext),
    .init          = init,
    .uninit        = uninit,
    .query_formats = query_formats,
    .inputs        = livepeer_inputs,
    .outputs       = livepeer_outputs,
    .priv_class    = &livepeer_class,
};
