#include <float.h>
#include <stdio.h>
#include <string.h>

#include "libavutil/avstring.h"
#include "libavutil/common.h"
#include "libavutil/hwcontext.h"
#include "libavutil/hwcontext_cuda_internal.h"
#include "libavutil/cuda_check.h"
#include "libavutil/internal.h"
#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"
#include "libavcodec/put_bits.h"
#include "libavcodec/get_bits.h"
#include "libavformat/avformat.h"
#include "avfilter.h"
#include "formats.h"
#include "internal.h"
#include "scale_eval.h"
#include "video.h"
#include "signature.h"
#include "signature_lookup.c"

#define DIV_UP(a, b) ( ((a) + (b) - 1) / (b) )
#define BLOCKX 32
#define BLOCKY 16
#define W_SIGN 32
#define H_SIGN 32
#define PIXELS_SIGN 1024
#define BLOCK_LCM (int64_t) 476985600
#define CHECK_CU(x) FF_CUDA_CHECK_DL(ctx, s->hwctx->internal->cuda_dl, x)

typedef struct CUDASignContext {
    const AVClass *class;

    char *filename;
    int  format;

    StreamContext* streamcontexts;

    AVCUDADeviceContext *hwctx;
    AVBufferRef *frames_ctx;
    
    CUmodule    cu_module;
    CUfunction  cu_func_boxsum;
    CUstream    cu_stream;
    
    CUdeviceptr boxgpubuff;
    int64_t     *boxcpubuff;

    float param;
} CUDASignContext;

#define OFFSET(x) offsetof(CUDASignContext, x)
#define FLAGS (AV_OPT_FLAG_FILTERING_PARAM|AV_OPT_FLAG_VIDEO_PARAM)
static const AVOption options[] = {
    { "filename",   "filename for output file",
        OFFSET(filename), AV_OPT_TYPE_STRING, {.str = ""}, 0, NB_FORMATS-1, FLAGS },
    { "format",     "set output format",
        OFFSET(format),       AV_OPT_TYPE_INT, {.i64 = FORMAT_BINARY}, 0, 1, FLAGS , "format" },
        { "binary", 0, 0, AV_OPT_TYPE_CONST, {.i64=FORMAT_BINARY}, 0, 0, FLAGS, "format" },
        { "xml",    0, 0, AV_OPT_TYPE_CONST, {.i64=FORMAT_XML},    0, 0, FLAGS, "format" },
    { NULL },
};

static const enum AVPixelFormat supported_formats[] = {
    AV_PIX_FMT_YUV420P,
    AV_PIX_FMT_NV12,
    AV_PIX_FMT_YUV444P,
    AV_PIX_FMT_YUV444P16,
    AV_PIX_FMT_YUV422P,
};

static int cudasign_query_formats(AVFilterContext *ctx)
{
    static const enum AVPixelFormat pixel_formats[] = {
        AV_PIX_FMT_CUDA, AV_PIX_FMT_NONE,
    };
    AVFilterFormats *pix_fmts = ff_make_format_list(pixel_formats);
    if (!pix_fmts)
        return AVERROR(ENOMEM);

    return ff_set_common_formats(ctx, pix_fmts);
}

static int format_is_supported(enum AVPixelFormat fmt)
{
    int i;
    for (i = 0; i < FF_ARRAY_ELEMS(supported_formats); i++)
        if (supported_formats[i] == fmt)
            return 1;
    return 0;
}

static AVFrame *get_pass_video_buffer(AVFilterLink *inlink, int w, int h)
{
    return ff_null_get_video_buffer(inlink, w, h);
}

static av_cold int cudasign_config_props(AVFilterLink *outlink)
{
    AVFilterContext *ctx = outlink->src;
    AVFilterLink *inlink = outlink->src->inputs[0];
    CUDASignContext *s  = ctx->priv;
    AVHWFramesContext *frames_ctx = (AVHWFramesContext*)inlink->hw_frames_ctx->data;
    AVCUDADeviceContext *device_hwctx = frames_ctx->device_ctx->hwctx;
    CUcontext dummy, cuda_ctx = device_hwctx->cuda_ctx;
    CudaFunctions *cu = device_hwctx->internal->cuda_dl;

    AVHWFramesContext *in_frames_ctx;
    enum AVPixelFormat in_format;
    StreamContext *sc;

    extern char vf_signature_cuda_ptx[];
    int ret;

    s->hwctx = device_hwctx;
    s->cu_stream = s->hwctx->stream;

    ret = CHECK_CU(cu->cuCtxPushCurrent(cuda_ctx));
    if (ret < 0)
        goto fail;

    ret = CHECK_CU(cu->cuModuleLoadData(&s->cu_module, vf_signature_cuda_ptx));
    if (ret < 0)
        goto fail;
    
    ret = CHECK_CU(cu->cuModuleGetFunction(&s->cu_func_boxsum, s->cu_module, "Subsample_Boxsumint64"));
    if (ret < 0)
        goto fail;

    ret = CHECK_CU(cu->cuMemAlloc(&s->boxgpubuff, PIXELS_SIGN * sizeof(int64_t)));
    if (ret < 0)
        goto fail;

    CHECK_CU(cu->cuCtxPopCurrent(&dummy));

     if (!ctx->inputs[0]->hw_frames_ctx) {
        av_log(ctx, AV_LOG_ERROR, "No hw context provided on input\n");
        return AVERROR(EINVAL);
    }
    in_frames_ctx = (AVHWFramesContext*)ctx->inputs[0]->hw_frames_ctx->data;
    in_format     = in_frames_ctx->sw_format;    

    if (!format_is_supported(in_format)) {
        av_log(ctx, AV_LOG_ERROR, "Unsupported input format: %s\n",
               av_get_pix_fmt_name(in_format));
        return AVERROR(ENOSYS);
    }

    //for passthrough frame
    s->frames_ctx = av_buffer_ref(ctx->inputs[0]->hw_frames_ctx);
    ctx->outputs[0]->hw_frames_ctx = av_buffer_ref(s->frames_ctx);
    if (!ctx->outputs[0]->hw_frames_ctx)
        return AVERROR(ENOMEM);

    //config init
    sc = s->streamcontexts;

    sc->time_base = inlink->time_base;
    /* test for overflow */
    sc->divide = (((uint64_t) inlink->w/32) * (inlink->w/32 + 1) * (inlink->h/32 * inlink->h/32 + 1) > INT64_MAX / (BLOCK_LCM * 255));
    if (sc->divide) {
        av_log(ctx, AV_LOG_WARNING, "Input dimension too high for precise calculation, numbers will be rounded.\n");
    }
    sc->w = inlink->w;
    sc->h = inlink->h;

    return 0;

fail:
    return ret;
}

static int call_boxsum_kernel(AVFilterContext *ctx, CUfunction func, int channels,
                              uint8_t *src_dptr, int src_width, int src_height, int src_pitch,
                              uint8_t *dst_dptr, int dst_width, int dst_height, int dst_pitch,
                              int pixel_size, int bit_depth)
{
    CUDASignContext *s = ctx->priv;
    CudaFunctions *cu = s->hwctx->internal->cuda_dl;
    CUdeviceptr dst_devptr = (CUdeviceptr)dst_dptr;
    CUtexObject tex = 0;
    void *args_uchar[] = { &tex, &dst_devptr, &dst_width, &dst_height, &dst_pitch,
                           &src_width, &src_height, &bit_depth, &s->param };
    int ret;

    CUDA_TEXTURE_DESC tex_desc = {
        .filterMode =  CU_TR_FILTER_MODE_POINT,
        .flags = CU_TRSF_READ_AS_INTEGER,
    };

    CUDA_RESOURCE_DESC res_desc = {
        .resType = CU_RESOURCE_TYPE_PITCH2D,
        .res.pitch2D.format = pixel_size == 1 ?
                              CU_AD_FORMAT_UNSIGNED_INT8 :
                              CU_AD_FORMAT_UNSIGNED_INT16,
        .res.pitch2D.numChannels = channels,
        .res.pitch2D.width = src_width,
        .res.pitch2D.height = src_height,
        .res.pitch2D.pitchInBytes = src_pitch,
        .res.pitch2D.devPtr = (CUdeviceptr)src_dptr,
    };

    ret = CHECK_CU(cu->cuTexObjectCreate(&tex, &res_desc, &tex_desc, NULL));
    if (ret < 0)
        goto exit;

    ret = CHECK_CU(cu->cuLaunchKernel(func,
                                      DIV_UP(dst_width, BLOCKX), DIV_UP(dst_height, BLOCKY), 1,
                                      BLOCKX, BLOCKY, 1, 0, s->cu_stream, args_uchar, NULL));

exit:
    if (tex)
        CHECK_CU(cu->cuTexObjectDestroy(tex));

    return ret;
}

static int run_cudaresize(AVFilterContext *ctx, AVFrame *in)
{
    CUDASignContext *s = ctx->priv;
    CudaFunctions *cu = s->hwctx->internal->cuda_dl;
    int ret;

    CUDA_MEMCPY2D cpy = { 0 };

    call_boxsum_kernel(ctx, s->cu_func_boxsum, 1,
                in->data[0], in->width, in->height, in->linesize[0],
                (uint8_t*)s->boxgpubuff, W_SIGN, H_SIGN, W_SIGN,
                1, 8);
    
    cpy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    cpy.dstMemoryType = CU_MEMORYTYPE_HOST;
    cpy.srcDevice = s->boxgpubuff;
    cpy.dstHost = s->boxcpubuff;
    cpy.srcPitch = W_SIGN * sizeof(int64_t);
    cpy.dstPitch = W_SIGN * sizeof(int64_t);
    cpy.WidthInBytes = W_SIGN * sizeof(int64_t);
    cpy.Height = H_SIGN;

    ret = CHECK_CU(cu->cuMemcpy2DAsync(&cpy, s->cu_stream));

    return ret;
}

static int cudasign_filter_frame(AVFilterLink *link, AVFrame *in)
{
    AVFilterContext         *ctx = link->dst;
    CUDASignContext         *s = ctx->priv;
    AVFilterLink            *outlink = ctx->outputs[0];
    CudaFunctions           *cu = s->hwctx->internal->cuda_dl;
    StreamContext           *sc = s->streamcontexts;
    FineSignature* fs;

    uint64_t intpic[32][32];
    uint64_t rowcount;
    int i, j, ret;
    int32_t dh1 = 1, dh2 = 1, dw1 = 1, dw2 = 1, a, b;
    int64_t denom;
    int64_t *pp;
    int64_t precfactor;
    
    CUcontext dummy;

    ret = CHECK_CU(cu->cuCtxPushCurrent(s->hwctx->cuda_ctx));
    if (ret < 0)
        goto fail;
    //run box filter
    ret = run_cudaresize(ctx, in);

    CHECK_CU(cu->cuCtxPopCurrent(&dummy));
    if (ret < 0)
        goto fail;
    
    precfactor = (sc->divide) ? 65536 : BLOCK_LCM;
    
    /* initialize fs */
    if (sc->curfinesig) {
        fs = av_mallocz(sizeof(FineSignature));
        if (!fs)
            return AVERROR(ENOMEM);
        sc->curfinesig->next = fs;
        fs->prev = sc->curfinesig;
        sc->curfinesig = fs;
    } else {
        fs = sc->curfinesig = sc->finesiglist;
        sc->curcoarsesig1->first = fs;
    }
    fs->pts = in->pts;
    fs->index = sc->lastindex++;
    
    //copy box scaled buffer
    pp = s->boxcpubuff;
    
    for (i =0; i< H_SIGN; i++) {
        memcpy(intpic[i], pp, W_SIGN * sizeof(int64_t));
        pp += W_SIGN;
    }
    /* The following calculates a summed area table (intpic) and brings the numbers
     * in intpic to the same denominator.
     * So you only have to handle the numinator in the following sections.
     */
    dh1 = link->h / 32;
    if (link->h % 32)
        dh2 = dh1 + 1;
    dw1 = link->w / 32;
    if (link->w % 32)
        dw2 = dw1 + 1;
    denom = (sc->divide) ? dh1 * dh2 * dw1 * dw2 : 1;

    for (i = 0; i < 32; i++) {
        rowcount = 0;
        a = 1;
        if (dh2 > 1) {
            a = ((link->h*(i+1))%32 == 0) ? (link->h*(i+1))/32 - 1 : (link->h*(i+1))/32;
            a -= ((link->h*i)%32 == 0) ? (link->h*i)/32 - 1 : (link->h*i)/32;
            a = (a == dh1)? dh2 : dh1;
        }
        for (j = 0; j < 32; j++) {
            b = 1;
            if (dw2 > 1) {
                b = ((link->w*(j+1))%32 == 0) ? (link->w*(j+1))/32 - 1 : (link->w*(j+1))/32;
                b -= ((link->w*j)%32 == 0) ? (link->w*j)/32 - 1 : (link->w*j)/32;
                b = (b == dw1)? dw2 : dw1;
            }
            rowcount += intpic[i][j] * a * b * precfactor / denom;
            if (i > 0) {
                intpic[i][j] = intpic[i-1][j] + rowcount;
            } else {
                intpic[i][j] = rowcount;
            }
        }
    }

    denom = (sc->divide) ? 1 : dh1 * dh2 * dw1 * dw2;
    ret = calc_signature(ctx, sc, fs, intpic, denom, precfactor);
    if (ret < 0)
        goto fail;

    //passthrough
    return ff_filter_frame(outlink, in);
fail:
    av_frame_free(&in);    
    return ret;
}

static int export(AVFilterContext *ctx, StreamContext *sc, int input)
{
    CUDASignContext* sic = ctx->priv;
    char filename[1024];

    if (av_strlcpy(filename, sic->filename, sizeof(filename)) >= sizeof(filename))
        return AVERROR(EINVAL);

    if (sic->format == FORMAT_XML) {
        return xml_export(ctx, sc, filename);
    } else {
        return binary_export(ctx, sc, filename);
    }
}

static av_cold int cudasign_init(AVFilterContext *ctx)
{
    CUDASignContext *s = ctx->priv;
    StreamContext *sc;

    s->boxcpubuff = (int64_t*)av_malloc_array(PIXELS_SIGN, sizeof(int64_t));
    if (!s->boxcpubuff)
        return AVERROR(ENOMEM);

    s->streamcontexts = av_mallocz(sizeof(StreamContext));
    if (!s->streamcontexts)
        return AVERROR(ENOMEM);
    sc = s->streamcontexts;

    sc->lastindex = 0;
    sc->finesiglist = av_mallocz(sizeof(FineSignature));
    if (!sc->finesiglist)
        return AVERROR(ENOMEM);
    sc->curfinesig = NULL;

    sc->coarsesiglist = av_mallocz(sizeof(CoarseSignature));
    if (!sc->coarsesiglist)
        return AVERROR(ENOMEM);
    sc->curcoarsesig1 = sc->coarsesiglist;
    sc->coarseend = sc->coarsesiglist;
    sc->coarsecount = 0;
    sc->midcoarse = 0; 

    return 0;
}

static av_cold void cudasign_uninit(AVFilterContext *ctx)
{
    CUDASignContext *s = ctx->priv;
    StreamContext *sc;
    FineSignature* finsig;
    CoarseSignature* cousig;
    void* tmp;

    if (s->hwctx && s->cu_module) {
        CudaFunctions *cu = s->hwctx->internal->cuda_dl;
        CUcontext dummy;

        CHECK_CU(cu->cuCtxPushCurrent(s->hwctx->cuda_ctx));
        if (s->boxgpubuff) {
            CHECK_CU(cu->cuMemFree(s->boxgpubuff));
            s->boxgpubuff = 0;
        }
        CHECK_CU(cu->cuModuleUnload(s->cu_module));
        s->cu_module = NULL;
        CHECK_CU(cu->cuCtxPopCurrent(&dummy));
    }
    //free hw_frames_ctx buffer for passthrough
    av_buffer_unref(&s->frames_ctx);

    if(s->boxcpubuff)
        av_freep(&s->boxcpubuff);

    if(s->streamcontexts) {
        sc = s->streamcontexts;

        if (sc->lastindex > 0 && strlen(s->filename) > 0) {
            export(ctx, sc, 0);
            sc->exported = 1;
        }

        finsig = sc->finesiglist;
        cousig = sc->coarsesiglist;

        while (finsig) {
            tmp = finsig;
            finsig = finsig->next;
            av_freep(&tmp);
        }
        sc->finesiglist = NULL;

        while (cousig) {
            tmp = cousig;
            cousig = cousig->next;
            av_freep(&tmp);
        }
        sc->coarsesiglist = NULL;        
        av_freep(&s->streamcontexts);
    }    
}

static const AVClass cudasign_class = {
    .class_name = "cudasign",
    .item_name  = av_default_item_name,
    .option     = options,
    .version    = LIBAVUTIL_VERSION_INT,
};

static const AVFilterPad cudasign_inputs[] = {
    {
        .name        = "default",
        .type        = AVMEDIA_TYPE_VIDEO,
        .filter_frame = cudasign_filter_frame,
        .get_video_buffer = get_pass_video_buffer,
    },
    { NULL }
};

static const AVFilterPad cudasign_outputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_VIDEO,
        .config_props = cudasign_config_props,
    },
    { NULL }
};

AVFilter ff_vf_signature_cuda = {
    .name      = "signature_cuda",
    .description = NULL_IF_CONFIG_SMALL("GPU accelerated video resizer"),

    .init          = cudasign_init,
    .uninit        = cudasign_uninit,
    .query_formats = cudasign_query_formats,

    .priv_size = sizeof(CUDASignContext),
    .priv_class = &cudasign_class,

    .inputs    = cudasign_inputs,
    .outputs   = cudasign_outputs,

    .flags_internal = FF_FILTER_FLAG_HWFRAME_AWARE,
};