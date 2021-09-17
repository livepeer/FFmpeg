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


static const enum AVPixelFormat supported_formats[] = {
    AV_PIX_FMT_YUV420P,
    AV_PIX_FMT_NV12,
    AV_PIX_FMT_YUV444P,
    AV_PIX_FMT_YUV444P16,
    AV_PIX_FMT_YUV422P,
};

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
    int mode;
    int nb_inputs;
    char *filename;
    int format;
    int thworddist;
    int thcomposdist;
    int thl1;
    int thdi;
    int thit;

    uint8_t l1distlut[243*242/2]; /* 243 + 242 + 241 ... */
    StreamContext* streamcontexts;

    AVCUDADeviceContext *hwctx;
    AVBufferRef *frames_ctx;
    AVFrame     *frame;
    
    int passthrough;
    
    CUmodule    cu_module;
    CUfunction  cu_func_int;
    CUstream    cu_stream;
    
    CUdeviceptr boxgpubuff;
    int64_t     *boxcpubuff;

    float param;
} CUDASignContext;

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
    
    ret = CHECK_CU(cu->cuModuleGetFunction(&s->cu_func_int, s->cu_module, "Subsample_Boxsumint64"));
    if (ret < 0)
        goto fail;

    ret = CHECK_CU(cu->cuMemAlloc(&s->boxgpubuff, PIXELS_SIGN * sizeof(int64_t)));
    if (ret < 0)
        return ret;

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

static int call_resize_kernel(AVFilterContext *ctx, CUfunction func, int channels,
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

static int scalecuda_resize(AVFilterContext *ctx, AVFrame *in)
{
    CUDASignContext *s = ctx->priv;
    CudaFunctions *cu = s->hwctx->internal->cuda_dl;
    int ret;

    CUDA_MEMCPY2D cpy = { 0 };

    call_resize_kernel(ctx, s->cu_func_int, 1,
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


static int get_block_size(const Block *b)
{
    return (b->to.y - b->up.y + 1) * (b->to.x - b->up.x + 1);
}

static uint64_t get_block_sum(StreamContext *sc, uint64_t intpic[32][32], const Block *b)
{
    uint64_t sum = 0;

    int x0, y0, x1, y1;

    x0 = b->up.x;
    y0 = b->up.y;
    x1 = b->to.x;
    y1 = b->to.y;

    if (x0-1 >= 0 && y0-1 >= 0) {
        sum = intpic[y1][x1] + intpic[y0-1][x0-1] - intpic[y1][x0-1] - intpic[y0-1][x1];
    } else if (x0-1 >= 0) {
        sum = intpic[y1][x1] - intpic[y1][x0-1];
    } else if (y0-1 >= 0) {
        sum = intpic[y1][x1] - intpic[y0-1][x1];
    } else {
        sum = intpic[y1][x1];
    }
    return sum;
}

static int cmp(const void *x, const void *y)
{
    const uint64_t *a = x, *b = y;
    return *a < *b ? -1 : ( *a > *b ? 1 : 0 );
}

static void set_bit(uint8_t* data, size_t pos)
{
    uint8_t mask = 1 << 7-(pos%8);
    data[pos/8] |= mask;
}

static int xml_export(AVFilterContext *ctx, StreamContext *sc, const char* filename)
{
    FineSignature* fs;
    CoarseSignature* cs;
    int i, j;
    FILE* f;
    unsigned int pot3[5] = { 3*3*3*3, 3*3*3, 3*3, 3, 1 };

    f = fopen(filename, "w");
    if (!f) {
        int err = AVERROR(EINVAL);
        char buf[128];
        av_strerror(err, buf, sizeof(buf));
        av_log(ctx, AV_LOG_ERROR, "cannot open xml file %s: %s\n", filename, buf);
        return err;
    }

    /* header */
    fprintf(f, "<?xml version='1.0' encoding='ASCII' ?>\n");
    fprintf(f, "<Mpeg7 xmlns=\"urn:mpeg:mpeg7:schema:2001\" xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" xsi:schemaLocation=\"urn:mpeg:mpeg7:schema:2001 schema/Mpeg7-2001.xsd\">\n");
    fprintf(f, "  <DescriptionUnit xsi:type=\"DescriptorCollectionType\">\n");
    fprintf(f, "    <Descriptor xsi:type=\"VideoSignatureType\">\n");
    fprintf(f, "      <VideoSignatureRegion>\n");
    fprintf(f, "        <VideoSignatureSpatialRegion>\n");
    fprintf(f, "          <Pixel>0 0 </Pixel>\n");
    fprintf(f, "          <Pixel>%d %d </Pixel>\n", sc->w - 1, sc->h - 1);
    fprintf(f, "        </VideoSignatureSpatialRegion>\n");
    fprintf(f, "        <StartFrameOfSpatialRegion>0</StartFrameOfSpatialRegion>\n");
    /* hoping num is 1, other values are vague */
    fprintf(f, "        <MediaTimeUnit>%d</MediaTimeUnit>\n", sc->time_base.den / sc->time_base.num);
    fprintf(f, "        <MediaTimeOfSpatialRegion>\n");
    fprintf(f, "          <StartMediaTimeOfSpatialRegion>0</StartMediaTimeOfSpatialRegion>\n");
    fprintf(f, "          <EndMediaTimeOfSpatialRegion>%" PRIu64 "</EndMediaTimeOfSpatialRegion>\n", sc->coarseend->last->pts);
    fprintf(f, "        </MediaTimeOfSpatialRegion>\n");

    /* coarsesignatures */
    for (cs = sc->coarsesiglist; cs; cs = cs->next) {
        fprintf(f, "        <VSVideoSegment>\n");
        fprintf(f, "          <StartFrameOfSegment>%" PRIu32 "</StartFrameOfSegment>\n", cs->first->index);
        fprintf(f, "          <EndFrameOfSegment>%" PRIu32 "</EndFrameOfSegment>\n", cs->last->index);
        fprintf(f, "          <MediaTimeOfSegment>\n");
        fprintf(f, "            <StartMediaTimeOfSegment>%" PRIu64 "</StartMediaTimeOfSegment>\n", cs->first->pts);
        fprintf(f, "            <EndMediaTimeOfSegment>%" PRIu64 "</EndMediaTimeOfSegment>\n", cs->last->pts);
        fprintf(f, "          </MediaTimeOfSegment>\n");
        for (i = 0; i < 5; i++) {
            fprintf(f, "          <BagOfWords>");
            for (j = 0; j < 31; j++) {
                uint8_t n = cs->data[i][j];
                if (j < 30) {
                    fprintf(f, "%d  %d  %d  %d  %d  %d  %d  %d  ", (n & 0x80) >> 7,
                                                                   (n & 0x40) >> 6,
                                                                   (n & 0x20) >> 5,
                                                                   (n & 0x10) >> 4,
                                                                   (n & 0x08) >> 3,
                                                                   (n & 0x04) >> 2,
                                                                   (n & 0x02) >> 1,
                                                                   (n & 0x01));
                } else {
                    /* print only 3 bit in last byte */
                    fprintf(f, "%d  %d  %d ", (n & 0x80) >> 7,
                                              (n & 0x40) >> 6,
                                              (n & 0x20) >> 5);
                }
            }
            fprintf(f, "</BagOfWords>\n");
        }
        fprintf(f, "        </VSVideoSegment>\n");
    }

    /* finesignatures */
    for (fs = sc->finesiglist; fs; fs = fs->next) {
        fprintf(f, "        <VideoFrame>\n");
        fprintf(f, "          <MediaTimeOfFrame>%" PRIu64 "</MediaTimeOfFrame>\n", fs->pts);
        /* confidence */
        fprintf(f, "          <FrameConfidence>%d</FrameConfidence>\n", fs->confidence);
        /* words */
        fprintf(f, "          <Word>");
        for (i = 0; i < 5; i++) {
            fprintf(f, "%d ", fs->words[i]);
            if (i < 4) {
                fprintf(f, " ");
            }
        }
        fprintf(f, "</Word>\n");
        /* framesignature */
        fprintf(f, "          <FrameSignature>");
        for (i = 0; i< SIGELEM_SIZE/5; i++) {
            if (i > 0) {
                fprintf(f, " ");
            }
            fprintf(f, "%d ", fs->framesig[i] / pot3[0]);
            for (j = 1; j < 5; j++)
                fprintf(f, " %d ", fs->framesig[i] % pot3[j-1] / pot3[j] );
        }
        fprintf(f, "</FrameSignature>\n");
        fprintf(f, "        </VideoFrame>\n");
    }
    fprintf(f, "      </VideoSignatureRegion>\n");
    fprintf(f, "    </Descriptor>\n");
    fprintf(f, "  </DescriptionUnit>\n");
    fprintf(f, "</Mpeg7>\n");

    fclose(f);
    return 0;
}

static int binary_export(AVFilterContext *ctx, StreamContext *sc, const char* filename)
{
    FILE* f;
    FineSignature* fs;
    CoarseSignature* cs;
    uint32_t numofsegments = (sc->lastindex + 44)/45;
    int i, j;
    PutBitContext buf;
    /* buffer + header + coarsesignatures + finesignature */
    int len = (512 + 6 * 32 + 3*16 + 2 +
        numofsegments * (4*32 + 1 + 5*243) +
        sc->lastindex * (2 + 32 + 6*8 + 608)) / 8;
    uint8_t* buffer = av_malloc_array(len, sizeof(uint8_t));
    if (!buffer)
        return AVERROR(ENOMEM);

    f = fopen(filename, "wb");
    if (!f) {
        int err = AVERROR(EINVAL);
        char buf[128];
        av_strerror(err, buf, sizeof(buf));
        av_log(ctx, AV_LOG_ERROR, "cannot open file %s: %s\n", filename, buf);
        av_freep(&buffer);
        return err;
    }
    init_put_bits(&buf, buffer, len);

    put_bits32(&buf, 1); /* NumOfSpatial Regions, only 1 supported */
    put_bits(&buf, 1, 1); /* SpatialLocationFlag, always the whole image */
    put_bits32(&buf, 0); /* PixelX,1 PixelY,1, 0,0 */
    put_bits(&buf, 16, sc->w-1 & 0xFFFF); /* PixelX,2 */
    put_bits(&buf, 16, sc->h-1 & 0xFFFF); /* PixelY,2 */
    put_bits32(&buf, 0); /* StartFrameOfSpatialRegion */
    put_bits32(&buf, sc->lastindex); /* NumOfFrames */
    /* hoping num is 1, other values are vague */
    /* den/num might be greater than 16 bit, so cutting it */
    put_bits(&buf, 16, 0xFFFF & (sc->time_base.den / sc->time_base.num)); /* MediaTimeUnit */
    put_bits(&buf, 1, 1); /* MediaTimeFlagOfSpatialRegion */
    put_bits32(&buf, 0); /* StartMediaTimeOfSpatialRegion */
    put_bits32(&buf, 0xFFFFFFFF & sc->coarseend->last->pts); /* EndMediaTimeOfSpatialRegion */
    put_bits32(&buf, numofsegments); /* NumOfSegments */
    /* coarsesignatures */
    for (cs = sc->coarsesiglist; cs; cs = cs->next) {
        put_bits32(&buf, cs->first->index); /* StartFrameOfSegment */
        put_bits32(&buf, cs->last->index); /* EndFrameOfSegment */
        put_bits(&buf, 1, 1); /* MediaTimeFlagOfSegment */
        put_bits32(&buf, 0xFFFFFFFF & cs->first->pts); /* StartMediaTimeOfSegment */
        put_bits32(&buf, 0xFFFFFFFF & cs->last->pts); /* EndMediaTimeOfSegment */
        for (i = 0; i < 5; i++) {
            /* put 243 bits ( = 7 * 32 + 19 = 8 * 28 + 19) into buffer */
            for (j = 0; j < 30; j++) {
                put_bits(&buf, 8, cs->data[i][j]);
            }
            put_bits(&buf, 3, cs->data[i][30] >> 5);
        }
    }
    /* finesignatures */
    put_bits(&buf, 1, 0); /* CompressionFlag, only 0 supported */
    for (fs = sc->finesiglist; fs; fs = fs->next) {
        put_bits(&buf, 1, 1); /* MediaTimeFlagOfFrame */
        put_bits32(&buf, 0xFFFFFFFF & fs->pts); /* MediaTimeOfFrame */
        put_bits(&buf, 8, fs->confidence); /* FrameConfidence */
        for (i = 0; i < 5; i++) {
            put_bits(&buf, 8, fs->words[i]); /* Words */
        }
        /* framesignature */
        for (i = 0; i < SIGELEM_SIZE/5; i++) {
            put_bits(&buf, 8, fs->framesig[i]);
        }
    }

    flush_put_bits(&buf);
    fwrite(buffer, 1, put_bits_count(&buf)/8, f);
    fclose(f);
    av_freep(&buffer);
    return 0;
}

static int export(AVFilterContext *ctx, StreamContext *sc, int input)
{
    CUDASignContext* sic = ctx->priv;
    char filename[1024];

    if (sic->nb_inputs > 1) {
        /* error already handled */
        av_assert0(av_get_frame_filename(filename, sizeof(filename), sic->filename, input) == 0);
    } else {
        if (av_strlcpy(filename, sic->filename, sizeof(filename)) >= sizeof(filename))
            return AVERROR(EINVAL);
    }
    if (sic->format == FORMAT_XML) {
        return xml_export(ctx, sc, filename);
    } else {
        return binary_export(ctx, sc, filename);
    }
}
static int cudasign_filter_frame(AVFilterLink *link, AVFrame *in)
{
    AVFilterContext         *ctx = link->dst;
    CUDASignContext         *s = ctx->priv;
    AVFilterLink            *outlink = ctx->outputs[0];
    CudaFunctions           *cu = s->hwctx->internal->cuda_dl;
    StreamContext           *sc = s->streamcontexts;
    FineSignature* fs;

    static const uint8_t pot3[5] = { 3*3*3*3, 3*3*3, 3*3, 3, 1 };
    /* indexes of words : 210,217,219,274,334  44,175,233,270,273  57,70,103,237,269  100,285,295,337,354  101,102,111,275,296
    s2usw = sorted to unsorted wordvec: 44 is at index 5, 57 at index 10...
    */
    static const unsigned int wordvec[25] = {44,57,70,100,101,102,103,111,175,210,217,219,233,237,269,270,273,274,275,285,295,296,334,337,354};
    static const uint8_t      s2usw[25]   = { 5,10,11, 15, 20, 21, 12, 22,  6,  0,  1,  2,  7, 13, 14,  8,  9,  3, 23, 16, 17, 24,  4, 18, 19};

    uint8_t wordt2b[5] = { 0, 0, 0, 0, 0 }; /* word ternary to binary */
    uint64_t intpic[32][32];
    uint64_t rowcount;
    uint64_t conflist[DIFFELEM_SIZE];
    int f = 0, g = 0, w = 0;
    int32_t dh1 = 1, dh2 = 1, dw1 = 1, dw2 = 1, a, b;
    int64_t denom;
    int i, j, k, ternary;
    uint64_t blocksum;
    int blocksize;
    int64_t th; /* threshold */
    int64_t sum;
    int64_t *pp;

    int64_t precfactor;
    
    CUcontext dummy;
    int ret = 0;

    ret = CHECK_CU(cu->cuCtxPushCurrent(s->hwctx->cuda_ctx));
    if (ret < 0)
        goto fail;
    //run box filter
    ret = scalecuda_resize(ctx, in);

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

    for (i = 0; i < ELEMENT_COUNT; i++) {
        const ElemCat* elemcat = elements[i];
        int64_t* elemsignature;
        uint64_t* sortsignature;

        elemsignature = av_malloc_array(elemcat->elem_count, sizeof(int64_t));
        if (!elemsignature)
            return AVERROR(ENOMEM);
        sortsignature = av_malloc_array(elemcat->elem_count, sizeof(int64_t));
        if (!sortsignature) {
            av_freep(&elemsignature);
            return AVERROR(ENOMEM);
        }

        for (j = 0; j < elemcat->elem_count; j++) {
            blocksum = 0;
            blocksize = 0;
            for (k = 0; k < elemcat->left_count; k++) {
                blocksum += get_block_sum(sc, intpic, &elemcat->blocks[j*elemcat->block_count+k]);
                blocksize += get_block_size(&elemcat->blocks[j*elemcat->block_count+k]);
            }
            sum = blocksum / blocksize;
            if (elemcat->av_elem) {
                sum -= 128 * precfactor * denom;
            } else {
                blocksum = 0;
                blocksize = 0;
                for (; k < elemcat->block_count; k++) {
                    blocksum += get_block_sum(sc, intpic, &elemcat->blocks[j*elemcat->block_count+k]);
                    blocksize += get_block_size(&elemcat->blocks[j*elemcat->block_count+k]);
                }
                sum -= blocksum / blocksize;
                conflist[g++] = FFABS(sum * 8 / (precfactor * denom));
            }

            elemsignature[j] = sum;
            sortsignature[j] = FFABS(sum);
        }

        /* get threshold */
        qsort(sortsignature, elemcat->elem_count, sizeof(uint64_t), cmp);
        th = sortsignature[(int) (elemcat->elem_count*0.333)];

        /* ternarize */
        for (j = 0; j < elemcat->elem_count; j++) {
            if (elemsignature[j] < -th) {
                ternary = 0;
            } else if (elemsignature[j] <= th) {
                ternary = 1;
            } else {
                ternary = 2;
            }
            fs->framesig[f/5] += ternary * pot3[f%5];

            if (f == wordvec[w]) {
                fs->words[s2usw[w]/5] += ternary * pot3[wordt2b[s2usw[w]/5]++];
                if (w < 24)
                    w++;
            }
            f++;
        }
        av_freep(&elemsignature);
        av_freep(&sortsignature);
    }

    /* confidence */
    qsort(conflist, DIFFELEM_SIZE, sizeof(uint64_t), cmp);
    fs->confidence = FFMIN(conflist[DIFFELEM_SIZE/2], 255);

    /* coarsesignature */
    if (sc->coarsecount == 0) {
        if (sc->curcoarsesig2) {
            sc->curcoarsesig1 = av_mallocz(sizeof(CoarseSignature));
            if (!sc->curcoarsesig1)
                return AVERROR(ENOMEM);
            sc->curcoarsesig1->first = fs;
            sc->curcoarsesig2->next = sc->curcoarsesig1;
            sc->coarseend = sc->curcoarsesig1;
        }
    }
    if (sc->coarsecount == 45) {
        sc->midcoarse = 1;
        sc->curcoarsesig2 = av_mallocz(sizeof(CoarseSignature));
        if (!sc->curcoarsesig2)
            return AVERROR(ENOMEM);
        sc->curcoarsesig2->first = fs;
        sc->curcoarsesig1->next = sc->curcoarsesig2;
        sc->coarseend = sc->curcoarsesig2;
    }
    for (i = 0; i < 5; i++) {
        set_bit(sc->curcoarsesig1->data[i], fs->words[i]);
    }
    /* assuming the actual frame is the last */
    sc->curcoarsesig1->last = fs;
    if (sc->midcoarse) {
        for (i = 0; i < 5; i++) {
            set_bit(sc->curcoarsesig2->data[i], fs->words[i]);
        }
        sc->curcoarsesig2->last = fs;
    }

    sc->coarsecount = (sc->coarsecount+1)%90;

    //passthrough
    return ff_filter_frame(outlink, in);
fail:
    av_frame_free(&in);    
    return ret;
}

static av_cold int cudasign_init(AVFilterContext *ctx)
{
    CUDASignContext *s = ctx->priv;
    StreamContext *sc;

    s->boxcpubuff = (int64_t*)av_malloc_array(PIXELS_SIGN, sizeof(int64_t));
    if (!s->boxcpubuff)
        return AVERROR(ENOMEM);

    s->streamcontexts = av_mallocz(s->nb_inputs * sizeof(StreamContext));
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

#define OFFSET(x) offsetof(CUDASignContext, x)
#define FLAGS (AV_OPT_FLAG_FILTERING_PARAM|AV_OPT_FLAG_VIDEO_PARAM)
static const AVOption options[] = {
    { "detectmode", "set the detectmode",
        OFFSET(mode),         AV_OPT_TYPE_INT,    {.i64 = MODE_OFF}, 0, NB_LOOKUP_MODE-1, FLAGS, "mode" },
        { "off",  NULL, 0, AV_OPT_TYPE_CONST, {.i64 = MODE_OFF},  0, 0, .flags = FLAGS, "mode" },
        { "full", NULL, 0, AV_OPT_TYPE_CONST, {.i64 = MODE_FULL}, 0, 0, .flags = FLAGS, "mode" },
        { "fast", NULL, 0, AV_OPT_TYPE_CONST, {.i64 = MODE_FAST}, 0, 0, .flags = FLAGS, "mode" },
    { "nb_inputs",  "number of inputs",
        OFFSET(nb_inputs),    AV_OPT_TYPE_INT,    {.i64 = 1},        1, INT_MAX,          FLAGS },
    { "filename",   "filename for output files",
        OFFSET(filename),     AV_OPT_TYPE_STRING, {.str = ""},       0, NB_FORMATS-1,     FLAGS },
    { "format",     "set output format",
        OFFSET(format),       AV_OPT_TYPE_INT,    {.i64 = FORMAT_BINARY}, 0, 1,           FLAGS , "format" },
        { "binary", 0, 0, AV_OPT_TYPE_CONST, {.i64=FORMAT_BINARY}, 0, 0, FLAGS, "format" },
        { "xml",    0, 0, AV_OPT_TYPE_CONST, {.i64=FORMAT_XML},    0, 0, FLAGS, "format" },
    { "th_d",       "threshold to detect one word as similar",
        OFFSET(thworddist),   AV_OPT_TYPE_INT,    {.i64 = 9000},     1, INT_MAX,          FLAGS },
    { "th_dc",      "threshold to detect all words as similar",
        OFFSET(thcomposdist), AV_OPT_TYPE_INT,    {.i64 = 60000},    1, INT_MAX,          FLAGS },
    { "th_xh",      "threshold to detect frames as similar",
        OFFSET(thl1),         AV_OPT_TYPE_INT,    {.i64 = 116},      1, INT_MAX,          FLAGS },
    { "th_di",      "minimum length of matching sequence in frames",
        OFFSET(thdi),         AV_OPT_TYPE_INT,    {.i64 = 0},        0, INT_MAX,          FLAGS },
    { "th_it",      "threshold for relation of good to all frames",
        OFFSET(thit),         AV_OPT_TYPE_DOUBLE, {.dbl = 0.5},    0.0, 1.0,              FLAGS },
    { NULL },
};

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