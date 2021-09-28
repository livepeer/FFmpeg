/*
 * Copyright (c) 2017 Gerion Entrup
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with FFmpeg; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

/**
 * @file
 * MPEG-7 video signature calculation and lookup filter
 * @see http://epubs.surrey.ac.uk/531590/1/MPEG-7%20Video%20Signature%20Author%27s%20Copy.pdf
 */

#include <float.h>
#include "libavcodec/put_bits.h"
#include "libavcodec/get_bits.h"
#include "libavformat/avformat.h"
#include "libavutil/opt.h"
#include "libavutil/avstring.h"
#include "libavutil/intreadwrite.h"
#include "libavutil/timestamp.h"
#include "avfilter.h"
#include "internal.h"
#include "signature.h"
#include "signature_lookup.c"

#define OFFSET(x) offsetof(SignatureContext, x)
#define FLAGS AV_OPT_FLAG_FILTERING_PARAM | AV_OPT_FLAG_VIDEO_PARAM
#define BLOCK_LCM (int64_t) 476985600
#define INPUTS_COUNT 2
#define MPEG7_FINESIG_NBITS 689

typedef struct BoundedCoarseSignature {
    // StartFrameOfSegment and EndFrameOfSegment
    uint32_t firstIndex, lastIndex;
    // StartMediaTimeOfSegment and EndMediaTimeOfSegment
    uint64_t firstPts, lastPts;
    CoarseSignature *cSign;
} BoundedCoarseSignature;

static const AVOption signature_options[] = {
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
    { NULL }
};

AVFILTER_DEFINE_CLASS(signature);

static int query_formats(AVFilterContext *ctx)
{
    /* all formats with a separate gray value */
    static const enum AVPixelFormat pix_fmts[] = {
        AV_PIX_FMT_GRAY8,
        AV_PIX_FMT_YUV410P, AV_PIX_FMT_YUV411P,
        AV_PIX_FMT_YUV420P, AV_PIX_FMT_YUV422P,
        AV_PIX_FMT_YUV440P, AV_PIX_FMT_YUV444P,
        AV_PIX_FMT_YUVJ411P, AV_PIX_FMT_YUVJ420P,
        AV_PIX_FMT_YUVJ422P, AV_PIX_FMT_YUVJ444P,
        AV_PIX_FMT_YUVJ440P,
        AV_PIX_FMT_NV12, AV_PIX_FMT_NV21,
        AV_PIX_FMT_NONE
    };

    return ff_set_common_formats(ctx, ff_make_format_list(pix_fmts));
}

static int config_input(AVFilterLink *inlink)
{
    AVFilterContext *ctx = inlink->dst;
    SignatureContext *sic = ctx->priv;
    StreamContext *sc = &(sic->streamcontexts[FF_INLINK_IDX(inlink)]);

    sc->time_base = inlink->time_base;
    /* test for overflow */
    sc->divide = (((uint64_t) inlink->w/32) * (inlink->w/32 + 1) * (inlink->h/32 * inlink->h/32 + 1) > INT64_MAX / (BLOCK_LCM * 255));
    if (sc->divide) {
        av_log(ctx, AV_LOG_WARNING, "Input dimension too high for precise calculation, numbers will be rounded.\n");
    }
    sc->w = inlink->w;
    sc->h = inlink->h;
    return 0;
}

static int filter_frame(AVFilterLink *inlink, AVFrame *picref)
{
    AVFilterContext *ctx = inlink->dst;
    SignatureContext *sic = ctx->priv;
    StreamContext *sc = &(sic->streamcontexts[FF_INLINK_IDX(inlink)]);
    FineSignature* fs;

    uint64_t intpic[32][32];
    uint64_t rowcount;
    uint8_t *p = picref->data[0];
    int inti, intj;
    int *intjlut;
    int i, j, ret;
    int32_t dh1 = 1, dh2 = 1, dw1 = 1, dw2 = 1, a, b;
    int64_t denom;

    int64_t precfactor = (sc->divide) ? 65536 : BLOCK_LCM;

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

    fs->pts = picref->pts;
    fs->index = sc->lastindex++;

    memset(intpic, 0, sizeof(uint64_t)*32*32);
    intjlut = av_malloc_array(inlink->w, sizeof(int));
    if (!intjlut)
        return AVERROR(ENOMEM);
    for (i = 0; i < inlink->w; i++) {
        intjlut[i] = (i*32)/inlink->w;
    }

    for (i = 0; i < inlink->h; i++) {
        inti = (i*32)/inlink->h;
        for (j = 0; j < inlink->w; j++) {
            intj = intjlut[j];
            intpic[inti][intj] += p[j];
        }
        p += picref->linesize[0];
    }
    av_freep(&intjlut);

    /* The following calculates a summed area table (intpic) and brings the numbers
     * in intpic to the same denominator.
     * So you only have to handle the numinator in the following sections.
     */
    dh1 = inlink->h / 32;
    if (inlink->h % 32)
        dh2 = dh1 + 1;
    dw1 = inlink->w / 32;
    if (inlink->w % 32)
        dw2 = dw1 + 1;
    denom = (sc->divide) ? dh1 * dh2 * dw1 * dw2 : 1;

    for (i = 0; i < 32; i++) {
        rowcount = 0;
        a = 1;
        if (dh2 > 1) {
            a = ((inlink->h*(i+1))%32 == 0) ? (inlink->h*(i+1))/32 - 1 : (inlink->h*(i+1))/32;
            a -= ((inlink->h*i)%32 == 0) ? (inlink->h*i)/32 - 1 : (inlink->h*i)/32;
            a = (a == dh1)? dh2 : dh1;
        }
        for (j = 0; j < 32; j++) {
            b = 1;
            if (dw2 > 1) {
                b = ((inlink->w*(j+1))%32 == 0) ? (inlink->w*(j+1))/32 - 1 : (inlink->w*(j+1))/32;
                b -= ((inlink->w*j)%32 == 0) ? (inlink->w*j)/32 - 1 : (inlink->w*j)/32;
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
    if (ret < 0) return ret;

    if (FF_INLINK_IDX(inlink) == 0)
        return ff_filter_frame(inlink->dst->outputs[0], picref);
    return 1;
}

static int export(AVFilterContext *ctx, StreamContext *sc, int input)
{
    SignatureContext* sic = ctx->priv;
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

static int request_frame(AVFilterLink *outlink)
{
    AVFilterContext *ctx = outlink->src;
    SignatureContext *sic = ctx->priv;
    StreamContext *sc, *sc2;
    MatchingInfo match;
    int i, j, ret;
    int lookup = 1; /* indicates wheather EOF of all files is reached */

    /* process all inputs */
    for (i = 0; i < sic->nb_inputs; i++){
        sc = &(sic->streamcontexts[i]);

        ret = ff_request_frame(ctx->inputs[i]);

        /* return if unexpected error occurs in input stream */
        if (ret < 0 && ret != AVERROR_EOF)
            return ret;

        /* export signature at EOF */
        if (ret == AVERROR_EOF && !sc->exported) {
            /* export if wanted */
            if (strlen(sic->filename) > 0) {
                if (export(ctx, sc, i) < 0)
                    return ret;
            }
            sc->exported = 1;
        }
        lookup &= sc->exported;
    }

    /* signature lookup */
    if (lookup && sic->mode != MODE_OFF) {
        /* iterate over every pair */
        for (i = 0; i < sic->nb_inputs; i++) {
            sc = &(sic->streamcontexts[i]);
            for (j = i+1; j < sic->nb_inputs; j++) {
                sc2 = &(sic->streamcontexts[j]);
                match = lookup_signatures(ctx, sic, sc, sc2, sic->mode);
                if (match.score != 0) {
                    av_log(ctx, AV_LOG_INFO, "matching of video %d at %f and %d at %f, %d frames matching\n",
                            i, ((double) match.first->pts * sc->time_base.num) / sc->time_base.den,
                            j, ((double) match.second->pts * sc2->time_base.num) / sc2->time_base.den,
                            match.matchframes);
                    if (match.whole)
                        av_log(ctx, AV_LOG_INFO, "whole video matching\n");
                } else {
                    av_log(ctx, AV_LOG_INFO, "no matching of video %d and %d\n", i, j);
                }
            }
        }
    }

    return ret;
}

static av_cold int init(AVFilterContext *ctx)
{

    SignatureContext *sic = ctx->priv;
    StreamContext *sc;
    int i, ret;
    char tmp[1024];

    sic->streamcontexts = av_mallocz(sic->nb_inputs * sizeof(StreamContext));
    if (!sic->streamcontexts)
        return AVERROR(ENOMEM);

    for (i = 0; i < sic->nb_inputs; i++) {
        AVFilterPad pad = {
            .type = AVMEDIA_TYPE_VIDEO,
            .name = av_asprintf("in%d", i),
            .config_props = config_input,
            .filter_frame = filter_frame,
        };

        if (!pad.name)
            return AVERROR(ENOMEM);
        if ((ret = ff_insert_inpad(ctx, i, &pad)) < 0) {
            av_freep(&pad.name);
            return ret;
        }

        sc = &(sic->streamcontexts[i]);

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
    }

    /* check filename */
    if (sic->nb_inputs > 1 && strlen(sic->filename) > 0 && av_get_frame_filename(tmp, sizeof(tmp), sic->filename, 0) == -1) {
        av_log(ctx, AV_LOG_ERROR, "The filename must contain %%d or %%0nd, if you have more than one input.\n");
        return AVERROR(EINVAL);
    }

    return 0;
}

static av_cold void uninit(AVFilterContext *ctx)
{
    SignatureContext *sic = ctx->priv;
    StreamContext *sc;
    void* tmp;
    FineSignature* finsig;
    CoarseSignature* cousig;
    int i;


    /* free the lists */
    if (sic->streamcontexts != NULL) {
        for (i = 0; i < sic->nb_inputs; i++) {
            sc = &(sic->streamcontexts[i]);
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
        }
        av_freep(&sic->streamcontexts);
    }
    for (unsigned i = 0; i < ctx->nb_inputs; i++)
        av_freep(&ctx->input_pads[i].name);
}

static int config_output(AVFilterLink *outlink)
{
    AVFilterContext *ctx = outlink->src;
    AVFilterLink *inlink = ctx->inputs[0];

    outlink->time_base = inlink->time_base;
    outlink->frame_rate = inlink->frame_rate;
    outlink->sample_aspect_ratio = inlink->sample_aspect_ratio;
    outlink->w = inlink->w;
    outlink->h = inlink->h;

    return 0;
}

static void release_streamcontext(StreamContext *sc)
{
    free(sc->coarsesiglist);
    free(sc->finesiglist);
}

static int get_filesize(const char *filename)
{
    int fileLength = 0;
    FILE *f = NULL;
    f = fopen(filename, "rb");
    if(f != NULL) {
        fseek(f, 0, SEEK_END);
        fileLength = ftell(f);
        fclose(f);
    }
    return fileLength;
}

static uint8_t * get_filebuffer(const char *filename, int* fileLength)
{
    FILE *f = NULL;
    unsigned int readLength, paddedLength = 0;
    uint8_t *buffer = NULL;

    //check input parameters
    if (strlen(filename) <= 0) return buffer;
    f = fopen(filename, "rb");
    if (f == NULL) {
        av_log(NULL, AV_LOG_ERROR, "Could not open the file %s\n", filename);
        return buffer;
    }
    *fileLength = get_filesize(filename);
    if(*fileLength > 0) {
        // Cast to float is necessary to avoid int division
        paddedLength = ceil(*fileLength / (float)AV_INPUT_BUFFER_PADDING_SIZE)*AV_INPUT_BUFFER_PADDING_SIZE + AV_INPUT_BUFFER_PADDING_SIZE;
        buffer = (uint8_t*)av_calloc(paddedLength, sizeof(uint8_t));
        if (!buffer) {
            av_log(NULL, AV_LOG_ERROR, "Could not allocate memory for reading signature file\n");
            fclose(f);
            return NULL;
        }
        // Read entire file into memory
        readLength = fread(buffer, sizeof(uint8_t), *fileLength, f);
        if(readLength != *fileLength) {
            av_log(NULL, AV_LOG_ERROR, "Could not read the file %s\n", filename);
            free(buffer);
            buffer = NULL;
        }
    }
    fclose(f);
    return buffer;
}

static int binary_import(uint8_t *buffer, int fileLength, StreamContext *sc)
{
    int ret = 0;

    unsigned int numOfSegments = 0;
    GetBitContext bitContext = { 0 };
    BoundedCoarseSignature *bCs;
    unsigned int i, j, k;
    int totalLength = 8 * fileLength;
    int finesigncount = 0;
    BoundedCoarseSignature *bCoarseList;

    init_get_bits(&bitContext, buffer, totalLength);

    // Skip the following data:
    // - NumOfSpatial Regions: (32 bits) only 1 supported
    // - SpatialLocationFlag: (1 bit) always the whole image
    // - PixelX_1: (16 bits) always 0
    // - PixelY_1: (16 bits) always 0
    skip_bits(&bitContext, 32 + 1 + 16 * 2);

    // width - 1, and height - 1
    // PixelX_2: (16 bits) is width - 1
    // PixelY_2: (16 bits) is height - 1
    sc->w = get_bits(&bitContext, 16);
    sc->h = get_bits(&bitContext, 16);
    ++sc->w;
    ++sc->h;

    // StartFrameOfSpatialRegion, always 0
    skip_bits(&bitContext, 32);

    // NumOfFrames
    // it's the number of fine signatures
    sc->lastindex = get_bits_long(&bitContext, 32);

    // MediaTimeUnit
    // sc->time_base.den / sc->time_base.num
    // hoping num is 1, other values are vague
    // den/num might be greater than 16 bit, so cutting it
    //put_bits(&buf, 16, 0xFFFF & (sc->time_base.den / sc->time_base.num));

    sc->time_base.den = get_bits(&bitContext, 16);
    sc->time_base.num = 1;

    // Skip the following data
    // - MediaTimeFlagOfSpatialRegion: (1 bit) always 1
    // - StartMediaTimeOfSpatialRegion: (32 bits) always 0
    // - EndMediaTimeOfSpatialRegion: (32 bits)
    skip_bits(&bitContext, 1 + 32*2);

    // Coarse signatures
    // numOfSegments = number of coarse signatures
    numOfSegments = get_bits_long(&bitContext, 32);
    if (!numOfSegments) {
        return -1;
    }

    sc->coarsesiglist = (CoarseSignature*)av_calloc(numOfSegments, sizeof(CoarseSignature));
    if(sc->coarsesiglist == NULL) {
        return AVERROR(ENOMEM);
    }

    bCoarseList = (BoundedCoarseSignature*)av_calloc(numOfSegments, sizeof(BoundedCoarseSignature));
    if(bCoarseList == NULL) {
        av_free(sc->coarsesiglist);
        return AVERROR(ENOMEM);
    }

    // CoarseSignature loading
    for (i = 0; i < numOfSegments; ++i) {
        bCs = &bCoarseList[i];
        bCs->cSign = &sc->coarsesiglist[i];

        if (i < numOfSegments - 1)
            bCs->cSign->next = &sc->coarsesiglist[i + 1];
        // each coarse signature is a VSVideoSegment
        // StartFrameOfSegment
        bCs->firstIndex = get_bits_long(&bitContext, 32);
        // EndFrameOfSegment
        bCs->lastIndex = get_bits_long(&bitContext, 32);

        // MediaTimeFlagOfSegment 1 bit, always 1
        skip_bits(&bitContext, 1);

        // Fine signature pts
        // StartMediaTimeOfSegment 32 bits
        bCs->firstPts = get_bits_long(&bitContext, 32);
        // EndMediaTimeOfSegment 32 bits
        bCs->lastPts = get_bits_long(&bitContext, 32);
        // Bag of words
        for ( j = 0; j < 5; ++j) {
            // read 243 bits ( = 7 * 32 + 19 = 8 * 28 + 19) into buffer
            for ( k = 0; k < 30; ++k) {
                // 30*8 bits = 30 bytes
                bCs->cSign->data[j][k] = get_bits(&bitContext, 8);
            }
            bCs->cSign->data[j][30] = get_bits(&bitContext, 3) << 5;
        }
    }
    sc->coarseend = &sc->coarsesiglist[numOfSegments - 1];

    // Finesignatures
    // CompressionFlag, only 0 supported
    skip_bits(&bitContext, 1);

    // Check lastindex for validity
    finesigncount = (totalLength - bitContext.index) / MPEG7_FINESIG_NBITS;

    if(sc->lastindex != finesigncount)
        sc->lastindex = finesigncount;
    sc->finesiglist = (FineSignature*)av_calloc(sc->lastindex, sizeof(FineSignature));
    if(sc->finesiglist == NULL) {
        av_free(sc->coarsesiglist);
        av_free(bCoarseList);
        return AVERROR(ENOMEM);
    }

    // Load fine signatures from file
    for (i = 0; i < sc->lastindex; ++i) {
        FineSignature *fs = &sc->finesiglist[i];

        // MediaTimeFlagOfFrame always 1
        skip_bits(&bitContext, 1);

        // MediaTimeOfFrame (PTS)
        fs->pts = get_bits_long(&bitContext, 32);

        // FrameConfidence
        fs->confidence = get_bits(&bitContext, 8);

        // words
        for (k = 0; k < 5; k++) {
            fs->words[k] = get_bits(&bitContext, 8);
        }
        // framesignature
        for (k = 0; k < SIGELEM_SIZE / 5; k++) {
            fs->framesig[k] = get_bits(&bitContext, 8);
        }
    }

    // Creating FineSignature linked list
    for (i = 0; i < sc->lastindex; ++i) {
        FineSignature *fs = &sc->finesiglist[i];
        // Building fine signature list
        // First element prev should be NULL
        // Last element next should be NULL
        if (i == 0) {
            fs->next = &fs[1];
            fs->prev = NULL;
        }
        else if (i == sc->lastindex - 1) {
            fs->next = NULL;
            fs->prev = &fs[-1];
        }
        else {
            fs->next = &fs[1];
            fs->prev = &fs[-1];
        }
    }

    // Fine signature ranges DO overlap
    // Assign FineSignatures to CoarseSignatures
    for (i = 0; i < numOfSegments; ++i) {
        BoundedCoarseSignature *bCs = &bCoarseList[i];

        // O = n^2 probably it can be done faster
        for (j = 0;  j < sc->lastindex  && sc->finesiglist[j].pts <= bCs->lastPts; ++j) {
            FineSignature *fs = &sc->finesiglist[j];

            if (fs->pts >= bCs->firstPts) {
                // Check if the fragment's pts is inside coarse signature
                // bounds. Upper bound is checked in for loop
                if (!bCs->cSign->first) {
                    bCs->cSign->first = fs;
                }

                if (bCs->cSign->last) {
                    if (bCs->cSign->last->pts <= fs->pts)
                        bCs->cSign->last = fs;
                }
                else {
                    bCs->cSign->last = fs;
                }
            }

        }
        bCs->cSign->first->index = bCs->firstIndex;
        bCs->cSign->last->index = bCs->lastIndex;
    }

    av_free(bCoarseList);

    return ret;
}

static int compare_signbuffer(uint8_t* signbuf1, int len1, uint8_t* signbuf2, int len2) {
    int ret = -1;
    StreamContext scontexts[INPUTS_COUNT] = { 0 };
    MatchingInfo result = { 0 };
    SignatureContext sigContext = {
        .class = NULL,
        .mode = MODE_FULL,
        .nb_inputs = INPUTS_COUNT,
        .filename = NULL,
        .thworddist = 9000,
        .thcomposdist = 60000,
        .thl1 = 116,
        .thdi = 0,
        .thit = 0.5,
        .streamcontexts = scontexts
    };
    if (binary_import(signbuf1, len1, &scontexts[0]) < 0 || binary_import(signbuf2, len2, &scontexts[1]) < 0) {
        av_log(NULL, AV_LOG_ERROR, "Could not create StreamContext from binary data\n");
        return ret;
    }
    result = lookup_signatures(NULL, &sigContext, &scontexts[0], &scontexts[1], MODE_FULL);

    if (result.score != 0) {
        if (result.whole) ret = 2;//full matching
        else ret = 1; //partial matching
    }
    else {
        ret = 0; //no matching
    }

    release_streamcontext(&scontexts[0]);
    release_streamcontext(&scontexts[1]);

    return ret;
}

int avfilter_compare_sign_bybuff(uint8_t *signbuf1, int len1, uint8_t *signbuf2, int len2)
{
    int ret = -1;

    if(signbuf1 != NULL && signbuf2 != NULL && len1 > 0 && len2 > 0) {
        ret = compare_signbuffer(signbuf1, len1, signbuf2, len2);
    }

    return ret;
}

int avfilter_compare_sign_bypath(char *signpath1, char *signpath2)
{
    int ret = -1;

    int len1, len2;
    uint8_t *buffer1, *buffer2;
    buffer1 = get_filebuffer(signpath1, &len1);
    if(buffer1 == NULL) return AVERROR(ENOMEM);
    buffer2 = get_filebuffer(signpath2, &len2);
    if(buffer2 == NULL) {
        free(buffer1);
        return AVERROR(ENOMEM);
    }
    ret = compare_signbuffer(buffer1, len1, buffer2, len2);

    if(buffer1 != NULL)
        free(buffer1);
    if(buffer2 != NULL)
        free(buffer2);

    return ret;
}

static const AVFilterPad signature_outputs[] = {
    {
        .name          = "default",
        .type          = AVMEDIA_TYPE_VIDEO,
        .request_frame = request_frame,
        .config_props  = config_output,
    },
    { NULL }
};

AVFilter ff_vf_signature = {
    .name          = "signature",
    .description   = NULL_IF_CONFIG_SMALL("Calculate the MPEG-7 video signature"),
    .priv_size     = sizeof(SignatureContext),
    .priv_class    = &signature_class,
    .init          = init,
    .uninit        = uninit,
    .query_formats = query_formats,
    .outputs       = signature_outputs,
    .inputs        = NULL,
    .flags         = AVFILTER_FLAG_DYNAMIC_INPUTS,
};
