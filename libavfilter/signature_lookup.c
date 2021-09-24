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
 */

#include "signature.h"

#define HOUGH_MAX_OFFSET 90
#define MAX_FRAMERATE 60

#define DIR_PREV 0
#define DIR_NEXT 1
#define DIR_PREV_END 2
#define DIR_NEXT_END 3

#define STATUS_NULL 0
#define STATUS_END_REACHED 1
#define STATUS_BEGIN_REACHED 2

static void fill_l1distlut(uint8_t lut[])
{
    int i, j, tmp_i, tmp_j,count;
    uint8_t dist;

    for (i = 0, count = 0; i < 242; i++) {
        for (j = i + 1; j < 243; j++, count++) {
            /* ternary distance between i and j */
            dist = 0;
            tmp_i = i; tmp_j = j;
            do {
                dist += FFABS((tmp_j % 3) - (tmp_i % 3));
                tmp_j /= 3;
                tmp_i /= 3;
            } while (tmp_i > 0 || tmp_j > 0);
            lut[count] = dist;
        }
    }
}

static unsigned int intersection_word(const uint8_t *first, const uint8_t *second)
{
    unsigned int val=0,i;
    for (i = 0; i < 28; i += 4) {
        val += av_popcount( (first[i]   & second[i]  ) << 24 |
                            (first[i+1] & second[i+1]) << 16 |
                            (first[i+2] & second[i+2]) << 8  |
                            (first[i+3] & second[i+3]) );
    }
    val += av_popcount( (first[28] & second[28]) << 16 |
                        (first[29] & second[29]) << 8  |
                        (first[30] & second[30]) );
    return val;
}

static unsigned int union_word(const uint8_t *first, const uint8_t *second)
{
    unsigned int val=0,i;
    for (i = 0; i < 28; i += 4) {
        val += av_popcount( (first[i]   | second[i]  ) << 24 |
                            (first[i+1] | second[i+1]) << 16 |
                            (first[i+2] | second[i+2]) << 8  |
                            (first[i+3] | second[i+3]) );
    }
    val += av_popcount( (first[28] | second[28]) << 16 |
                        (first[29] | second[29]) << 8  |
                        (first[30] | second[30]) );
    return val;
}

static unsigned int get_l1dist(AVFilterContext *ctx, SignatureContext *sc, const uint8_t *first, const uint8_t *second)
{
    unsigned int i;
    unsigned int dist = 0;
    uint8_t f, s;

    for (i = 0; i < SIGELEM_SIZE/5; i++) {
        if (first[i] != second[i]) {
            f = first[i];
            s = second[i];
            if (f > s) {
                /* little variation of gauss sum formula */
                dist += sc->l1distlut[243*242/2 - (243-s)*(242-s)/2 + f - s - 1];
            } else {
                dist += sc->l1distlut[243*242/2 - (243-f)*(242-f)/2 + s - f - 1];
            }
        }
    }
    return dist;
}

/**
 * calculates the jaccard distance and evaluates a pair of coarse signatures as good
 * @return 0 if pair is bad, 1 otherwise
 */
static int get_jaccarddist(SignatureContext *sc, CoarseSignature *first, CoarseSignature *second)
{
    int jaccarddist, i, composdist = 0, cwthcount = 0;
    for (i = 0; i < 5; i++) {
        if ((jaccarddist = intersection_word(first->data[i], second->data[i])) > 0) {
            jaccarddist /= union_word(first->data[i], second->data[i]);
        }
        if (jaccarddist >= sc->thworddist) {
            if (++cwthcount > 2) {
                /* more than half (5/2) of distances are too wide */
                return 0;
            }
        }
        composdist += jaccarddist;
        if (composdist > sc->thcomposdist) {
            return 0;
        }
    }
    return 1;
}

/**
 * step through the coarsesignatures as long as a good candidate is found
 * @return 0 if no candidate is found, 1 otherwise
 */
static int find_next_coarsecandidate(SignatureContext *sc, CoarseSignature *secondstart, CoarseSignature **first, CoarseSignature **second, int start)
{
    /* go one coarsesignature foreword */
    if (!start) {
        if ((*second)->next) {
            *second = (*second)->next;
        } else if ((*first)->next) {
            *second = secondstart;
            *first = (*first)->next;
        } else {
            return 0;
        }
    }

    while (1) {
        if (get_jaccarddist(sc, *first, *second))
            return 1;

        /* next signature */
        if ((*second)->next) {
            *second = (*second)->next;
        } else if ((*first)->next) {
            *second = secondstart;
            *first = (*first)->next;
        } else {
            return 0;
        }
    }
}

/**
 * compares framesignatures and sorts out signatures with a l1 distance above a given threshold.
 * Then tries to find out offset and differences between framerates with a hough transformation
 */
static MatchingInfo* get_matching_parameters(AVFilterContext *ctx, SignatureContext *sc, FineSignature *first, FineSignature *second)
{
    FineSignature *f, *s;
    size_t i, j, k, l, hmax = 0, score;
    int framerate, offset, l1dist;
    double m;
    MatchingInfo *cands = NULL, *c = NULL;

    struct {
        uint8_t size;
        unsigned int dist;
        FineSignature *a;
        uint8_t b_pos[COARSE_SIZE];
        FineSignature *b[COARSE_SIZE];
    } pairs[COARSE_SIZE];

    typedef struct hspace_elem {
        int dist;
        size_t score;
        FineSignature *a;
        FineSignature *b;
    } hspace_elem;

    /* houghspace */
    hspace_elem** hspace = av_malloc_array(MAX_FRAMERATE, sizeof(hspace_elem *));

    /* initialize houghspace */
    for (i = 0; i < MAX_FRAMERATE; i++) {
        hspace[i] = av_malloc_array(2 * HOUGH_MAX_OFFSET + 1, sizeof(hspace_elem));
        for (j = 0; j < 2 * HOUGH_MAX_OFFSET + 1; j++) {
            hspace[i][j].score = 0;
            hspace[i][j].dist = 99999;
        }
    }

    /* l1 distances */
    for (i = 0, f = first; i < COARSE_SIZE && f->next; i++, f = f->next) {
        pairs[i].size = 0;
        pairs[i].dist = 99999;
        pairs[i].a = f;
        for (j = 0, s = second; j < COARSE_SIZE && s->next; j++, s = s->next) {
            /* l1 distance of finesignature */
            l1dist = get_l1dist(ctx, sc, f->framesig, s->framesig);
            if (l1dist < sc->thl1) {
                if (l1dist < pairs[i].dist) {
                    pairs[i].size = 1;
                    pairs[i].dist = l1dist;
                    pairs[i].b_pos[0] = j;
                    pairs[i].b[0] = s;
                } else if (l1dist == pairs[i].dist) {
                    pairs[i].b[pairs[i].size] = s;
                    pairs[i].b_pos[pairs[i].size] = j;
                    pairs[i].size++;
                }
            }
        }
    }
    /* last incomplete coarsesignature */
    if (f->next == NULL) {
        for (; i < COARSE_SIZE; i++) {
            pairs[i].size = 0;
            pairs[i].dist = 99999;
        }
    }

    /* hough transformation */
    for (i = 0; i < COARSE_SIZE; i++) {
        for (j = 0; j < pairs[i].size; j++) {
            for (k = i + 1; k < COARSE_SIZE; k++) {
                for (l = 0; l < pairs[k].size; l++) {
                    if (pairs[i].b[j] != pairs[k].b[l]) {
                        /* linear regression */
                        m = (pairs[k].b_pos[l]-pairs[i].b_pos[j]) / (k-i); /* good value between 0.0 - 2.0 */
                        framerate = nearbyint(m*30 + 0.5); /* round up to 0 - 60 */
                        if (framerate>0 && framerate <= MAX_FRAMERATE) {
                            offset = pairs[i].b_pos[j] - nearbyint(m*i + 0.5); /* only second part has to be rounded up */
                            if (offset > -HOUGH_MAX_OFFSET && offset < HOUGH_MAX_OFFSET) {
                                if (pairs[i].dist < pairs[k].dist) {
                                    if (pairs[i].dist < hspace[framerate-1][offset+HOUGH_MAX_OFFSET].dist) {
                                        hspace[framerate-1][offset+HOUGH_MAX_OFFSET].dist = pairs[i].dist;
                                        hspace[framerate-1][offset+HOUGH_MAX_OFFSET].a = pairs[i].a;
                                        hspace[framerate-1][offset+HOUGH_MAX_OFFSET].b = pairs[i].b[j];
                                    }
                                } else {
                                    if (pairs[k].dist < hspace[framerate-1][offset+HOUGH_MAX_OFFSET].dist) {
                                        hspace[framerate-1][offset+HOUGH_MAX_OFFSET].dist = pairs[k].dist;
                                        hspace[framerate-1][offset+HOUGH_MAX_OFFSET].a = pairs[k].a;
                                        hspace[framerate-1][offset+HOUGH_MAX_OFFSET].b = pairs[k].b[l];
                                    }
                                }

                                score = hspace[framerate-1][offset+HOUGH_MAX_OFFSET].score + 1;
                                if (score > hmax )
                                    hmax = score;
                                hspace[framerate-1][offset+HOUGH_MAX_OFFSET].score = score;
                            }
                        }
                    }
                }
            }
        }
    }

    if (hmax > 0) {
        hmax = (int) (0.7*hmax);
        for (i = 0; i < MAX_FRAMERATE; i++) {
            for (j = 0; j < HOUGH_MAX_OFFSET; j++) {
                if (hmax < hspace[i][j].score) {
                    if (c == NULL) {
                        c = av_malloc(sizeof(MatchingInfo));
                        if (!c)
                            av_log(ctx, AV_LOG_FATAL, "Could not allocate memory");
                        cands = c;
                    } else {
                        c->next = av_malloc(sizeof(MatchingInfo));
                        if (!c->next)
                            av_log(ctx, AV_LOG_FATAL, "Could not allocate memory");
                        c = c->next;
                    }
                    c->framerateratio = (i+1.0) / 30;
                    c->score = hspace[i][j].score;
                    c->offset = j-90;
                    c->first = hspace[i][j].a;
                    c->second = hspace[i][j].b;
                    c->next = NULL;

                    /* not used */
                    c->meandist = 0;
                    c->matchframes = 0;
                    c->whole = 0;
                }
            }
        }
    }
    for (i = 0; i < MAX_FRAMERATE; i++) {
        av_freep(&hspace[i]);
    }
    av_freep(&hspace);
    return cands;
}

static int iterate_frame(double frr, FineSignature **a, FineSignature **b, int fcount, int *bcount, int dir)
{
    int step;

    /* between 1 and 2, because frr is between 1 and 2 */
    step = ((int) 0.5 + fcount     * frr) /* current frame */
          -((int) 0.5 + (fcount-1) * frr);/* last frame */

    if (dir == DIR_NEXT) {
        if (frr >= 1.0) {
            if ((*a)->next) {
                *a = (*a)->next;
            } else {
                return DIR_NEXT_END;
            }

            if (step == 1) {
                if ((*b)->next) {
                    *b = (*b)->next;
                    (*bcount)++;
                } else {
                    return DIR_NEXT_END;
                }
            } else {
                if ((*b)->next && (*b)->next->next) {
                    *b = (*b)->next->next;
                    (*bcount)++;
                } else {
                    return DIR_NEXT_END;
                }
            }
        } else {
            if ((*b)->next) {
                *b = (*b)->next;
                (*bcount)++;
            } else {
                return DIR_NEXT_END;
            }

            if (step == 1) {
                if ((*a)->next) {
                    *a = (*a)->next;
                } else {
                    return DIR_NEXT_END;
                }
            } else {
                if ((*a)->next && (*a)->next->next) {
                    *a = (*a)->next->next;
                } else {
                    return DIR_NEXT_END;
                }
            }
        }
        return DIR_NEXT;
    } else {
        if (frr >= 1.0) {
            if ((*a)->prev) {
                *a = (*a)->prev;
            } else {
                return DIR_PREV_END;
            }

            if (step == 1) {
                if ((*b)->prev) {
                    *b = (*b)->prev;
                    (*bcount)++;
                } else {
                    return DIR_PREV_END;
                }
            } else {
                if ((*b)->prev && (*b)->prev->prev) {
                    *b = (*b)->prev->prev;
                    (*bcount)++;
                } else {
                    return DIR_PREV_END;
                }
            }
        } else {
            if ((*b)->prev) {
                *b = (*b)->prev;
                (*bcount)++;
            } else {
                return DIR_PREV_END;
            }

            if (step == 1) {
                if ((*a)->prev) {
                    *a = (*a)->prev;
                } else {
                    return DIR_PREV_END;
                }
            } else {
                if ((*a)->prev && (*a)->prev->prev) {
                    *a = (*a)->prev->prev;
                } else {
                    return DIR_PREV_END;
                }
            }
        }
        return DIR_PREV;
    }
}

static MatchingInfo evaluate_parameters(AVFilterContext *ctx, SignatureContext *sc, MatchingInfo *infos, MatchingInfo bestmatch, int mode)
{
    int dist, distsum = 0, bcount = 1, dir = DIR_NEXT;
    int fcount = 0, goodfcount = 0, gooda = 0, goodb = 0;
    double meandist, minmeandist = bestmatch.meandist;
    int tolerancecount = 0;
    FineSignature *a, *b, *aprev, *bprev;
    int status = STATUS_NULL;

    for (; infos != NULL; infos = infos->next) {
        a = infos->first;
        b = infos->second;
        while (1) {
            dist = get_l1dist(ctx, sc, a->framesig, b->framesig);

            if (dist > sc->thl1) {
                if (a->confidence >= 1 || b->confidence >= 1) {
                    /* bad frame (because high different information) */
                    tolerancecount++;
                }

                if (tolerancecount > 2) {
                    a = aprev;
                    b = bprev;
                    if (dir == DIR_NEXT) {
                        /* turn around */
                        a = infos->first;
                        b = infos->second;
                        dir = DIR_PREV;
                    } else {
                        break;
                    }
                }
            } else {
                /* good frame */
                distsum += dist;
                goodfcount++;
                tolerancecount=0;

                aprev = a;
                bprev = b;

                if (a->confidence < 1) gooda++;
                if (b->confidence < 1) goodb++;
            }

            fcount++;

            dir = iterate_frame(infos->framerateratio, &a, &b, fcount, &bcount, dir);
            if (dir == DIR_NEXT_END) {
                status = STATUS_END_REACHED;
                a = infos->first;
                b = infos->second;
                dir = iterate_frame(infos->framerateratio, &a, &b, fcount, &bcount, DIR_PREV);
            }

            if (dir == DIR_PREV_END) {
                status |= STATUS_BEGIN_REACHED;
                break;
            }

            if (sc->thdi != 0 && bcount >= sc->thdi) {
                break; /* enough frames found */
            }
        }

        if (bcount < sc->thdi)
            continue; /* matching sequence is too short */
        if ((double) goodfcount / (double) fcount < sc->thit)
            continue;
        if ((double) goodfcount*0.5 < FFMAX(gooda, goodb))
            continue;

        meandist = (double) goodfcount / (double) distsum;

        if (meandist < minmeandist ||
                status == (STATUS_END_REACHED | STATUS_BEGIN_REACHED) ||
                mode == MODE_FAST){
            minmeandist = meandist;
            /* bestcandidate in this iteration */
            bestmatch.meandist = meandist;
            bestmatch.matchframes = bcount;
            bestmatch.framerateratio = infos->framerateratio;
            bestmatch.score = infos->score;
            bestmatch.offset = infos->offset;
            bestmatch.first = infos->first;
            bestmatch.second = infos->second;
            bestmatch.whole = 0; /* will be set to true later */
            bestmatch.next = NULL;
        }

        /* whole sequence is automatically best match */
        if (status == (STATUS_END_REACHED | STATUS_BEGIN_REACHED)) {
            bestmatch.whole = 1;
            break;
        }

        /* first matching sequence is enough, finding the best one is not necessary */
        if (mode == MODE_FAST) {
            break;
        }
    }
    return bestmatch;
}

static void sll_free(MatchingInfo *sll)
{
    void *tmp;
    while (sll) {
        tmp = sll;
        sll = sll->next;
        av_freep(&tmp);
    }
}

static MatchingInfo lookup_signatures(AVFilterContext *ctx, SignatureContext *sc, StreamContext *first, StreamContext *second, int mode)
{
    CoarseSignature *cs, *cs2;
    MatchingInfo *infos;
    MatchingInfo bestmatch;
    MatchingInfo *i;

    cs = first->coarsesiglist;
    cs2 = second->coarsesiglist;

    /* score of bestmatch is 0, if no match is found */
    bestmatch.score = 0;
    bestmatch.meandist = 99999;
    bestmatch.whole = 0;

    fill_l1distlut(sc->l1distlut);

    /* stage 1: coarsesignature matching */
    if (find_next_coarsecandidate(sc, second->coarsesiglist, &cs, &cs2, 1) == 0)
        return bestmatch; /* no candidate found */
    do {
        av_log(ctx, AV_LOG_DEBUG, "Stage 1: got coarsesignature pair. "
               "indices of first frame: %"PRIu32" and %"PRIu32"\n",
               cs->first->index, cs2->first->index);
        /* stage 2: l1-distance and hough-transform */
        av_log(ctx, AV_LOG_DEBUG, "Stage 2: calculate matching parameters\n");
        infos = get_matching_parameters(ctx, sc, cs->first, cs2->first);
        if (av_log_get_level() == AV_LOG_DEBUG) {
            for (i = infos; i != NULL; i = i->next) {
                av_log(ctx, AV_LOG_DEBUG, "Stage 2: matching pair at %"PRIu32" and %"PRIu32", "
                       "ratio %f, offset %d\n", i->first->index, i->second->index,
                       i->framerateratio, i->offset);
            }
        }
        /* stage 3: evaluation */
        av_log(ctx, AV_LOG_DEBUG, "Stage 3: evaluate\n");
        if (infos) {
            bestmatch = evaluate_parameters(ctx, sc, infos, bestmatch, mode);
            av_log(ctx, AV_LOG_DEBUG, "Stage 3: best matching pair at %"PRIu32" and %"PRIu32", "
                   "ratio %f, offset %d, score %d, %d frames matching\n",
                   bestmatch.first->index, bestmatch.second->index,
                   bestmatch.framerateratio, bestmatch.offset, bestmatch.score, bestmatch.matchframes);
            sll_free(infos);
        }
    } while (find_next_coarsecandidate(sc, second->coarsesiglist, &cs, &cs2, 0) && !bestmatch.whole);
    return bestmatch;

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

/**
 * sets the bit at position pos to 1 in data
 */
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

static int calc_signature(AVFilterContext *ctx, StreamContext *sc, FineSignature* fs, uint64_t intpic[32][32], int64_t denom, int64_t precfactor)
{
    int i, j, k, ternary;
    uint64_t blocksum;
    int blocksize;
    int64_t th; /* threshold */
    int64_t sum;
    uint64_t conflist[DIFFELEM_SIZE];
    int f = 0, g = 0, w = 0;
        static const uint8_t pot3[5] = { 3*3*3*3, 3*3*3, 3*3, 3, 1 };
    /* indexes of words : 210,217,219,274,334  44,175,233,270,273  57,70,103,237,269  100,285,295,337,354  101,102,111,275,296
    s2usw = sorted to unsorted wordvec: 44 is at index 5, 57 at index 10...
    */
    static const unsigned int wordvec[25] = {44,57,70,100,101,102,103,111,175,210,217,219,233,237,269,270,273,274,275,285,295,296,334,337,354};
    static const uint8_t      s2usw[25]   = { 5,10,11, 15, 20, 21, 12, 22,  6,  0,  1,  2,  7, 13, 14,  8,  9,  3, 23, 16, 17, 24,  4, 18, 19};

    uint8_t wordt2b[5] = { 0, 0, 0, 0, 0 }; /* word ternary to binary */
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

    /* debug printing finesignature */
    if (av_log_get_level() == AV_LOG_DEBUG) {
        av_log(ctx, AV_LOG_DEBUG, "input %d, confidence: %d\n", 0, fs->confidence);

        av_log(ctx, AV_LOG_DEBUG, "words:");
        for (i = 0; i < 5; i++) {
            av_log(ctx, AV_LOG_DEBUG, " %d:", fs->words[i] );
            av_log(ctx, AV_LOG_DEBUG, " %d", fs->words[i] / pot3[0] );
            for (j = 1; j < 5; j++)
                av_log(ctx, AV_LOG_DEBUG, ",%d", fs->words[i] % pot3[j-1] / pot3[j] );
            av_log(ctx, AV_LOG_DEBUG, ";");
        }
        av_log(ctx, AV_LOG_DEBUG, "\n");

        av_log(ctx, AV_LOG_DEBUG, "framesignature:");
        for (i = 0; i < SIGELEM_SIZE/5; i++) {
            av_log(ctx, AV_LOG_DEBUG, " %d", fs->framesig[i] / pot3[0] );
            for (j = 1; j < 5; j++)
                av_log(ctx, AV_LOG_DEBUG, ",%d", fs->framesig[i] % pot3[j-1] / pot3[j] );
        }
        av_log(ctx, AV_LOG_DEBUG, "\n");
    }

    return 0;
}
