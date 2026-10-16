// CPU-only neural upscaler runner (ncnn, no Vulkan). Two engines, same command line style as the
// *-ncnn-vulkan tools so the Rust side can treat them alike:
//   Real-ESRGAN:  esrgan-cpu -i in.png -o out.png -m modeldir -n modelname [-s scale] [-j threads] [-t tile]
//   waifu2x:      esrgan-cpu -e waifu2x -i in.png -o out.png -m models-cunet -N noise(-1..3) -s scale(1|2|4) [-j threads] [-t tile]
// Models are the same ncnn .param/.bin files the Vulkan tools use. Algorithms follow
// Real-ESRGAN-ncnn-vulkan and nihui/waifu2x-ncnn-vulkan (CPU code path).
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "net.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

static int scale_from_name(const std::string& n, int def)
{
    size_t p = n.rfind("-x");
    if (p == std::string::npos) return def;
    int v = atoi(n.c_str() + p + 2);
    return v >= 2 && v <= 4 ? v : def;
}

static void progress(int done, int total)
{
    fprintf(stderr, "%.2f%%\n", 100.0 * done / total);
    fflush(stderr);
}

// ------------------------------------------------------------------------------------------------
// Real-ESRGAN
// ------------------------------------------------------------------------------------------------
static int run_esrgan(ncnn::Net& net, const unsigned char* px, int w, int h, int scale, int tile, std::vector<unsigned char>& res)
{
    const int pad = 10;
    const int ow = w * scale;
    res.assign((size_t)ow * h * scale * 3, 0);
    const int tx = (w + tile - 1) / tile, ty = (h + tile - 1) / tile;
    const float norm[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    int done = 0;
    for (int yi = 0; yi < ty; yi++)
    {
        for (int xi = 0; xi < tx; xi++)
        {
            int x0 = xi * tile, y0 = yi * tile;
            int x1 = std::min(x0 + tile, w), y1 = std::min(y0 + tile, h);
            int ix0 = std::max(x0 - pad, 0), iy0 = std::max(y0 - pad, 0);
            int ix1 = std::min(x1 + pad, w), iy1 = std::min(y1 + pad, h);
            int iw = ix1 - ix0, ih = iy1 - iy0;
            ncnn::Mat src = ncnn::Mat::from_pixels_roi(px, ncnn::Mat::PIXEL_RGB, w, h, ix0, iy0, iw, ih);
            // replicate-pad to a multiple of 4 (some layers require it)
            int pw = (iw + 3) / 4 * 4, ph = (ih + 3) / 4 * 4;
            ncnn::Mat in_pad;
            ncnn::copy_make_border(src, in_pad, 0, ph - ih, 0, pw - iw, ncnn::BORDER_REPLICATE, 0.f);
            in_pad.substract_mean_normalize(0, norm);

            ncnn::Extractor ex = net.create_extractor();
            ex.input("data", in_pad);
            ncnn::Mat o;
            if (ex.extract("output", o) != 0) return 1;
            int ox0 = (x0 - ix0) * scale, oy0 = (y0 - iy0) * scale;
            int cw = (x1 - x0) * scale, ch = (y1 - y0) * scale;
            for (int cc = 0; cc < 3; cc++)
            {
                const float* pl = o.channel(cc);
                for (int yy = 0; yy < ch; yy++)
                {
                    const float* row = pl + (size_t)(oy0 + yy) * o.w + ox0;
                    unsigned char* dst = &res[((size_t)(y0 * scale + yy) * ow + x0 * scale) * 3 + cc];
                    for (int xx = 0; xx < cw; xx++)
                    {
                        float v = row[xx] * 255.f + 0.5f;
                        dst[(size_t)xx * 3] = (unsigned char)(v < 0 ? 0 : v > 255 ? 255 : v);
                    }
                }
            }
            progress(++done, tx * ty);
        }
    }
    return 0;
}

// ------------------------------------------------------------------------------------------------
// waifu2x: one pass with per-pass scale 1 (denoise only) or 2
// ------------------------------------------------------------------------------------------------
static int waifu2x_pass(ncnn::Net& net, const unsigned char* px, int w, int h, int scale, int prepadding, int tile,
                        std::vector<unsigned char>& res, int done_base, int total_passes)
{
    const int ow = w * scale;
    res.assign((size_t)ow * h * scale * 3, 0);
    const int xtiles = (w + tile - 1) / tile, ytiles = (h + tile - 1) / tile;
    int done = 0;
    for (int yi = 0; yi < ytiles; yi++)
    {
        const int tile_h_nopad = std::min((yi + 1) * tile, h) - yi * tile;
        int prepadding_bottom = prepadding;
        if (scale == 1) prepadding_bottom += (tile_h_nopad + 3) / 4 * 4 - tile_h_nopad;
        if (scale == 2) prepadding_bottom += (tile_h_nopad + 1) / 2 * 2 - tile_h_nopad;
        const int in_tile_y0 = std::max(yi * tile - prepadding, 0);
        const int in_tile_y1 = std::min((yi + 1) * tile + prepadding_bottom, h);

        for (int xi = 0; xi < xtiles; xi++)
        {
            const int tile_w_nopad = std::min((xi + 1) * tile, w) - xi * tile;
            int prepadding_right = prepadding;
            if (scale == 1) prepadding_right += (tile_w_nopad + 3) / 4 * 4 - tile_w_nopad;
            if (scale == 2) prepadding_right += (tile_w_nopad + 1) / 2 * 2 - tile_w_nopad;
            const int in_tile_x0 = std::max(xi * tile - prepadding, 0);
            const int in_tile_x1 = std::min((xi + 1) * tile + prepadding_right, w);

            ncnn::Mat in = ncnn::Mat::from_pixels_roi(px, ncnn::Mat::PIXEL_RGB, w, h, in_tile_x0, in_tile_y0, in_tile_x1 - in_tile_x0, in_tile_y1 - in_tile_y0);
            const float norm[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
            in.substract_mean_normalize(0, norm);

            int pad_top = std::max(prepadding - yi * tile, 0);
            int pad_bottom = std::max(std::min((yi + 1) * tile + prepadding_bottom - h, prepadding_bottom), 0);
            int pad_left = std::max(prepadding - xi * tile, 0);
            int pad_right = std::max(std::min((xi + 1) * tile + prepadding_right - w, prepadding_right), 0);
            ncnn::Mat padded;
            ncnn::copy_make_border(in, padded, pad_top, pad_bottom, pad_left, pad_right, ncnn::BORDER_REPLICATE, 0.f);

            ncnn::Extractor ex = net.create_extractor();
            ex.input("Input1", padded);
            ncnn::Mat o;
            if (ex.extract("Eltwise4", o) != 0) return 1;

            const int cw = tile_w_nopad * scale, ch = tile_h_nopad * scale;
            if (o.w < cw || o.h < ch)
            {
                fprintf(stderr, "unexpected output size %dx%d (need %dx%d)\n", o.w, o.h, cw, ch);
                return 1;
            }
            for (int q = 0; q < 3; q++)
            {
                const float* pl = o.channel(q);
                for (int yy = 0; yy < ch; yy++)
                {
                    const float* row = pl + (size_t)yy * o.w;
                    unsigned char* dst = &res[((size_t)(yi * tile * scale + yy) * ow + (size_t)xi * tile * scale) * 3 + q];
                    for (int xx = 0; xx < cw; xx++)
                    {
                        float v = row[xx] * 255.f + 0.5f;
                        dst[(size_t)xx * 3] = (unsigned char)(v < 0 ? 0 : v > 255 ? 255 : v);
                    }
                }
            }
            ++done;
            progress(done_base * xtiles * ytiles + done, total_passes * xtiles * ytiles);
        }
    }
    return 0;
}

int main(int argc, char** argv)
{
    std::string in, out, models = "models", name = "realesr-animevideov3-x2", engine = "esrgan";
    int scale = 2, threads = 0, tile = 0, noise = 1;
    for (int i = 1; i + 1 < argc; i += 2)
    {
        std::string a = argv[i];
        const char* v = argv[i + 1];
        if (a == "-i") in = v;
        else if (a == "-o") out = v;
        else if (a == "-m") models = v;
        else if (a == "-n") name = v;
        else if (a == "-e") engine = v;
        else if (a == "-N") noise = atoi(v);
        else if (a == "-s") scale = atoi(v);
        else if (a == "-j") threads = atoi(v);
        else if (a == "-t") tile = atoi(v);
    }
    if (in.empty() || out.empty())
    {
        fprintf(stderr, "usage: esrgan-cpu [-e esrgan|waifu2x] -i in -o out -m models -n name [-N noise] [-s scale] [-j threads] [-t tile]\n");
        return 2;
    }

    int w, h, c;
    unsigned char* px = stbi_load(in.c_str(), &w, &h, &c, 3);
    if (!px)
    {
        fprintf(stderr, "cannot read %s\n", in.c_str());
        return 1;
    }

    ncnn::Net net;
    net.opt.use_vulkan_compute = false;
    if (threads > 0) net.opt.num_threads = threads;

    std::vector<unsigned char> res;
    int ow = 0, oh = 0;

    if (engine == "waifu2x")
    {
        // models = path of e.g. ".../models-cunet"; scale 1 = denoise only, 2 and 4 = 2x passes
        if (!(scale == 1 || scale == 2 || scale == 4)) scale = 2;
        std::string base;
        if (noise < 0) base = "scale2.0x_model";
        else if (scale == 1) base = "noise" + std::to_string(noise) + "_model";
        else base = "noise" + std::to_string(noise) + "_scale2.0x_model";
        std::string pp = models + "/" + base + ".param", bp = models + "/" + base + ".bin";
        if (net.load_param(pp.c_str()) != 0 || net.load_model(bp.c_str()) != 0)
        {
            fprintf(stderr, "cannot load model %s\n", pp.c_str());
            return 1;
        }
        const bool cunet = models.find("cunet") != std::string::npos;
        int prepadding = 7;
        if (cunet) prepadding = (noise >= 0 && scale == 1) ? 28 : 18;
        if (tile <= 0) tile = cunet ? 200 : 300;
        const int passes = scale == 1 ? 1 : (scale == 2 ? 1 : 2);
        fprintf(stderr, "[cpu] waifu2x %dx%d scale=%d noise=%d threads=%d model=%s\n", w, h, scale, noise, net.opt.num_threads, models.c_str());
        if (noise < 0 && scale == 1) { res.assign(px, px + (size_t)w * h * 3); ow = w; oh = h; }
        else
        {
            std::vector<unsigned char> cur(px, px + (size_t)w * h * 3), next;
            int cw = w, ch = h;
            for (int p = 0; p < passes; p++)
            {
                int ps = scale == 1 ? 1 : 2;
                if (waifu2x_pass(net, cur.data(), cw, ch, ps, prepadding, tile, next, p, passes) != 0)
                {
                    fprintf(stderr, "inference failed\n");
                    return 1;
                }
                cw *= ps; ch *= ps;
                cur.swap(next);
            }
            res.swap(cur); ow = cw; oh = ch;
        }
    }
    else
    {
        scale = scale_from_name(name, scale);
        if (tile < 32) tile = 200;
        std::string pp = models + "/" + name + ".param", bp = models + "/" + name + ".bin";
        if (net.load_param(pp.c_str()) != 0 || net.load_model(bp.c_str()) != 0)
        {
            fprintf(stderr, "cannot load model %s\n", pp.c_str());
            return 1;
        }
        fprintf(stderr, "[cpu] %dx%d scale=%d threads=%d model=%s\n", w, h, scale, net.opt.num_threads, name.c_str());
        if (run_esrgan(net, px, w, h, scale, tile, res) != 0)
        {
            fprintf(stderr, "inference failed\n");
            return 1;
        }
        ow = w * scale; oh = h * scale;
    }
    stbi_image_free(px);
    if (!stbi_write_png(out.c_str(), ow, oh, 3, res.data(), ow * 3))
    {
        fprintf(stderr, "cannot write %s\n", out.c_str());
        return 1;
    }
    return 0;
}
