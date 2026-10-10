// ============================================================
//  stitcher.cpp
//
//  My panorama stitcher, written from scratch in one file while
//  learning computer vision. No OpenCV here on purpose, I wanted
//  to see how it actually works.
//
//  The idea:
//    1. find corners (keypoints) with Harris
//    2. make a small patch descriptor for each corner
//    3. match descriptors between two photos
//    4. guess a 3x3 perspective matrix with RANSAC
//    5. chain the matrices and warp all photos onto one canvas
//
//  Build:  make
//  Run:    ./stitcher img1.ppm img2.ppm ... result.ppm
// ============================================================

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <cstdlib>
#include <algorithm>
using namespace std;

// ---------------- settings (kept changing these) ----------------
const int   PATCH_RADIUS     = 8;      // descriptor is a 17x17 patch
const int   MAX_KEYPOINTS    = 500;
const float HARRIS_K         = 0.04f;
const float HARRIS_THRESHOLD = 0.005f; // relative to the strongest corner
const float MATCH_RATIO      = 0.80f;  // Lowe's ratio test
const int   RANSAC_ITERS     = 3000;
const float RANSAC_THRESHOLD = 4.0f;   // pixels, inlier if closer than this
const int   MIN_MATCHES      = 10;
const int   MAX_CANVAS       = 8000;   // just a safety limit

// ---------------------------- small types ----------------------------
struct Image {
    int w = 0;
    int h = 0;
    vector<unsigned char> pix; // RGBRGBRGB...
};

struct KeyPoint {
    int x = 0;
    int y = 0;
    float score = 0.0f;
    vector<float> desc;
};

struct Match {
    int a = 0; // index in first list
    int b = 0; // index in second list
    float dist = 0.0f;
};

struct Pt {
    double x = 0.0;
    double y = 0.0;
};

// ---------------------------- PPM files ----------------------------
Image readPPM(string filename) {
    Image img;
    ifstream f(filename.c_str(), ios::binary);
    if (!f) {
        cout << "could not open " << filename << endl;
        return img;
    }
    string magic;
    int maxval = 255;
    f >> magic >> img.w >> img.h >> maxval;
    if (magic != "P6") {
        cout << "sorry, only P6 ppm is supported: " << filename << endl;
        img.w = 0;
        img.h = 0;
        return img;
    }
    f.get(); // the single whitespace after the header
    img.pix.resize((size_t)img.w * img.h * 3);
    f.read((char*)img.pix.data(), img.pix.size());
    return img;
}

void writePPM(string filename, const Image& img) {
    ofstream f(filename.c_str(), ios::binary);
    f << "P6\n" << img.w << " " << img.h << "\n255\n";
    f.write((const char*)img.pix.data(), img.pix.size());
}

// ---------------------------- helpers ----------------------------
vector<float> toGray(const Image& im) {
    vector<float> g((size_t)im.w * im.h, 0.0f);
    for (int y = 0; y < im.h; y++) {
        for (int x = 0; x < im.w; x++) {
            size_t i = ((size_t)y * im.w + x) * 3;
            float r = im.pix[i + 0];
            float gg = im.pix[i + 1];
            float b = im.pix[i + 2];
            g[(size_t)y * im.w + x] = 0.299f * r + 0.587f * gg + 0.114f * b;
        }
    }
    return g;
}

// ------------------------- step 1: Harris corners ----------------------------
vector<KeyPoint> detectKeypoints(const vector<float>& gray, int w, int h) {
    // gradients with a Sobel filter
    vector<float> Ix((size_t)w * h, 0.0f);
    vector<float> Iy((size_t)w * h, 0.0f);
    for (int y = 1; y < h - 1; y++) {
        for (int x = 1; x < w - 1; x++) {
            float p1 = gray[(size_t)(y - 1) * w + (x + 1)];
            float p2 = gray[(size_t)y * w + (x + 1)];
            float p3 = gray[(size_t)(y + 1) * w + (x + 1)];
            float p4 = gray[(size_t)(y - 1) * w + (x - 1)];
            float p5 = gray[(size_t)y * w + (x - 1)];
            float p6 = gray[(size_t)(y + 1) * w + (x - 1)];
            float p7 = gray[(size_t)(y + 1) * w + x];
            float p8 = gray[(size_t)(y - 1) * w + x];
            Ix[(size_t)y * w + x] = (p1 + 2 * p2 + p3) - (p4 + 2 * p5 + p6);
            Iy[(size_t)y * w + x] = (p3 + 2 * p7 + p6) - (p1 + 2 * p8 + p4);
        }
    }

    // Harris response: R = det(M) - k * trace(M)^2
    vector<float> R((size_t)w * h, 0.0f);
    float maxR = 0.0f;
    for (int y = 2; y < h - 2; y++) {
        for (int x = 2; x < w - 2; x++) {
            float sxx = 0, syy = 0, sxy = 0;
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    size_t idx = (size_t)(y + dy) * w + (x + dx);
                    float gx = Ix[idx];
                    float gy = Iy[idx];
                    sxx += gx * gx;
                    syy += gy * gy;
                    sxy += gx * gy;
                }
            }
            float det = sxx * syy - sxy * sxy;
            float tr = sxx + syy;
            float r = det - HARRIS_K * tr * tr;
            R[(size_t)y * w + x] = r;
            if (r > maxR) maxR = r;
        }
    }

    // keep local maxima above the threshold
    float thresh = maxR * HARRIS_THRESHOLD;
    vector<KeyPoint> found;
    for (int y = PATCH_RADIUS + 1; y < h - PATCH_RADIUS - 1; y++) {
        for (int x = PATCH_RADIUS + 1; x < w - PATCH_RADIUS - 1; x++) {
            float r = R[(size_t)y * w + x];
            if (r < thresh) continue;
            bool isMax = true;
            for (int dy = -1; dy <= 1 && isMax; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    if (dx == 0 && dy == 0) continue;
                    if (R[(size_t)(y + dy) * w + (x + dx)] > r) { isMax = false; break; }
                }
            }
            if (!isMax) continue;
            KeyPoint kp;
            kp.x = x;
            kp.y = y;
            kp.score = r;
            found.push_back(kp);
        }
    }

    sort(found.begin(), found.end(),
         [](const KeyPoint& p, const KeyPoint& q) { return p.score > q.score; });
    if ((int)found.size() > MAX_KEYPOINTS) found.resize(MAX_KEYPOINTS);
    return found;
}

// descriptor = the grey patch around the corner, normalised
void describeKeypoints(const vector<float>& g, int w, int h, vector<KeyPoint>& kps) {
    int side = 2 * PATCH_RADIUS + 1;
    for (size_t k = 0; k < kps.size(); k++) {
        int cx = kps[k].x;
        int cy = kps[k].y;
        vector<float> raw;
        raw.reserve(side * side);
        float mean = 0.0f;
        for (int dy = -PATCH_RADIUS; dy <= PATCH_RADIUS; dy++) {
            for (int dx = -PATCH_RADIUS; dx <= PATCH_RADIUS; dx++) {
                float v = g[(size_t)(cy + dy) * w + (cx + dx)];
                raw.push_back(v);
                mean += v;
            }
        }
        mean /= raw.size();
        float var = 0.0f;
        for (size_t i = 0; i < raw.size(); i++) {
            float d = raw[i] - mean;
            var += d * d;
        }
        var /= raw.size();
        float sd = sqrt(var) + 1e-6f;
        for (size_t i = 0; i < raw.size(); i++) {
            raw[i] = (raw[i] - mean) / sd;
        }
        kps[k].desc = raw;
    }
}

// ---------------------------- matching ----------------------------
float descriptorDistance(const vector<float>& a, const vector<float>& b) {
    float sum = 0.0f;
    for (size_t i = 0; i < a.size(); i++) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sum; // sum of squared differences
}

vector<Match> matchFeatures(const vector<KeyPoint>& A, const vector<KeyPoint>& B) {
    vector<Match> raw;
    if (A.empty() || B.empty()) return raw;

    // ratio test for every point in A
    for (size_t i = 0; i < A.size(); i++) {
        float best = 1e30f, second = 1e30f;
        int bestj = -1;
        for (size_t j = 0; j < B.size(); j++) {
            float d = descriptorDistance(A[i].desc, B[j].desc);
            if (d < best) {
                second = best;
                best = d;
                bestj = (int)j;
            } else if (d < second) {
                second = d;
            }
        }
        if (bestj >= 0 && best < MATCH_RATIO * second) {
            Match m;
            m.a = (int)i;
            m.b = bestj;
            m.dist = best;
            raw.push_back(m);
        }
    }

    // keep only mutual best matches (helps drop bad ones)
    vector<Match> good;
    for (size_t k = 0; k < raw.size(); k++) {
        int bi = raw[k].b;
        float best = 1e30f;
        int besti = -1;
        for (size_t i = 0; i < A.size(); i++) {
            float d = descriptorDistance(A[i].desc, B[bi].desc);
            if (d < best) { best = d; besti = (int)i; }
        }
        if (besti == raw[k].a) good.push_back(raw[k]);
    }

    sort(good.begin(), good.end(),
         [](const Match& p, const Match& q) { return p.dist < q.dist; });
    return good;
}

// ------------------------ linear algebra (small) ------------------------
// solve n x n system with Gaussian elimination
bool solveLinear(vector<vector<double> > M, vector<double> rhs, vector<double>& x) {
    int n = (int)rhs.size();
    for (int col = 0; col < n; col++) {
        int piv = col;
        for (int r = col + 1; r < n; r++)
            if (fabs(M[r][col]) > fabs(M[piv][col])) piv = r;
        if (fabs(M[piv][col]) < 1e-12) return false;
        swap(M[col], M[piv]);
        swap(rhs[col], rhs[piv]);
        for (int r = 0; r < n; r++) {
            if (r == col) continue;
            double factor = M[r][col] / M[col][col];
            for (int c = col; c < n; c++) M[r][c] -= factor * M[col][c];
            rhs[r] -= factor * rhs[col];
        }
    }
    x.resize(n);
    for (int i = 0; i < n; i++) x[i] = rhs[i] / M[i][i];
    return true;
}

void applyH(const double H[9], double x, double y, double& u, double& v) {
    double w = H[6] * x + H[7] * y + H[8];
    if (fabs(w) < 1e-12) w = 1e-12;
    u = (H[0] * x + H[1] * y + H[2]) / w;
    v = (H[3] * x + H[4] * y + H[5]) / w;
}

void invertH(const double H[9], double Hi[9]) {
    double det = H[0] * (H[4] * H[8] - H[5] * H[7])
               - H[1] * (H[3] * H[8] - H[5] * H[6])
               + H[2] * (H[3] * H[7] - H[4] * H[6]);
    if (fabs(det) < 1e-12) det = 1e-12;
    Hi[0] = (H[4] * H[8] - H[5] * H[7]) / det;
    Hi[1] = (H[2] * H[7] - H[1] * H[8]) / det;
    Hi[2] = (H[1] * H[5] - H[2] * H[4]) / det;
    Hi[3] = (H[5] * H[6] - H[3] * H[8]) / det;
    Hi[4] = (H[0] * H[8] - H[2] * H[6]) / det;
    Hi[5] = (H[2] * H[3] - H[0] * H[5]) / det;
    Hi[6] = (H[3] * H[7] - H[4] * H[6]) / det;
    Hi[7] = (H[1] * H[6] - H[0] * H[7]) / det;
    Hi[8] = (H[0] * H[4] - H[1] * H[3]) / det;
}

// 3x3 matrix multiply: C = A * B (row major)
void mul3x3(const double A[9], const double B[9], double C[9]) {
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            double s = 0.0;
            for (int k = 0; k < 3; k++) s += A[i * 3 + k] * B[k * 3 + j];
            C[i * 3 + j] = s;
        }
    }
}

// Hartley normalisation: move points to the origin and scale them.
// Without this the DLT is badly conditioned and the matrix is useless.
void normalizePoints(const vector<Pt>& p, double T[9], vector<Pt>& pn) {
    int n = (int)p.size();
    double cx = 0, cy = 0;
    for (int i = 0; i < n; i++) { cx += p[i].x; cy += p[i].y; }
    cx /= n; cy /= n;
    double d = 0;
    for (int i = 0; i < n; i++)
        d += sqrt((p[i].x - cx) * (p[i].x - cx) + (p[i].y - cy) * (p[i].y - cy));
    d /= n;
    double s = d > 1e-12 ? sqrt(2.0) / d : 1.0;
    T[0] = s; T[1] = 0; T[2] = -s * cx;
    T[3] = 0; T[4] = s; T[5] = -s * cy;
    T[6] = 0; T[7] = 0; T[8] = 1;
    pn.resize(n);
    for (int i = 0; i < n; i++) {
        pn[i].x = s * (p[i].x - cx);
        pn[i].y = s * (p[i].y - cy);
    }
}

// homography from exactly 4 correspondences (direct linear transform)
bool homographyFrom4(const vector<Pt>& src, const vector<Pt>& dst, double H[9]) {
    double Ts[9], Td[9];
    vector<Pt> sn, dn;
    normalizePoints(src, Ts, sn);
    normalizePoints(dst, Td, dn);

    vector<vector<double> > M(8, vector<double>(8, 0.0));
    vector<double> rhs(8, 0.0);
    for (int i = 0; i < 4; i++) {
        double x = sn[i].x, y = sn[i].y;
        double u = dn[i].x, v = dn[i].y;
        int r = 2 * i;
        M[r][0] = x; M[r][1] = y; M[r][2] = 1;
        M[r][6] = -x * u; M[r][7] = -y * u;
        rhs[r] = u;
        M[r + 1][3] = x; M[r + 1][4] = y; M[r + 1][5] = 1;
        M[r + 1][6] = -x * v; M[r + 1][7] = -y * v;
        rhs[r + 1] = v;
    }
    vector<double> h;
    if (!solveLinear(M, rhs, h)) return false;
    double Hn[9];
    for (int i = 0; i < 8; i++) Hn[i] = h[i];
    Hn[8] = 1.0;

    // undo the normalisation: H = Td^-1 * Hn * Ts
    double TdInv[9];
    invertH(Td, TdInv);
    double tmp[9];
    mul3x3(Hn, Ts, tmp);
    mul3x3(TdInv, tmp, H);
    return true;
}

// least squares homography over many points
bool refineHomography(const vector<Pt>& src, const vector<Pt>& dst, double H[9]) {
    int n = (int)src.size();
    if (n < 4) return false;
    double Ts[9], Td[9];
    vector<Pt> sn, dn;
    normalizePoints(src, Ts, sn);
    normalizePoints(dst, Td, dn);

    vector<vector<double> > M(8, vector<double>(8, 0.0));
    vector<double> rhs(8, 0.0);
    for (int i = 0; i < n; i++) {
        double x = sn[i].x, y = sn[i].y;
        double u = dn[i].x, v = dn[i].y;
        double a1[8] = {x, y, 1, 0, 0, 0, -x * u, -y * u};
        double a2[8] = {0, 0, 0, x, y, 1, -x * v, -y * v};
        for (int p = 0; p < 8; p++) {
            rhs[p] += a1[p] * u + a2[p] * v;
            for (int q = 0; q < 8; q++) {
                M[p][q] += a1[p] * a1[q] + a2[p] * a2[q];
            }
        }
    }
    vector<double> h;
    if (!solveLinear(M, rhs, h)) return false;
    double Hn[9];
    for (int i = 0; i < 8; i++) Hn[i] = h[i];
    Hn[8] = 1.0;

    double TdInv[9];
    invertH(Td, TdInv);
    double tmp[9];
    mul3x3(Hn, Ts, tmp);
    mul3x3(TdInv, tmp, H);
    return true;
}

int countInliers(const double H[9], const vector<Pt>& src, const vector<Pt>& dst) {
    int count = 0;
    for (size_t i = 0; i < src.size(); i++) {
        double u, v;
        applyH(H, src[i].x, src[i].y, u, v);
        double dx = u - dst[i].x;
        double dy = v - dst[i].y;
        if (sqrt(dx * dx + dy * dy) < RANSAC_THRESHOLD) count++;
    }
    return count;
}

bool ransacHomography(const vector<Pt>& src, const vector<Pt>& dst, double Hout[9], int& inliers) {
    int n = (int)src.size();
    if (n < 4) return false;
    int bestCount = -1;
    double bestH[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};

    for (int it = 0; it < RANSAC_ITERS; it++) {
        int idx[4];
        for (int k = 0; k < 4; k++) idx[k] = rand() % n;
        bool dup = false;
        for (int k = 0; k < 4 && !dup; k++)
            for (int l = k + 1; l < 4; l++)
                if (idx[k] == idx[l]) dup = true;
        if (dup) continue;

        vector<Pt> s(4), d(4);
        for (int k = 0; k < 4; k++) { s[k] = src[idx[k]]; d[k] = dst[idx[k]]; }
        double H[9];
        if (!homographyFrom4(s, d, H)) continue;
        int c = countInliers(H, src, dst);
        if (c > bestCount) {
            bestCount = c;
            for (int i = 0; i < 9; i++) bestH[i] = H[i];
        }
    }

    if (bestCount < MIN_MATCHES) return false;

    // refine using all the inliers we found
    vector<Pt> inS, inD;
    for (size_t i = 0; i < src.size(); i++) {
        double u, v;
        applyH(bestH, src[i].x, src[i].y, u, v);
        double dx = u - dst[i].x, dy = v - dst[i].y;
        if (sqrt(dx * dx + dy * dy) < RANSAC_THRESHOLD) {
            inS.push_back(src[i]);
            inD.push_back(dst[i]);
        }
    }
    double refined[9];
    if (refineHomography(inS, inD, refined)) {
        for (int i = 0; i < 9; i++) bestH[i] = refined[i];
    }
    for (int i = 0; i < 9; i++) Hout[i] = bestH[i];
    inliers = countInliers(bestH, src, dst);
    return true;
}

// ------------------------ match two photos -> homography ------------------------
// returns H that maps image B coordinates into image A coordinates
bool computePairHomography(const Image& A, const Image& B, double H[9], int& inliers) {
    vector<float> gA = toGray(A);
    vector<float> gB = toGray(B);

    vector<KeyPoint> kA = detectKeypoints(gA, A.w, A.h);
    describeKeypoints(gA, A.w, A.h, kA);
    vector<KeyPoint> kB = detectKeypoints(gB, B.w, B.h);
    describeKeypoints(gB, B.w, B.h, kB);
    cout << "  corners: " << kA.size() << " and " << kB.size() << endl;

    vector<Match> matches = matchFeatures(kA, kB);
    cout << "  matches: " << matches.size() << endl;
    if ((int)matches.size() < MIN_MATCHES) return false;

    vector<Pt> src, dst; // src = B points, dst = A points
    for (size_t i = 0; i < matches.size(); i++) {
        Pt s, d;
        s.x = kB[matches[i].b].x;
        s.y = kB[matches[i].b].y;
        d.x = kA[matches[i].a].x;
        d.y = kA[matches[i].a].y;
        src.push_back(s);
        dst.push_back(d);
    }
    return ransacHomography(src, dst, H, inliers);
}

// ------------------------ warp all photos onto one canvas ------------------------
// bounding box of the first "count" photos once warped into the reference frame
void canvasBounds(const vector<Image>& imgs, const vector<vector<double> >& Hs, size_t count,
                  double& minX, double& maxX, double& minY, double& maxY) {
    minX = minY = 1e18;
    maxX = maxY = -1e18;
    for (size_t i = 0; i < count; i++) {
        double cx[4] = {0, (double)imgs[i].w, (double)imgs[i].w, 0};
        double cy[4] = {0, 0, (double)imgs[i].h, (double)imgs[i].h};
        for (int c = 0; c < 4; c++) {
            double u, v;
            applyH(Hs[i].data(), cx[c], cy[c], u, v);
            if (u < minX) minX = u;
            if (u > maxX) maxX = u;
            if (v < minY) minY = v;
            if (v > maxY) maxY = v;
        }
    }
}

// Hs[i] maps image i coordinates into the reference frame (image 0)
Image blendMany(const vector<Image>& imgs, const vector<vector<double> >& Hs) {
    double minX, maxX, minY, maxY;
    canvasBounds(imgs, Hs, imgs.size(), minX, maxX, minY, maxY);

    int shiftX = minX < 0 ? (int)ceil(-minX) : 0;
    int shiftY = minY < 0 ? (int)ceil(-minY) : 0;
    int CW = (int)ceil(maxX) + shiftX;
    int CH = (int)ceil(maxY) + shiftY;
    if (CW <= 0 || CH <= 0 || CW > MAX_CANVAS || CH > MAX_CANVAS) {
        cout << "canvas size looks wrong, giving up" << endl;
        return imgs[0];
    }

    vector<float> sum((size_t)CW * CH * 3, 0.0f);
    vector<unsigned char> cnt((size_t)CW * CH, 0);

    for (size_t i = 0; i < imgs.size(); i++) {
        double Hi[9];
        invertH(Hs[i].data(), Hi);
        const Image& im = imgs[i];
        for (int y = 0; y < CH; y++) {
            for (int x = 0; x < CW; x++) {
                double bx, by;
                applyH(Hi, x - shiftX, y - shiftY, bx, by);
                if (bx < 0 || bx >= im.w - 1 || by < 0 || by >= im.h - 1) continue;
                int x0 = (int)floor(bx), y0 = (int)floor(by);
                double fx = bx - x0, fy = by - y0;
                size_t di = ((size_t)y * CW + x) * 3;
                for (int ch = 0; ch < 3; ch++) {
                    double v00 = im.pix[((size_t)y0 * im.w + x0) * 3 + ch];
                    double v10 = im.pix[((size_t)y0 * im.w + (x0 + 1)) * 3 + ch];
                    double v01 = im.pix[((size_t)(y0 + 1) * im.w + x0) * 3 + ch];
                    double v11 = im.pix[((size_t)(y0 + 1) * im.w + (x0 + 1)) * 3 + ch];
                    double top = v00 * (1 - fx) + v10 * fx;
                    double bot = v01 * (1 - fx) + v11 * fx;
                    sum[di + ch] += (float)(top * (1 - fy) + bot * fy);
                }
                cnt[(size_t)y * CW + x]++;
            }
        }
    }

    Image out;
    out.w = CW;
    out.h = CH;
    out.pix.resize((size_t)CW * CH * 3, 0);
    for (int y = 0; y < CH; y++) {
        for (int x = 0; x < CW; x++) {
            size_t ci = (size_t)y * CW + x;
            if (cnt[ci] == 0) continue;
            size_t di = ci * 3;
            for (int ch = 0; ch < 3; ch++) {
                float val = sum[di + ch] / cnt[ci];
                if (val < 0) val = 0;
                if (val > 255) val = 255;
                out.pix[di + ch] = (unsigned char)(val + 0.5f);
            }
        }
    }
    return out;
}

// ---------------------------- main ----------------------------
int main(int argc, char** argv) {
    if (argc < 4) {
        cout << "usage: ./stitcher img1.ppm img2.ppm [more.ppm ...] result.ppm" << endl;
        return 1;
    }
    srand(12345); // fixed seed so runs are repeatable

    string outName = argv[argc - 1];

    vector<Image> imgs;
    for (int i = 1; i < argc - 1; i++) {
        Image im = readPPM(argv[i]);
        if (im.pix.empty()) {
            cout << "skipping unreadable " << argv[i] << endl;
            continue;
        }
        imgs.push_back(im);
    }
    if (imgs.size() < 2) {
        cout << "need at least two photos" << endl;
        return 1;
    }
    cout << "loaded " << imgs.size() << " photos" << endl;

    // Hs[0] is the reference frame; each next one is the chain so far
    vector<vector<double> > Hs(imgs.size(), vector<double>(9, 0.0));
    Hs[0][0] = 1; Hs[0][4] = 1; Hs[0][8] = 1;

    size_t used = 1;
    for (size_t k = 1; k < imgs.size(); k++) {
        cout << "matching photo " << (k + 1) << " to " << k << endl;
        double Hpair[9];
        int inliers = 0;
        if (!computePairHomography(imgs[k - 1], imgs[k], Hpair, inliers)) {
            cout << "  could not match, stopping here" << endl;
            break;
        }
        cout << "  ransac inliers: " << inliers << endl;
        double Hcur[9];
        mul3x3(Hs[k - 1].data(), Hpair, Hcur); // k -> (k-1) -> ... -> reference
        Hs[k].assign(Hcur, Hcur + 9);

        // guard: if this photo stretches the canvas absurdly, the chained
        // homography has gone bad. Stop and keep the panorama we already have.
        double oMinX, oMaxX, oMinY, oMaxY, nMinX, nMaxX, nMinY, nMaxY;
        canvasBounds(imgs, Hs, k, oMinX, oMaxX, oMinY, oMaxY);
        canvasBounds(imgs, Hs, k + 1, nMinX, nMaxX, nMinY, nMaxY);
        double growW = (nMaxX - nMinX) - (oMaxX - oMinX);
        double growH = (nMaxY - nMinY) - (oMaxY - oMinY);
        if (growW > 2.0 * imgs[k].w || growH > 2.0 * imgs[k].h) {
            cout << "  this photo would blow up the canvas, stopping here" << endl;
            break;
        }
        used++;
    }

    imgs.resize(used);
    Hs.resize(used);
    cout << "blending " << imgs.size() << " photos" << endl;

    Image result = blendMany(imgs, Hs);
    writePPM(outName, result);
    cout << "wrote " << outName << " (" << result.w << "x" << result.h << ")" << endl;
    return 0;
}
