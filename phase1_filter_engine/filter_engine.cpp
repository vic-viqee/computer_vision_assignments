#include <iostream>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>

// ==========================================
// 1. a simple matrix class to hold pixels
// ==========================================
// it is a template so it can store uchar, float, whatever.
// all the pixels live in one long vector, stored row by row.

template <typename T>
class Matrix {
public:
    Matrix() : rows_(0), cols_(0), channels_(1) {}

    Matrix(size_t rows, size_t cols, size_t channels = 1, T fill = T())
        : rows_(rows), cols_(cols), channels_(channels),
          data_(rows * cols * channels, fill) {}

    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
    size_t channels() const { return channels_; }
    size_t size() const { return data_.size(); }

    // pixel (r, c, ch) is stored at position (r * cols + c) * channels + ch
    T& operator()(size_t r, size_t c, size_t ch) {
        return data_[(r * cols_ + c) * channels_ + ch];
    }
    const T& operator()(size_t r, size_t c, size_t ch) const {
        return data_[(r * cols_ + c) * channels_ + ch];
    }

    // shortcut for single channel matrices (grayscale)
    T& operator()(size_t r, size_t c) {
        return (*this)(r, c, 0);
    }
    const T& operator()(size_t r, size_t c) const {
        return (*this)(r, c, 0);
    }

    const std::vector<T>& data() const { return data_; }
    std::vector<T>& data() { return data_; }

private:
    size_t rows_;
    size_t cols_;
    size_t channels_;
    std::vector<T> data_;
};

// ==========================================
// 2. put the result back into pixel range
// ==========================================

template <typename T>
T clamp_val(double val) {
    if (std::is_integral<T>::value) {
        // integer types like uchar only go up to a max, so cut the value down
        double lo = (double)std::numeric_limits<T>::min();
        double hi = (double)std::numeric_limits<T>::max();
        if (val < lo) return (T)lo;
        if (val > hi) return (T)hi;
        return (T)val;
    }
    // floats keep any number, so nothing to clamp
    return (T)val;
}

// ==========================================
// 3. the actual filter (convolution)
// ==========================================
// it slides the kernel over every pixel and sums up
// pixel * kernel weight for its little neighborhood

template <typename T>
Matrix<T> filter2D(const Matrix<T>& src, const Matrix<float>& kernel) {
    size_t rows = src.rows();
    size_t cols = src.cols();
    size_t chans = src.channels();

    size_t kh = kernel.rows();
    size_t kw = kernel.cols();
    int pad_r = (int)(kh / 2);
    int pad_c = (int)(kw / 2);

    Matrix<T> out(rows, cols, chans, T());

    for (size_t r = 0; r < rows; r++) {
        for (size_t c = 0; c < cols; c++) {
            for (size_t ch = 0; ch < chans; ch++) {
                double acc = 0.0;

                // go through the kernel window around this pixel
                for (size_t kr = 0; kr < kh; kr++) {
                    for (size_t kc = 0; kc < kw; kc++) {
                        int rr = (int)r + (int)kr - pad_r;
                        int cc = (int)c + (int)kc - pad_c;

                        // outside the image means zero, so just skip it
                        if (rr >= 0 && rr < (int)rows && cc >= 0 && cc < (int)cols) {
                            acc += (double)src(rr, cc, ch) * (double)kernel(kr, kc);
                        }
                    }
                }
                out(r, c, ch) = clamp_val<T>(acc);
            }
        }
    }
    return out;
}

int main() {
    std::cout << "=== c++ filter engine test ===\n\n";

    // 1) test with unsigned char pixels
    Matrix<unsigned char> img(5, 5, 1, 50);
    img(2, 2) = 220;                    // one bright pixel in the middle

    Matrix<float> box(3, 3, 1, 1.0f / 9.0f);   // 3x3 box blur kernel
    Matrix<unsigned char> uchar_out = filter2D(img, box);

    std::cout << "uchar center before: " << (int)img(2, 2) << "\n";
    std::cout << "uchar center after box blur: " << (int)uchar_out(2, 2) << "\n\n";

    // 2) test with float pixels
    Matrix<float> fimg(5, 5, 1, 0.1f);
    fimg(2, 2) = 1.0f;

    Matrix<float> float_out = filter2D(fimg, box);

    std::cout << "float center before: " << fimg(2, 2) << "\n";
    std::cout << "float center after box blur: " << float_out(2, 2) << "\n";

    return 0;
}