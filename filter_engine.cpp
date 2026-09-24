#include <iostream>
#include <vector>
#include <cstdint>
#include <cmath>
#include <type_traits>
#include <limits>
#include <algorithm>

// ==========================================
// 1. Generic Matrix Container Template
// ==========================================

template <typename T>
class Matrix {
private:
    size_t rows_;
    size_t cols_;
    size_t channels_;
    std::vector<T> data_;

public:
    Matrix() : rows_(0), cols_(0), channels_(1) {}
    
    Matrix(size_t rows, size_t cols, size_t channels = 1, T initial_val = T())
        : rows_(rows), cols_(cols), channels_(channels), data_(rows * cols * channels, initial_val) {}

    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
    size_t channels() const { return channels_; }
    size_t size() const { return data_.size(); }

    const std::vector<T>& data() const { return data_; }
    std::vector<T>& data() { return data_; }

    // 2D Row-Major Indexing: (y * width + x) * channels
    T& operator()(size_t r, size_t c) {
        return data_[(r * cols_ + c) * channels_];
    }

    const T& operator()(size_t r, size_t c) const {
        return data_[(r * cols_ + c) * channels_];
    }

    T& operator()(size_t r, size_t c, size_t ch) {
        return data_[(r * cols_ + c) * channels_ + ch];
    }

    const T& operator()(size_t r, size_t c, size_t ch) const {
        return data_[(r * cols_ + c) * channels_ + ch];
    }

    // STL-Compatible Generic Iterators
    using iterator = typename std::vector<T>::iterator;
    using const_iterator = typename std::vector<T>::const_iterator;

    iterator begin() { return data_.begin(); }
    iterator end() { return data_.end(); }
    const_iterator begin() const { return data_.begin(); }
    const_iterator end() const { return data_.end(); }
};

// ==========================================
// 2. Generic C++ Spatial Filter Engine
// ==========================================

template <typename T>
T clamp_pixel(double val) {
    if constexpr (std::is_integral_v<T>) {
        double min_val = static_cast<double>(std::numeric_limits<T>::min());
        double max_val = static_cast<double>(std::numeric_limits<T>::max());
        return static_cast<T>(std::clamp(val, min_val, max_val));
    } else {
        return static_cast<T>(val);
    }
}

template <typename T, typename K = float>
Matrix<T> filter2D(const Matrix<T>& src, const Matrix<K>& kernel) {
    size_t rows = src.rows();
    size_t cols = src.cols();
    size_t channels = src.channels();
    
    size_t k_rows = kernel.rows();
    size_t k_cols = kernel.cols();
    int pad_r = static_cast<int>(k_rows / 2);
    int pad_c = static_cast<int>(k_cols / 2);

    Matrix<T> dst(rows, cols, channels, T());

    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            for (size_t ch = 0; ch < channels; ++ch) {
                double accumulator = 0.0;

                // Manual Spatial Neighborhood Loop
                for (size_t kr = 0; kr < k_rows; ++kr) {
                    for (size_t kc = 0; kc < k_cols; ++kc) {
                        int in_r = static_cast<int>(r) + static_cast<int>(kr) - pad_r;
                        int in_c = static_cast<int>(c) + static_cast<int>(kc) - pad_c;

                        // Zero-padding boundary check
                        if (in_r >= 0 && in_r < static_cast<int>(rows) &&
                            in_c >= 0 && in_c < static_cast<int>(cols)) {
                            double pixel_val = static_cast<double>(src(in_r, in_c, ch));
                            double kernel_val = static_cast<double>(kernel(kr, kc));
                            accumulator += pixel_val * kernel_val;
                        }
                    }
                }
                dst(r, c, ch) = clamp_pixel<T>(accumulator);
            }
        }
    }
    return dst;
}

int main() {
    std::cout << "=== C++ Generic Filter Engine Verification ===\n";

    // 1. Test Matrix with uint8_t (uchar)
    Matrix<uint8_t> uchar_img(5, 5, 1, 50);
    uchar_img(2, 2) = 220; // Center impulse point

    Matrix<float> box_kernel(3, 3, 1, 1.0f / 9.0f);
    Matrix<uint8_t> uchar_out = filter2D(uchar_img, box_kernel);

    std::cout << "uchar Image (2,2) before filter: " << static_cast<int>(uchar_img(2, 2)) << "\n";
    std::cout << "uchar Image (2,2) after box blur: " << static_cast<int>(uchar_out(2, 2)) << "\n\n";

    // 2. Test Matrix with float
    Matrix<float> float_img(5, 5, 1, 0.1f);
    float_img(2, 2) = 1.0f;

    Matrix<float> float_out = filter2D(float_img, box_kernel);

    std::cout << "float Image (2,2) before filter: " << float_img(2, 2) << "\n";
    std::cout << "float Image (2,2) after box blur: " << float_out(2, 2) << "\n";

    return 0;
}