#pragma once
#include <vector>
#include <cmath>
#include <stdexcept>

namespace reteco {

class TimeGrid {
public:
    TimeGrid() = default;

    explicit TimeGrid(const std::vector<double>& times) : times_(times) {
        validate();
    }

    static TimeGrid make_graded(double T, std::size_t num_steps, double r = 1.0) {
        if (!std::isfinite(T) || T <= 0.0) {
            throw std::invalid_argument("TimeGrid: T must be finite and > 0.");
        }
        if (num_steps < 1) {
            throw std::invalid_argument("TimeGrid: num_steps must be >= 1.");
        }
        if (!std::isfinite(r) || r < 1.0) {
            throw std::invalid_argument("TimeGrid: grading exponent r must be finite and >= 1.0.");
        }
        std::vector<double> t(num_steps + 1, 0.0);
        for (std::size_t n = 0; n <= num_steps; ++n) {
            const double ratio = static_cast<double>(n) / static_cast<double>(num_steps);
            t[n] = T * std::pow(ratio, r);
        }
        return TimeGrid(t);
    }

    std::size_t num_steps() const {
        ensure_initialized();
        return times_.size() - 1;
    }

    double time_at(std::size_t n) const {
        ensure_initialized();
        if (n >= times_.size()) {
            throw std::out_of_range("TimeGrid::time_at: index out of range.");
        }
        return times_[n];
    }

    double dt(std::size_t k) const {
        ensure_initialized();
        if (k < 1 || k >= times_.size()) {
            throw std::out_of_range("TimeGrid::dt: step index k must be in [1, num_steps].");
        }
        return times_[k] - times_[k - 1];
    }

    const std::vector<double>& times() const {
        ensure_initialized();
        return times_;
    }

    bool is_uniform(double rtol = 1.0e-12) const {
        ensure_initialized();
        if (times_.size() <= 2) return true;
        const double dt0 = dt(1);
        for (std::size_t k = 2; k < times_.size(); ++k) {
            if (std::abs(dt(k) - dt0) > rtol * dt0) return false;
        }
        return true;
    }

private:
    std::vector<double> times_;

    void ensure_initialized() const {
        if (times_.size() < 2) {
            throw std::runtime_error("TimeGrid: grid is not initialized.");
        }
    }

    void validate() const {
        if (times_.size() < 2) {
            throw std::invalid_argument("TimeGrid: requires at least 2 points.");
        }
        for (double val : times_) {
            if (!std::isfinite(val)) {
                throw std::invalid_argument("TimeGrid: all time points must be finite.");
            }
        }
        if (std::abs(times_.front()) > 1.0e-15) {
            throw std::invalid_argument("TimeGrid: t_0 must be 0.0.");
        }
        for (std::size_t k = 1; k < times_.size(); ++k) {
            if (!(times_[k] > times_[k - 1])) {
                throw std::invalid_argument("TimeGrid: time points must be strictly increasing.");
            }
        }
    }
};

} // namespace reteco