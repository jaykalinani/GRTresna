#include "grtresna_reader.hxx"

#include <loop.hxx>

#include <cctk.h>
#include <cctk_Arguments.h>
#include <cctk_Parameters.h>

#include <array>
#include <atomic>
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <mutex>
#include <string>

namespace GRTresnaIDX {
using namespace Loop;

namespace {
std::unique_ptr<GRTresnaReader> g_reader;
std::mutex g_reader_mutex;
std::string g_cached_filename;
bool g_cached_use_source_center = false;
std::array<double, 3> g_cached_source_center{{0.0, 0.0, 0.0}};
double g_cached_chi_floor = -1.0;

inline bool same_source_center(const std::array<double, 3> &a,
                               const CCTK_REAL *b) {
  for (int d = 0; d < 3; ++d) {
    if (a[d] != b[d]) {
      return false;
    }
  }
  return true;
}

void ensure_reader_loaded(const char *filename, const bool use_center,
                          const CCTK_REAL *center, const CCTK_REAL chi_floor,
                          const int reader_verbosity) {
  std::lock_guard<std::mutex> lock(g_reader_mutex);

  ReaderConfig config;
  config.use_source_center = use_center;
  config.source_center = {{center[0], center[1], center[2]}};
  config.chi_floor = chi_floor;
  config.verbosity = reader_verbosity;

  const bool need_reload =
      (!g_reader) || (g_cached_filename != std::string(filename)) ||
      (g_cached_use_source_center != use_center) ||
      (!same_source_center(g_cached_source_center, center)) ||
      (g_cached_chi_floor != static_cast<double>(chi_floor));

  if (need_reload) {
    g_reader.reset(new GRTresnaReader());
    g_reader->load_file(filename, config);

    g_cached_filename = filename;
    g_cached_use_source_center = use_center;
    g_cached_source_center = {{center[0], center[1], center[2]}};
    g_cached_chi_floor = chi_floor;
  }
}

inline double det3(const ADMSample &s) {
  return s.gxx * (s.gyy * s.gzz - s.gyz * s.gyz) -
         s.gxy * (s.gxy * s.gzz - s.gyz * s.gxz) +
         s.gxz * (s.gxy * s.gyz - s.gyy * s.gxz);
}

inline bool inverse3(const ADMSample &s, const double det, double &igxx,
                     double &igxy, double &igxz, double &igyy, double &igyz,
                     double &igzz) {
  if (!std::isfinite(det) || std::abs(det) < 1.0e-30) {
    return false;
  }
  igxx = (s.gyy * s.gzz - s.gyz * s.gyz) / det;
  igxy = (s.gxz * s.gyz - s.gxy * s.gzz) / det;
  igxz = (s.gxy * s.gyz - s.gxz * s.gyy) / det;
  igyy = (s.gxx * s.gzz - s.gxz * s.gxz) / det;
  igyz = (s.gxy * s.gxz - s.gxx * s.gyz) / det;
  igzz = (s.gxx * s.gyy - s.gxy * s.gxy) / det;
  return true;
}

inline bool determinant_lapse(const ADMSample &s, double &alp_det) {
  const double det = det3(s);
  if (!(std::isfinite(det) && det > 0.0)) {
    return false;
  }

  const double psi = std::pow(det, 1.0 / 12.0);
  if (!(std::isfinite(psi) && psi > 0.0)) {
    return false;
  }

  alp_det = 1.0 / psi;
  return std::isfinite(alp_det);
}

template <typename T>
CCTK_HOST CCTK_ATTRIBUTE_ALWAYS_INLINE inline T
calc_avg_c2v_order2(const GF3D2<const T> &gf, const PointDesc &p) {
  T gf_avg = 0;
  for (int dk = 0; dk < 2; ++dk) {
    for (int dj = 0; dj < 2; ++dj) {
      for (int di = 0; di < 2; ++di) {
        gf_avg += gf(p.I - p.DI[0] * di - p.DI[1] * dj - p.DI[2] * dk);
      }
    }
  }
  return gf_avg * T(0.125);
}

template <typename T>
CCTK_HOST CCTK_ATTRIBUTE_ALWAYS_INLINE inline T
calc_avg_c2v_order4(const GF3D2<const T> &gf, const PointDesc &p) {
  T gf_avg = 0;
  const std::array<T, 4> wt{{-1 / T(16), +9 / T(16), +9 / T(16),
                             -1 / T(16)}};

  // Symmetrized fourth-order cell-center to vertex interpolation copied into
  // this thorn from the AsterX/AsterUtils stencil pattern.
  for (int i = 0; i < 3; ++i) {
    const int j = (i == 0) ? 1 : ((i == 1) ? 2 : 0);
    const int k = (i == 0) ? 2 : ((i == 1) ? 0 : 1);

    std::array<std::array<T, 4>, 4> gf_i{};
    for (int dk = 0; dk < 4; ++dk) {
      for (int dj = 0; dj < 4; ++dj) {
        for (int di = 0; di < 4; ++di) {
          gf_i[dk][dj] += wt[di] * gf(p.I + p.DI[i] * (di - 2) +
                                      p.DI[j] * (dj - 2) +
                                      p.DI[k] * (dk - 2));
        }
      }
    }

    std::array<T, 4> gf_j{};
    for (int dk = 0; dk < 4; ++dk) {
      for (int dj = 0; dj < 4; ++dj) {
        gf_j[dk] += wt[dj] * gf_i[dk][dj];
      }
    }

    for (int dk = 0; dk < 4; ++dk) {
      gf_avg += wt[dk] * gf_j[dk];
    }
  }

  return gf_avg / T(3);
}

template <typename T>
CCTK_HOST CCTK_ATTRIBUTE_ALWAYS_INLINE inline T
calc_avg_c2v(const GF3D2<const T> &gf, const PointDesc &p, const int order) {
  return order == 4 ? calc_avg_c2v_order4(gf, p)
                    : calc_avg_c2v_order2(gf, p);
}

inline int preferred_source_level_for_grid(const GRTresnaReader *reader,
                                           const int carpet_level,
                                           const int extra_coarse) {
  if (reader == nullptr) {
    return -1;
  }

  const int requested = carpet_level - std::max(0, extra_coarse);
  return requested >= 0 && requested < reader->num_levels() ? requested : -1;
}

template <int CI, int CJ, int CK>
inline void grid_all_coordinate_bounds(const GridDescBase &grid, double &xmin,
                                       double &ymin, double &zmin,
                                       double &xmax, double &ymax,
                                       double &zmax) {
  vect<int, dim> all_min, all_max;
  grid.box_all<CI, CJ, CK>(grid.nghostzones, all_min, all_max);

  const int centering[3] = {CI, CJ, CK};
  double lo[3];
  double hi[3];
  for (int d = 0; d < 3; ++d) {
    const double offset = centering[d] == 0 ? 0.5 : 0.0;
    const int imin = all_min[d];
    const int imax = all_max[d] - 1;
    lo[d] = grid.x0[d] + (grid.lbnd[d] + imin - offset) * grid.dx[d];
    hi[d] = grid.x0[d] + (grid.lbnd[d] + imax - offset) * grid.dx[d];
    if (hi[d] < lo[d]) {
      std::swap(lo[d], hi[d]);
    }
  }

  xmin = lo[0];
  ymin = lo[1];
  zmin = lo[2];
  xmax = hi[0];
  ymax = hi[1];
  zmax = hi[2];
}
} // namespace

extern "C" void GRTresnaIDX_ParamCheck(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_GRTresnaIDX_ParamCheck;
  DECLARE_CCTK_PARAMETERS;

  const bool enabled = CCTK_EQUALS(initial_data, "GRTresnaIDX") ||
                       CCTK_EQUALS(initial_lapse, "GRTresnaIDX") ||
                       CCTK_EQUALS(initial_shift, "GRTresnaIDX");
  if (!enabled) {
    return;
  }

  if (grtresna_filename[0] == '\0') {
    CCTK_ERROR("GRTresnaIDX: grtresna_filename must be set");
  }
  if (!(CCTK_EQUALS(grtresna_format, "auto") ||
        CCTK_EQUALS(grtresna_format, "grtresna_hdf5"))) {
    CCTK_ERROR("GRTresnaIDX: unsupported grtresna_format");
  }
  if (!(CCTK_EQUALS(interpolation, "nearest") ||
        CCTK_EQUALS(interpolation, "trilinear"))) {
    CCTK_ERROR("GRTresnaIDX: interpolation must be nearest or trilinear");
  }
  if (!(CCTK_EQUALS(adm_population_method, "direct") ||
        CCTK_EQUALS(adm_population_method, "cell-centered-c2v"))) {
    CCTK_ERROR(
        "GRTresnaIDX: adm_population_method must be direct or cell-centered-c2v");
  }
  if (!(c2v_order == 2 || c2v_order == 4)) {
    CCTK_ERROR("GRTresnaIDX: c2v_order must be either 2 or 4");
  }
  if (!(CCTK_EQUALS(out_of_bounds, "clamp") ||
        CCTK_EQUALS(out_of_bounds, "error"))) {
    CCTK_ERROR("GRTresnaIDX: out_of_bounds must be clamp or error");
  }
  if (!(CCTK_EQUALS(lapse_profile, "file") ||
        CCTK_EQUALS(lapse_profile, "determinant-psi"))) {
    CCTK_ERROR("GRTresnaIDX: lapse_profile must be file or determinant-psi");
  }
  if (!(chi_floor > 0.0)) {
    CCTK_ERROR("GRTresnaIDX: chi_floor must be positive");
  }
}

extern "C" void GRTresnaIDX_InitialData(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_GRTresnaIDX_InitialData;
  DECLARE_CCTK_PARAMETERS;

  const bool write_metric_curv = CCTK_EQUALS(initial_data, "GRTresnaIDX");
  const bool write_lapse = CCTK_EQUALS(initial_lapse, "GRTresnaIDX");
  const bool write_shift = CCTK_EQUALS(initial_shift, "GRTresnaIDX");
  if (!(write_metric_curv || write_lapse || write_shift)) {
    return;
  }

  ensure_reader_loaded(grtresna_filename, static_cast<bool>(use_source_center),
                       source_center, chi_floor, verbosity);

  const auto *reader = g_reader.get();
  const InterpolationMethod interp_method =
      CCTK_EQUALS(interpolation, "nearest") ? InterpolationMethod::nearest
                                            : InterpolationMethod::trilinear;
  const OutOfBoundsPolicy oob_policy =
      CCTK_EQUALS(out_of_bounds, "error") ? OutOfBoundsPolicy::error
                                          : OutOfBoundsPolicy::clamp;
  const bool use_determinant_lapse =
      write_lapse && CCTK_EQUALS(lapse_profile, "determinant-psi");
  const bool sample_metric_curv = write_metric_curv || use_determinant_lapse;
  const bool sample_file_lapse = write_lapse && !use_determinant_lapse;

  double xmin = std::numeric_limits<double>::infinity();
  double ymin = std::numeric_limits<double>::infinity();
  double zmin = std::numeric_limits<double>::infinity();
  double xmax = -std::numeric_limits<double>::infinity();
  double ymax = -std::numeric_limits<double>::infinity();
  double zmax = -std::numeric_limits<double>::infinity();
  grid_all_coordinate_bounds<0, 0, 0>(grid, xmin, ymin, zmin, xmax, ymax,
                                      zmax);
  const int preferred_source_level = preferred_source_level_for_grid(
      reader, static_cast<int>(grid.level), static_cast<int>(extra_coarse_levels));
  if (verbosity >= 2 && CCTK_MyProc(cctkGH) == 0) {
    CCTK_VINFO("GRTresnaIDX direct fill on CarpetX level %d prefers source level %d",
               int(grid.level), preferred_source_level);
  }
  reader->prefetch_region(xmin, ymin, zmin, xmax, ymax, zmax, interp_method,
                          preferred_source_level);

  std::atomic<bool> sample_failed(false);
  const bool do_sanity_checks = sanity_checks && write_metric_curv;

  double local_min_det = std::numeric_limits<double>::infinity();
  double local_max_det = -std::numeric_limits<double>::infinity();
  double local_min_trk = std::numeric_limits<double>::infinity();
  double local_max_trk = -std::numeric_limits<double>::infinity();
  double local_min_alp = std::numeric_limits<double>::infinity();
  double local_max_alp = -std::numeric_limits<double>::infinity();

  grid.loop_all<0, 0, 0>(
      grid.nghostzones, [=, &sample_failed, &local_min_det, &local_max_det,
                         &local_min_trk, &local_max_trk, &local_min_alp,
                         &local_max_alp](const PointDesc &p)
                             CCTK_ATTRIBUTE_ALWAYS_INLINE {
        if (sample_failed.load(std::memory_order_relaxed)) {
          return;
        }

        ADMSample s{};
        const bool ok = reader->sample_adm(p.x, p.y, p.z, interp_method,
                                           oob_policy, s, sample_metric_curv,
                                           sample_file_lapse, write_shift,
                                           preferred_source_level);
        if (!ok) {
          sample_failed.store(true, std::memory_order_relaxed);
          return;
        }

        if (write_metric_curv) {
          gxx(p.I) = s.gxx;
          gxy(p.I) = s.gxy;
          gxz(p.I) = s.gxz;
          gyy(p.I) = s.gyy;
          gyz(p.I) = s.gyz;
          gzz(p.I) = s.gzz;

          kxx(p.I) = s.kxx;
          kxy(p.I) = s.kxy;
          kxz(p.I) = s.kxz;
          kyy(p.I) = s.kyy;
          kyz(p.I) = s.kyz;
          kzz(p.I) = s.kzz;
        }

        if (write_lapse) {
          if (use_determinant_lapse && !determinant_lapse(s, s.alp)) {
            sample_failed.store(true, std::memory_order_relaxed);
            return;
          }
          alp(p.I) = s.alp;
        }
        if (write_shift) {
          betax(p.I) = s.betax;
          betay(p.I) = s.betay;
          betaz(p.I) = s.betaz;
        }

        if (do_sanity_checks) {
          const double det = det3(s);
          double trk = std::numeric_limits<double>::quiet_NaN();
          double igxx = 0.0, igxy = 0.0, igxz = 0.0, igyy = 0.0, igyz = 0.0,
                 igzz = 0.0;
          if (inverse3(s, det, igxx, igxy, igxz, igyy, igyz, igzz)) {
            trk = igxx * s.kxx + 2.0 * igxy * s.kxy + 2.0 * igxz * s.kxz +
                  igyy * s.kyy + 2.0 * igyz * s.kyz + igzz * s.kzz;
          }

#pragma omp critical
          {
            if (std::isfinite(det)) {
              local_min_det = std::min(local_min_det, det);
              local_max_det = std::max(local_max_det, det);
            }
            if (std::isfinite(trk)) {
              local_min_trk = std::min(local_min_trk, trk);
              local_max_trk = std::max(local_max_trk, trk);
            }
            if (std::isfinite(s.alp)) {
              local_min_alp = std::min(local_min_alp, s.alp);
              local_max_alp = std::max(local_max_alp, s.alp);
            }
          }
        }
      });

  if (sample_failed.load(std::memory_order_relaxed)) {
    CCTK_ERROR(
        "GRTresnaIDX failed to sample source data. Check file coverage and out_of_bounds policy.");
  }

  if (do_sanity_checks) {
    if (!std::isfinite(local_min_det)) {
      local_min_det = 0.0;
      local_max_det = 0.0;
    }
    if (!std::isfinite(local_min_trk)) {
      local_min_trk = 0.0;
      local_max_trk = 0.0;
    }
    if (!std::isfinite(local_min_alp)) {
      local_min_alp = 0.0;
      local_max_alp = 0.0;
    }

    // This routine is scheduled locally by CarpetX. Do not use raw MPI
    // collectives here: ranks can enter this function for different local
    // components/levels at different times, which deadlocks. Keep this check
    // local; use a separate GLOBAL scheduled routine for true global
    // reductions if they are needed later.
    if (CCTK_MyProc(cctkGH) == 0) {
      CCTK_VINFO(
          "GRTresnaIDX local sanity on level %d: det(g) in [%e, %e], tr(K) in [%e, %e], alp in [%e, %e]",
          int(grid.level), local_min_det, local_max_det, local_min_trk,
          local_max_trk, local_min_alp, local_max_alp);
    }
  }
}

extern "C" void GRTresnaIDX_InitialDataCell(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_GRTresnaIDX_InitialDataCell;
  DECLARE_CCTK_PARAMETERS;

  const bool enabled = CCTK_EQUALS(initial_data, "GRTresnaIDX") ||
                       CCTK_EQUALS(initial_lapse, "GRTresnaIDX") ||
                       CCTK_EQUALS(initial_shift, "GRTresnaIDX");
  if (!enabled) {
    return;
  }

  ensure_reader_loaded(grtresna_filename, static_cast<bool>(use_source_center),
                       source_center, chi_floor, verbosity);

  const auto *reader = g_reader.get();
  const InterpolationMethod interp_method =
      CCTK_EQUALS(interpolation, "nearest") ? InterpolationMethod::nearest
                                            : InterpolationMethod::trilinear;
  const OutOfBoundsPolicy oob_policy =
      CCTK_EQUALS(out_of_bounds, "error") ? OutOfBoundsPolicy::error
                                          : OutOfBoundsPolicy::clamp;
  const bool use_determinant_lapse =
      CCTK_EQUALS(lapse_profile, "determinant-psi");
  const bool sample_file_lapse = !use_determinant_lapse;

  double xmin = std::numeric_limits<double>::infinity();
  double ymin = std::numeric_limits<double>::infinity();
  double zmin = std::numeric_limits<double>::infinity();
  double xmax = -std::numeric_limits<double>::infinity();
  double ymax = -std::numeric_limits<double>::infinity();
  double zmax = -std::numeric_limits<double>::infinity();
  grid_all_coordinate_bounds<1, 1, 1>(grid, xmin, ymin, zmin, xmax, ymax,
                                      zmax);
  const int preferred_source_level = preferred_source_level_for_grid(
      reader, static_cast<int>(grid.level), static_cast<int>(extra_coarse_levels));
  if (verbosity >= 2 && CCTK_MyProc(cctkGH) == 0) {
    CCTK_VINFO(
        "GRTresnaIDX cell-centered fill on CarpetX level %d prefers source level %d",
        int(grid.level), preferred_source_level);
  }
  reader->prefetch_region(xmin, ymin, zmin, xmax, ymax, zmax, interp_method,
                          preferred_source_level);

  std::atomic<bool> sample_failed(false);
  const bool do_sanity_checks = sanity_checks;

  double local_min_det = std::numeric_limits<double>::infinity();
  double local_max_det = -std::numeric_limits<double>::infinity();
  double local_min_trk = std::numeric_limits<double>::infinity();
  double local_max_trk = -std::numeric_limits<double>::infinity();
  double local_min_alp = std::numeric_limits<double>::infinity();
  double local_max_alp = -std::numeric_limits<double>::infinity();

  grid.loop_all<1, 1, 1>(
      grid.nghostzones, [=, &sample_failed, &local_min_det, &local_max_det,
                         &local_min_trk, &local_max_trk, &local_min_alp,
                         &local_max_alp](const PointDesc &p)
                             CCTK_ATTRIBUTE_ALWAYS_INLINE {
        if (sample_failed.load(std::memory_order_relaxed)) {
          return;
        }

        ADMSample s{};
        const bool ok =
            reader->sample_adm(p.x, p.y, p.z, interp_method, oob_policy, s,
                               true, sample_file_lapse, true,
                               preferred_source_level);
        if (!ok) {
          sample_failed.store(true, std::memory_order_relaxed);
          return;
        }

        if (use_determinant_lapse && !determinant_lapse(s, s.alp)) {
          sample_failed.store(true, std::memory_order_relaxed);
          return;
        }

        gxx_cell(p.I) = s.gxx;
        gxy_cell(p.I) = s.gxy;
        gxz_cell(p.I) = s.gxz;
        gyy_cell(p.I) = s.gyy;
        gyz_cell(p.I) = s.gyz;
        gzz_cell(p.I) = s.gzz;

        kxx_cell(p.I) = s.kxx;
        kxy_cell(p.I) = s.kxy;
        kxz_cell(p.I) = s.kxz;
        kyy_cell(p.I) = s.kyy;
        kyz_cell(p.I) = s.kyz;
        kzz_cell(p.I) = s.kzz;

        alp_cell(p.I) = s.alp;
        betax_cell(p.I) = s.betax;
        betay_cell(p.I) = s.betay;
        betaz_cell(p.I) = s.betaz;

        if (do_sanity_checks) {
          const double det = det3(s);
          double trk = std::numeric_limits<double>::quiet_NaN();
          double igxx = 0.0, igxy = 0.0, igxz = 0.0, igyy = 0.0, igyz = 0.0,
                 igzz = 0.0;
          if (inverse3(s, det, igxx, igxy, igxz, igyy, igyz, igzz)) {
            trk = igxx * s.kxx + 2.0 * igxy * s.kxy + 2.0 * igxz * s.kxz +
                  igyy * s.kyy + 2.0 * igyz * s.kyz + igzz * s.kzz;
          }

#pragma omp critical
          {
            if (std::isfinite(det)) {
              local_min_det = std::min(local_min_det, det);
              local_max_det = std::max(local_max_det, det);
            }
            if (std::isfinite(trk)) {
              local_min_trk = std::min(local_min_trk, trk);
              local_max_trk = std::max(local_max_trk, trk);
            }
            if (std::isfinite(s.alp)) {
              local_min_alp = std::min(local_min_alp, s.alp);
              local_max_alp = std::max(local_max_alp, s.alp);
            }
          }
        }
      });

  if (sample_failed.load(std::memory_order_relaxed)) {
    CCTK_ERROR(
        "GRTresnaIDX failed to sample source data at cell centers. Check file coverage and out_of_bounds policy.");
  }

  if (do_sanity_checks) {
    if (!std::isfinite(local_min_det)) {
      local_min_det = 0.0;
      local_max_det = 0.0;
    }
    if (!std::isfinite(local_min_trk)) {
      local_min_trk = 0.0;
      local_max_trk = 0.0;
    }
    if (!std::isfinite(local_min_alp)) {
      local_min_alp = 0.0;
      local_max_alp = 0.0;
    }

    if (CCTK_MyProc(cctkGH) == 0) {
      CCTK_VINFO(
          "GRTresnaIDX cell-centered local sanity on level %d: det(g) in [%e, %e], tr(K) in [%e, %e], alp in [%e, %e]",
          int(grid.level), local_min_det, local_max_det, local_min_trk,
          local_max_trk, local_min_alp, local_max_alp);
    }
  }
}

extern "C" void GRTresnaIDX_InitialDataC2V(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_GRTresnaIDX_InitialDataC2V;
  DECLARE_CCTK_PARAMETERS;

  const int order = c2v_order;
  const int required_ghosts = order == 4 ? 2 : 1;
  for (int d = 0; d < 3; ++d) {
    if (grid.nghostzones[d] < required_ghosts) {
      CCTK_VERROR(
          "GRTresnaIDX c2v_order=%d needs at least %d ghost zones in every direction; direction %d has %d",
          order, required_ghosts, d, grid.nghostzones[d]);
    }
  }

  grid.loop_int<0, 0, 0>(
      grid.nghostzones,
      [=](const PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
        gxx(p.I) = calc_avg_c2v(gxx_cell, p, order);
        gxy(p.I) = calc_avg_c2v(gxy_cell, p, order);
        gxz(p.I) = calc_avg_c2v(gxz_cell, p, order);
        gyy(p.I) = calc_avg_c2v(gyy_cell, p, order);
        gyz(p.I) = calc_avg_c2v(gyz_cell, p, order);
        gzz(p.I) = calc_avg_c2v(gzz_cell, p, order);

        kxx(p.I) = calc_avg_c2v(kxx_cell, p, order);
        kxy(p.I) = calc_avg_c2v(kxy_cell, p, order);
        kxz(p.I) = calc_avg_c2v(kxz_cell, p, order);
        kyy(p.I) = calc_avg_c2v(kyy_cell, p, order);
        kyz(p.I) = calc_avg_c2v(kyz_cell, p, order);
        kzz(p.I) = calc_avg_c2v(kzz_cell, p, order);

        alp(p.I) = calc_avg_c2v(alp_cell, p, order);
        betax(p.I) = calc_avg_c2v(betax_cell, p, order);
        betay(p.I) = calc_avg_c2v(betay_cell, p, order);
        betaz(p.I) = calc_avg_c2v(betaz_cell, p, order);
      });

  if (verbosity > 0 && CCTK_MyProc(cctkGH) == 0) {
    CCTK_VINFO("GRTresnaIDX interpolated cell-centered ADM data to vertices with c2v_order=%d",
               order);
  }
}

extern "C" void GRTresnaIDX_SourceRegridError(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_GRTresnaIDX_SourceRegridError;
  DECLARE_CCTK_PARAMETERS;

  if (!set_source_regrid_error) {
    return;
  }

  ensure_reader_loaded(grtresna_filename, static_cast<bool>(use_source_center),
                       source_center, chi_floor, verbosity);

  const auto *reader = g_reader.get();
  const int offset = std::max(0, static_cast<int>(extra_coarse_levels));
  const int carpet_level = static_cast<int>(grid.level);

  const bool coarse_level = carpet_level < offset;
  const double source_radius = reader->base_domain_radius();
  const bool have_source_radius = source_radius > 0.0 && std::isfinite(source_radius);
  const double coarse_tag_radius =
      coarse_level && have_source_radius
          ? std::ldexp(source_radius, offset - carpet_level - 1)
          : 0.0;

  const int target_source_level = carpet_level - offset + 1;
  const bool have_target_level =
      target_source_level >= 0 && target_source_level < reader->num_levels();

  grid.loop_all<1, 1, 1>(grid.nghostzones, [=](const PointDesc &p)
                             CCTK_ATTRIBUTE_ALWAYS_INLINE {
    bool tag = false;
    if (coarse_level) {
      const double xlo = p.x - 0.5 * p.dx;
      const double xhi = p.x + 0.5 * p.dx;
      const double ylo = p.y - 0.5 * p.dy;
      const double yhi = p.y + 0.5 * p.dy;
      const double zlo = p.z - 0.5 * p.dz;
      const double zhi = p.z + 0.5 * p.dz;
      tag = coarse_tag_radius > 0.0 && xhi >= -coarse_tag_radius &&
            xlo <= coarse_tag_radius && yhi >= -coarse_tag_radius &&
            ylo <= coarse_tag_radius && zhi >= -coarse_tag_radius &&
            zlo <= coarse_tag_radius;
    } else {
      tag = have_target_level &&
            reader->cell_overlaps_level(p.x, p.y, p.z, p.dx, p.dy, p.dz,
                                        target_source_level);
    }

    regrid_error(p.I) = tag ? CCTK_REAL(1.0) : CCTK_REAL(0.0);
  });
}

} // namespace GRTresnaIDX
