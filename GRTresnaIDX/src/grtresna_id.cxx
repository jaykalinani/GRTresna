#include "grtresna_reader.hxx"

#include <loop.hxx>

#include <cctk.h>
#include <cctk_Arguments.h>
#include <cctk_Parameters.h>

#include <array>
#include <atomic>
#include <cmath>
#include <limits>
#include <memory>
#include <string>

#ifdef CCTK_MPI
#include <mpi.h>
#endif

namespace GRTresnaIDX {
using namespace Loop;

namespace {
std::unique_ptr<GRTresnaReader> g_reader;
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
  if (!(CCTK_EQUALS(out_of_bounds, "clamp") ||
        CCTK_EQUALS(out_of_bounds, "error"))) {
    CCTK_ERROR("GRTresnaIDX: out_of_bounds must be clamp or error");
  }
  if (!(chi_floor > 0.0)) {
    CCTK_ERROR("GRTresnaIDX: chi_floor must be positive");
  }
}

extern "C" void GRTresnaIDX_InitialData(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_GRTresnaIDX_InitialData;
  DECLARE_CCTK_PARAMETERS;

  ReaderConfig config;
  config.use_source_center = use_source_center;
  config.source_center = {{source_center[0], source_center[1], source_center[2]}};
  config.chi_floor = chi_floor;
  config.verbosity = verbosity;

  const bool need_reload =
      (!g_reader) || (g_cached_filename != std::string(grtresna_filename)) ||
      (g_cached_use_source_center != static_cast<bool>(use_source_center)) ||
      (!same_source_center(g_cached_source_center, source_center)) ||
      (g_cached_chi_floor != static_cast<double>(chi_floor));

  if (need_reload) {
    g_reader.reset(new GRTresnaReader());
    g_reader->load_file(grtresna_filename, config);

    g_cached_filename = grtresna_filename;
    g_cached_use_source_center = use_source_center;
    g_cached_source_center = {{source_center[0], source_center[1], source_center[2]}};
    g_cached_chi_floor = chi_floor;
  }

  const auto *reader = g_reader.get();
  const InterpolationMethod interp_method =
      CCTK_EQUALS(interpolation, "nearest") ? InterpolationMethod::nearest
                                            : InterpolationMethod::trilinear;
  const OutOfBoundsPolicy oob_policy =
      CCTK_EQUALS(out_of_bounds, "error") ? OutOfBoundsPolicy::error
                                          : OutOfBoundsPolicy::clamp;

  std::atomic<bool> sample_failed(false);

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
        const bool ok =
            reader->sample_adm(p.x, p.y, p.z, interp_method, oob_policy, s);
        if (!ok) {
          sample_failed.store(true, std::memory_order_relaxed);
          return;
        }

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

        alp(p.I) = s.alp;
        betax(p.I) = s.betax;
        betay(p.I) = s.betay;
        betaz(p.I) = s.betaz;

        if (sanity_checks) {
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

  if (read_matter && !reader->has_matter_data() && CCTK_MyProc(cctkGH) == 0) {
    CCTK_VWARN(CCTK_WARN_ALERT,
               "GRTresnaIDX: read_matter=yes, but no matter components found in source file");
  }

  if (sanity_checks) {
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

    double global_min_det = local_min_det;
    double global_max_det = local_max_det;
    double global_min_trk = local_min_trk;
    double global_max_trk = local_max_trk;
    double global_min_alp = local_min_alp;
    double global_max_alp = local_max_alp;

#ifdef CCTK_MPI
    MPI_Allreduce(&local_min_det, &global_min_det, 1, MPI_DOUBLE, MPI_MIN,
                  MPI_COMM_WORLD);
    MPI_Allreduce(&local_max_det, &global_max_det, 1, MPI_DOUBLE, MPI_MAX,
                  MPI_COMM_WORLD);
    MPI_Allreduce(&local_min_trk, &global_min_trk, 1, MPI_DOUBLE, MPI_MIN,
                  MPI_COMM_WORLD);
    MPI_Allreduce(&local_max_trk, &global_max_trk, 1, MPI_DOUBLE, MPI_MAX,
                  MPI_COMM_WORLD);
    MPI_Allreduce(&local_min_alp, &global_min_alp, 1, MPI_DOUBLE, MPI_MIN,
                  MPI_COMM_WORLD);
    MPI_Allreduce(&local_max_alp, &global_max_alp, 1, MPI_DOUBLE, MPI_MAX,
                  MPI_COMM_WORLD);
#endif

    if (CCTK_MyProc(cctkGH) == 0) {
      CCTK_VINFO(
          "GRTresnaIDX sanity: det(g) in [%e, %e], tr(K) in [%e, %e], alp in [%e, %e]",
          global_min_det, global_max_det, global_min_trk, global_max_trk,
          global_min_alp, global_max_alp);
    }
  }
}

} // namespace GRTresnaIDX
