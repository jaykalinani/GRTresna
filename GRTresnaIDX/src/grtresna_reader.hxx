#ifndef GRTRESNAIDX_GRTRESNA_READER_HXX
#define GRTRESNAIDX_GRTRESNA_READER_HXX

#include <array>
#include <atomic>
#include <cstdint>
#include <limits>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <hdf5.h>

namespace GRTresnaIDX {

enum class InterpolationMethod { nearest, trilinear };
enum class OutOfBoundsPolicy { clamp, error, asymptotic };

struct ReaderConfig {
  bool use_source_center = false;
  std::array<double, 3> source_center{{0.0, 0.0, 0.0}};
  double chi_floor = 1.0e-14;
  int verbosity = 1;
};

struct ADMSample {
  double gxx, gxy, gxz, gyy, gyz, gzz;
  double kxx, kxy, kxz, kyy, kyz, kzz;
  double alp, betax, betay, betaz;
};

class GRTresnaReader {
public:
  GRTresnaReader();

  void load_file(const std::string &filename, const ReaderConfig &config);
  void prefetch_region(double xmin, double ymin, double zmin, double xmax,
                       double ymax, double zmax,
                       InterpolationMethod method,
                       int preferred_source_level = -1) const;

  bool sample_adm(double x, double y, double z, InterpolationMethod method,
                  OutOfBoundsPolicy oob_policy, ADMSample &out,
                  bool need_metric_curv = true, bool need_lapse = true,
                  bool need_shift = true,
                  int preferred_source_level = -1) const;

  bool is_loaded() const { return loaded_; }
  int num_levels() const { return num_levels_; }
  double base_domain_radius() const;

  bool cell_overlaps_level(double x, double y, double z, double dx, double dy,
                           double dz, int level_idx) const;

private:
  enum class VariableBasis { adm, bssn };

  struct SourceBox {
    std::array<int, 3> lo;
    std::array<int, 3> hi;
    std::array<int, 3> n_with_ghost;
    int nghost;
    std::size_t begin;
    std::size_t end;
    mutable std::atomic<bool> data_loaded;
    mutable std::vector<double> data;

    SourceBox()
        : lo{{0, 0, 0}}, hi{{-1, -1, -1}}, n_with_ghost{{0, 0, 0}},
          nghost(0), begin(0), end(0), data_loaded(false), data() {}

    SourceBox(const SourceBox &other)
        : lo(other.lo), hi(other.hi), n_with_ghost(other.n_with_ghost),
          nghost(other.nghost), begin(other.begin), end(other.end),
          data_loaded(other.data_loaded.load(std::memory_order_acquire)),
          data(other.data) {}

    SourceBox(SourceBox &&other) noexcept
        : lo(other.lo), hi(other.hi), n_with_ghost(other.n_with_ghost),
          nghost(other.nghost), begin(other.begin), end(other.end),
          data_loaded(other.data_loaded.load(std::memory_order_acquire)),
          data(std::move(other.data)) {}

    SourceBox &operator=(const SourceBox &other) {
      if (this != &other) {
        lo = other.lo;
        hi = other.hi;
        n_with_ghost = other.n_with_ghost;
        nghost = other.nghost;
        begin = other.begin;
        end = other.end;
        data = other.data;
        data_loaded.store(other.data_loaded.load(std::memory_order_acquire),
                          std::memory_order_release);
      }
      return *this;
    }

    SourceBox &operator=(SourceBox &&other) noexcept {
      if (this != &other) {
        lo = other.lo;
        hi = other.hi;
        n_with_ghost = other.n_with_ghost;
        nghost = other.nghost;
        begin = other.begin;
        end = other.end;
        data = std::move(other.data);
        data_loaded.store(other.data_loaded.load(std::memory_order_acquire),
                          std::memory_order_release);
      }
      return *this;
    }
  };

  struct SourceLevel {
    double dx;
    std::size_t data_size;
    std::array<int, 3> lo_union;
    std::array<int, 3> hi_union;
    std::array<int, 3> data_lo_union;
    std::array<int, 3> data_hi_union;
    std::array<int, 3> box_size;
    bool uniform_box_size;
    std::unordered_map<std::uint64_t, std::size_t> interior_box_index;
    std::vector<SourceBox> boxes;
  };

  bool loaded_;
  ReaderConfig config_;
  std::string filename_;
  mutable std::mutex data_mutex_;

  std::array<double, 3> center_;
  int num_levels_;
  std::array<int, 3> global_lo_;
  std::array<int, 3> global_hi_;
  int ncomp_;

  VariableBasis basis_;
  std::unordered_map<std::string, int> comp_to_index_;
  std::vector<SourceLevel> levels_;

  // ADM basis indices
  int idx_gxx_, idx_gxy_, idx_gxz_, idx_gyy_, idx_gyz_, idx_gzz_;
  int idx_kxx_, idx_kxy_, idx_kxz_, idx_kyy_, idx_kyz_, idx_kzz_;
  int idx_alp_, idx_betax_, idx_betay_, idx_betaz_;

  // BSSN-like basis indices
  int idx_chi_;
  int idx_h11_, idx_h12_, idx_h13_, idx_h22_, idx_h23_, idx_h33_;
  int idx_ktrace_;
  int idx_A11_, idx_A12_, idx_A13_, idx_A22_, idx_A23_, idx_A33_;
  int idx_lapse_;
  int idx_shift1_, idx_shift2_, idx_shift3_;

  void detect_variable_basis();
  int component_index(const char *name, bool required) const;

  bool point_in_box(const SourceBox &box, int i, int j, int k) const;
  bool point_in_box_data(const SourceBox &box, int i, int j, int k) const;
  bool box_intersects_index_range(const SourceBox &box,
                                  const std::array<int, 3> &ilo,
                                  const std::array<int, 3> &ihi) const;
  static constexpr std::size_t invalid_box_index =
      std::numeric_limits<std::size_t>::max();
  static std::uint64_t pack_index3(int i, int j, int k);
  static int floor_div(int a, int b);
  std::size_t find_box_containing_index(const SourceLevel &level, int i, int j,
                                        int k) const;
  std::size_t find_box_with_data_index(const SourceLevel &level, int i, int j,
                                       int k) const;
  const SourceBox *find_box_with_data(const SourceLevel &level, int i, int j,
                                      int k) const;
  void load_box_data_locked(int level_idx, std::size_t box_idx,
                            hid_t data_dset) const;
  void load_box_data(int level_idx, std::size_t box_idx) const;
  bool sample_component_at_level(int level_idx, int i, int j, int k, int comp,
                                 double &out) const;
  bool sample_components_at_level(int level_idx, int i, int j, int k,
                                  const int *components, int component_count,
                                  double *out) const;
  bool sample_adm_nearest(double x, double y, double z,
                          OutOfBoundsPolicy oob_policy, ADMSample &out,
                          bool need_metric_curv, bool need_lapse,
                          bool need_shift, int preferred_source_level) const;
  bool sample_adm_asymptotic(double x, double y, double z,
                             InterpolationMethod method, ADMSample &out,
                             bool need_metric_curv, bool need_lapse,
                             bool need_shift,
                             int preferred_source_level) const;

  double sample_component_nearest(int comp, double x, double y, double z,
                                  OutOfBoundsPolicy oob_policy,
                                  bool &ok, int preferred_source_level) const;
  double sample_component_trilinear(int comp, double x, double y, double z,
                                    OutOfBoundsPolicy oob_policy,
                                    bool &ok, int preferred_source_level) const;

  double sample_component(int comp, double x, double y, double z,
                          InterpolationMethod method,
                          OutOfBoundsPolicy oob_policy, bool &ok,
                          int preferred_source_level) const;

  static std::string normalize_name(std::string name);
};

} // namespace GRTresnaIDX

#endif
