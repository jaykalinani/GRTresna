#ifndef GRTRESNAIDX_GRTRESNA_READER_HXX
#define GRTRESNAIDX_GRTRESNA_READER_HXX

#include <array>
#include <string>
#include <unordered_map>
#include <vector>

namespace GRTresnaIDX {

enum class InterpolationMethod { nearest, trilinear };
enum class OutOfBoundsPolicy { clamp, error };

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

  bool sample_adm(double x, double y, double z, InterpolationMethod method,
                  OutOfBoundsPolicy oob_policy, ADMSample &out) const;

  bool is_loaded() const { return loaded_; }
  bool has_matter_data() const { return has_matter_data_; }

private:
  enum class VariableBasis { adm, bssn };

  bool loaded_;
  ReaderConfig config_;

  // Current implementation supports level_0 with exactly one box.
  double dx_;
  std::array<double, 3> center_;
  std::array<int, 3> lo_;
  std::array<int, 3> hi_;
  std::array<int, 3> n_with_ghost_;
  int nghost_;
  int ncomp_;

  bool has_matter_data_;
  VariableBasis basis_;
  std::unordered_map<std::string, int> comp_to_index_;

  // Stored as [cell_index * ncomp + component]
  std::vector<double> data_;

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
  double sample_component(int comp, double x, double y, double z,
                          InterpolationMethod method,
                          OutOfBoundsPolicy oob_policy, bool &ok) const;
  double cell_value(int i, int j, int k, int comp, bool &ok) const;

  static std::string normalize_name(std::string name);
};

} // namespace GRTresnaIDX

#endif
