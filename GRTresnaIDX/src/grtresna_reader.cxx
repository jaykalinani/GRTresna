#include "grtresna_reader.hxx"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include <cctk.h>
#include <hdf5.h>

namespace GRTresnaIDX {
namespace {

inline bool has_attribute(const hid_t obj, const std::string &name) {
  const htri_t exists = H5Aexists(obj, name.c_str());
  return exists > 0;
}

int read_int_attribute(const hid_t obj, const std::string &name,
                       const bool required, const int default_value) {
  if (!has_attribute(obj, name)) {
    if (required) {
      CCTK_VERROR("Missing required HDF5 attribute '%s'", name.c_str());
    }
    return default_value;
  }

  const hid_t attr = H5Aopen(obj, name.c_str(), H5P_DEFAULT);
  if (attr < 0) {
    CCTK_VERROR("Failed to open HDF5 attribute '%s'", name.c_str());
  }
  int value = default_value;
  if (H5Aread(attr, H5T_NATIVE_INT, &value) < 0) {
    CCTK_VERROR("Failed to read integer HDF5 attribute '%s'", name.c_str());
  }
  if (H5Aclose(attr) < 0) {
    CCTK_VERROR("Failed to close HDF5 attribute '%s'", name.c_str());
  }
  return value;
}

double read_real_attribute(const hid_t obj, const std::string &name,
                           const bool required, const double default_value) {
  if (!has_attribute(obj, name)) {
    if (required) {
      CCTK_VERROR("Missing required HDF5 attribute '%s'", name.c_str());
    }
    return default_value;
  }

  const hid_t attr = H5Aopen(obj, name.c_str(), H5P_DEFAULT);
  if (attr < 0) {
    CCTK_VERROR("Failed to open HDF5 attribute '%s'", name.c_str());
  }
  double value = default_value;
  if (H5Aread(attr, H5T_NATIVE_DOUBLE, &value) < 0) {
    CCTK_VERROR("Failed to read real HDF5 attribute '%s'", name.c_str());
  }
  if (H5Aclose(attr) < 0) {
    CCTK_VERROR("Failed to close HDF5 attribute '%s'", name.c_str());
  }
  return value;
}

std::string read_string_attribute(const hid_t obj, const std::string &name) {
  if (!has_attribute(obj, name)) {
    CCTK_VERROR("Missing required HDF5 string attribute '%s'", name.c_str());
  }

  const hid_t attr = H5Aopen(obj, name.c_str(), H5P_DEFAULT);
  if (attr < 0) {
    CCTK_VERROR("Failed to open HDF5 string attribute '%s'", name.c_str());
  }

  const hid_t attr_type = H5Aget_type(attr);
  if (attr_type < 0) {
    CCTK_VERROR("Failed to get type of HDF5 attribute '%s'", name.c_str());
  }

  std::string value;
  if (H5Tis_variable_str(attr_type) > 0) {
    char *buffer = nullptr;
    if (H5Aread(attr, attr_type, &buffer) < 0) {
      CCTK_VERROR("Failed to read HDF5 string attribute '%s'", name.c_str());
    }
    value = buffer ? std::string(buffer) : std::string();
    if (buffer != nullptr) {
      H5free_memory(buffer);
    }
  } else {
    const size_t bytes = H5Tget_size(attr_type);
    std::vector<char> buffer(bytes + 1, '\0');
    if (H5Aread(attr, attr_type, buffer.data()) < 0) {
      CCTK_VERROR("Failed to read HDF5 string attribute '%s'", name.c_str());
    }
    value = std::string(buffer.data());
  }

  if (H5Tclose(attr_type) < 0) {
    CCTK_VERROR("Failed to close HDF5 datatype for attribute '%s'",
                name.c_str());
  }
  if (H5Aclose(attr) < 0) {
    CCTK_VERROR("Failed to close HDF5 attribute '%s'", name.c_str());
  }
  return value;
}

hid_t open_dataset_any(const hid_t group,
                       const std::initializer_list<const char *> names) {
  for (const auto *name : names) {
    if (H5Lexists(group, name, H5P_DEFAULT) <= 0) {
      continue;
    }
    const hid_t dset = H5Dopen2(group, name, H5P_DEFAULT);
    if (dset >= 0) {
      return dset;
    }
  }
  CCTK_VERROR("Could not find any expected HDF5 dataset in current group");
  return -1;
}

std::vector<long long>
read_int64_dataset(const hid_t group,
                   const std::initializer_list<const char *> names) {
  const hid_t dset = open_dataset_any(group, names);
  const hid_t space = H5Dget_space(dset);
  if (space < 0) {
    CCTK_VERROR("Failed to get HDF5 dataspace for int64 dataset");
  }

  const int ndims = H5Sget_simple_extent_ndims(space);
  if (ndims < 0) {
    CCTK_VERROR("Failed to get rank of int64 dataset");
  }

  std::vector<hsize_t> dims(ndims > 0 ? ndims : 1, 1);
  if (ndims > 0 && H5Sget_simple_extent_dims(space, dims.data(), nullptr) < 0) {
    CCTK_VERROR("Failed to get dimensions of int64 dataset");
  }

  size_t total = 1;
  for (int d = 0; d < ndims; ++d) {
    total *= static_cast<size_t>(dims[d]);
  }

  std::vector<long long> out(total);
  if (H5Dread(dset, H5T_NATIVE_LLONG, H5S_ALL, H5S_ALL, H5P_DEFAULT,
              out.data()) < 0) {
    CCTK_VERROR("Failed to read int64 dataset");
  }

  if (H5Sclose(space) < 0 || H5Dclose(dset) < 0) {
    CCTK_VERROR("Failed to close HDF5 handles for int64 dataset");
  }

  return out;
}

std::vector<double>
read_real_dataset(const hid_t group,
                  const std::initializer_list<const char *> names) {
  const hid_t dset = open_dataset_any(group, names);
  const hid_t space = H5Dget_space(dset);
  if (space < 0) {
    CCTK_VERROR("Failed to get HDF5 dataspace for real dataset");
  }

  const int ndims = H5Sget_simple_extent_ndims(space);
  if (ndims < 0) {
    CCTK_VERROR("Failed to get rank of real dataset");
  }

  std::vector<hsize_t> dims(ndims > 0 ? ndims : 1, 1);
  if (ndims > 0 && H5Sget_simple_extent_dims(space, dims.data(), nullptr) < 0) {
    CCTK_VERROR("Failed to get dimensions of real dataset");
  }

  size_t total = 1;
  for (int d = 0; d < ndims; ++d) {
    total *= static_cast<size_t>(dims[d]);
  }

  std::vector<double> out(total);
  if (H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
              out.data()) < 0) {
    CCTK_VERROR("Failed to read real dataset");
  }

  if (H5Sclose(space) < 0 || H5Dclose(dset) < 0) {
    CCTK_VERROR("Failed to close HDF5 handles for real dataset");
  }

  return out;
}

std::array<int, 6> read_single_box(const hid_t level_group) {
  const hid_t dset = open_dataset_any(level_group, {"boxes"});
  const hid_t space = H5Dget_space(dset);
  if (space < 0) {
    CCTK_VERROR("Failed to get dataspace of 'boxes' dataset");
  }

  const int ndims = H5Sget_simple_extent_ndims(space);
  if (ndims != 2) {
    CCTK_VERROR(
        "Unsupported 'boxes' rank %d; expected rank-2 integer array", ndims);
  }

  hsize_t dims[2] = {0, 0};
  if (H5Sget_simple_extent_dims(space, dims, nullptr) < 0) {
    CCTK_VERROR("Failed to read dimensions of 'boxes' dataset");
  }

  if (!((dims[1] == 6) || (dims[0] == 6))) {
    CCTK_VERROR("Unsupported 'boxes' shape [%llu,%llu]; expected [N,6] or [6,N]",
                static_cast<unsigned long long>(dims[0]),
                static_cast<unsigned long long>(dims[1]));
  }

  const size_t total = static_cast<size_t>(dims[0] * dims[1]);
  std::vector<int> raw(total);
  if (H5Dread(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
              raw.data()) < 0) {
    CCTK_VERROR("Failed to read 'boxes' dataset as integers");
  }

  const size_t nboxes =
      (dims[1] == 6) ? static_cast<size_t>(dims[0]) : static_cast<size_t>(dims[1]);
  if (nboxes != 1) {
    CCTK_VERROR(
        "Current reader supports exactly one source box on level_0; file has %llu boxes",
        static_cast<unsigned long long>(nboxes));
  }

  std::array<int, 6> box{};
  if (dims[1] == 6) {
    for (int a = 0; a < 6; ++a) {
      box[a] = raw[static_cast<size_t>(a)];
    }
  } else {
    for (int a = 0; a < 6; ++a) {
      box[a] = raw[static_cast<size_t>(a * nboxes)];
    }
  }

  if (H5Sclose(space) < 0 || H5Dclose(dset) < 0) {
    CCTK_VERROR("Failed to close HDF5 handles for 'boxes' dataset");
  }
  return box;
}

inline int clamp_int(const int v, const int lo, const int hi) {
  return std::max(lo, std::min(v, hi));
}

inline double lerp(const double a, const double b, const double w) {
  return a + w * (b - a);
}

} // namespace

GRTresnaReader::GRTresnaReader()
    : loaded_(false), dx_(0.0), center_{{0.0, 0.0, 0.0}}, lo_{{0, 0, 0}},
      hi_{{-1, -1, -1}}, n_with_ghost_{{0, 0, 0}}, nghost_(0), ncomp_(0),
      has_matter_data_(false), basis_(VariableBasis::adm), idx_gxx_(-1),
      idx_gxy_(-1), idx_gxz_(-1), idx_gyy_(-1), idx_gyz_(-1), idx_gzz_(-1),
      idx_kxx_(-1), idx_kxy_(-1), idx_kxz_(-1), idx_kyy_(-1), idx_kyz_(-1),
      idx_kzz_(-1), idx_alp_(-1), idx_betax_(-1), idx_betay_(-1),
      idx_betaz_(-1), idx_chi_(-1), idx_h11_(-1), idx_h12_(-1), idx_h13_(-1),
      idx_h22_(-1), idx_h23_(-1), idx_h33_(-1), idx_ktrace_(-1), idx_A11_(-1),
      idx_A12_(-1), idx_A13_(-1), idx_A22_(-1), idx_A23_(-1), idx_A33_(-1),
      idx_lapse_(-1), idx_shift1_(-1), idx_shift2_(-1), idx_shift3_(-1) {}

std::string GRTresnaReader::normalize_name(std::string name) {
  for (char &c : name) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return name;
}

int GRTresnaReader::component_index(const char *name, const bool required) const {
  const std::string key = normalize_name(name);
  const auto it = comp_to_index_.find(key);
  if (it == comp_to_index_.end()) {
    if (required) {
      CCTK_VERROR("Required component '%s' not found in source metadata",
                  name);
    }
    return -1;
  }
  return it->second;
}

void GRTresnaReader::detect_variable_basis() {
  // Try ADM basis first
  const int gxx = component_index("gxx", false);
  const int gxy = component_index("gxy", false);
  const int gxz = component_index("gxz", false);
  const int gyy = component_index("gyy", false);
  const int gyz = component_index("gyz", false);
  const int gzz = component_index("gzz", false);
  const int kxx = component_index("kxx", false);
  const int kxy = component_index("kxy", false);
  const int kxz = component_index("kxz", false);
  const int kyy = component_index("kyy", false);
  const int kyz = component_index("kyz", false);
  const int kzz = component_index("kzz", false);
  int alp = component_index("alp", false);
  if (alp < 0) {
    alp = component_index("lapse", false);
  }
  int betax = component_index("betax", false);
  int betay = component_index("betay", false);
  int betaz = component_index("betaz", false);
  if (betax < 0 || betay < 0 || betaz < 0) {
    betax = component_index("shift1", false);
    betay = component_index("shift2", false);
    betaz = component_index("shift3", false);
  }

  const bool have_adm = (gxx >= 0 && gxy >= 0 && gxz >= 0 && gyy >= 0 &&
                         gyz >= 0 && gzz >= 0 && kxx >= 0 && kxy >= 0 &&
                         kxz >= 0 && kyy >= 0 && kyz >= 0 && kzz >= 0 &&
                         alp >= 0 && betax >= 0 && betay >= 0 && betaz >= 0);
  if (have_adm) {
    basis_ = VariableBasis::adm;
    idx_gxx_ = gxx;
    idx_gxy_ = gxy;
    idx_gxz_ = gxz;
    idx_gyy_ = gyy;
    idx_gyz_ = gyz;
    idx_gzz_ = gzz;
    idx_kxx_ = kxx;
    idx_kxy_ = kxy;
    idx_kxz_ = kxz;
    idx_kyy_ = kyy;
    idx_kyz_ = kyz;
    idx_kzz_ = kzz;
    idx_alp_ = alp;
    idx_betax_ = betax;
    idx_betay_ = betay;
    idx_betaz_ = betaz;
    return;
  }

  // Then BSSN-like basis
  const int chi = component_index("chi", false);
  const int h11 = component_index("h11", false);
  const int h12 = component_index("h12", false);
  const int h13 = component_index("h13", false);
  const int h22 = component_index("h22", false);
  const int h23 = component_index("h23", false);
  const int h33 = component_index("h33", false);
  const int ktrace = component_index("k", false);
  const int A11 = component_index("a11", false);
  const int A12 = component_index("a12", false);
  const int A13 = component_index("a13", false);
  const int A22 = component_index("a22", false);
  const int A23 = component_index("a23", false);
  const int A33 = component_index("a33", false);
  const int lapse = component_index("lapse", false);
  const int shift1 = component_index("shift1", false);
  const int shift2 = component_index("shift2", false);
  const int shift3 = component_index("shift3", false);

  const bool have_bssn = (chi >= 0 && h11 >= 0 && h12 >= 0 && h13 >= 0 &&
                          h22 >= 0 && h23 >= 0 && h33 >= 0 && ktrace >= 0 &&
                          A11 >= 0 && A12 >= 0 && A13 >= 0 && A22 >= 0 &&
                          A23 >= 0 && A33 >= 0 && lapse >= 0 && shift1 >= 0 &&
                          shift2 >= 0 && shift3 >= 0);
  if (have_bssn) {
    basis_ = VariableBasis::bssn;
    idx_chi_ = chi;
    idx_h11_ = h11;
    idx_h12_ = h12;
    idx_h13_ = h13;
    idx_h22_ = h22;
    idx_h23_ = h23;
    idx_h33_ = h33;
    idx_ktrace_ = ktrace;
    idx_A11_ = A11;
    idx_A12_ = A12;
    idx_A13_ = A13;
    idx_A22_ = A22;
    idx_A23_ = A23;
    idx_A33_ = A33;
    idx_lapse_ = lapse;
    idx_shift1_ = shift1;
    idx_shift2_ = shift2;
    idx_shift3_ = shift3;
    return;
  }

  CCTK_VERROR(
      "Could not detect supported source variable basis. Expected ADM "
      "(gij/Kij/alp/betai) or BSSN-like (chi,hij,K,Aij,lapse,shifti) fields.");
}

void GRTresnaReader::load_file(const std::string &filename,
                               const ReaderConfig &config) {
  loaded_ = false;
  config_ = config;
  comp_to_index_.clear();
  data_.clear();

  const hid_t file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file_id < 0) {
    CCTK_VERROR("Could not open GRTresna input file '%s'", filename.c_str());
  }

  const int num_levels = read_int_attribute(file_id, "num_levels", false, 1);
  if (num_levels != 1) {
    CCTK_VERROR("Current reader supports only num_levels=1; file has num_levels=%d",
                num_levels);
  }

  ncomp_ = read_int_attribute(file_id, "num_components", false, 0);
  if (ncomp_ <= 0) {
    ncomp_ = 0;
    for (int i = 0;; ++i) {
      const std::string attr_name = "component_" + std::to_string(i);
      if (!has_attribute(file_id, attr_name)) {
        break;
      }
      ++ncomp_;
    }
  }
  if (ncomp_ <= 0) {
    CCTK_VERROR("Could not determine num_components from source file metadata");
  }

  for (int i = 0; i < ncomp_; ++i) {
    const std::string attr_name = "component_" + std::to_string(i);
    const std::string var_name = normalize_name(read_string_attribute(file_id, attr_name));
    comp_to_index_[var_name] = i;
  }

  detect_variable_basis();

  const hid_t level_group = H5Gopen2(file_id, "level_0", H5P_DEFAULT);
  if (level_group < 0) {
    CCTK_VERROR("Missing required HDF5 group 'level_0' in source file");
  }

  dx_ = read_real_attribute(level_group, "dx", true, 0.0);
  const std::array<int, 6> box = read_single_box(level_group);
  lo_ = {{box[0], box[1], box[2]}};
  hi_ = {{box[3], box[4], box[5]}};

  const auto offsets =
      read_int64_dataset(level_group, {"data:offsets=0", "data:offsets"});
  if (offsets.size() != 2) {
    CCTK_VERROR(
        "Expected exactly 2 offsets for single-box input; found %llu entries",
        static_cast<unsigned long long>(offsets.size()));
  }
  if (offsets[0] < 0 || offsets[1] <= offsets[0]) {
    CCTK_VERROR("Invalid offsets in source file: [%lld, %lld]",
                static_cast<long long>(offsets[0]),
                static_cast<long long>(offsets[1]));
  }

  const auto raw_data =
      read_real_dataset(level_group, {"data:datatype=0", "data:datatype=1", "data"});
  const size_t begin = static_cast<size_t>(offsets[0]);
  const size_t end = static_cast<size_t>(offsets[1]);
  if (end > raw_data.size()) {
    CCTK_VERROR(
        "Source data shorter than declared offsets: end=%llu, data_size=%llu",
        static_cast<unsigned long long>(end),
        static_cast<unsigned long long>(raw_data.size()));
  }

  const size_t values_in_box = end - begin;
  if (values_in_box % static_cast<size_t>(ncomp_) != 0) {
    CCTK_VERROR(
        "Data length for level_0 box (%llu) is not divisible by num_components (%d)",
        static_cast<unsigned long long>(values_in_box), ncomp_);
  }

  const int nx = hi_[0] - lo_[0] + 1;
  const int ny = hi_[1] - lo_[1] + 1;
  const int nz = hi_[2] - lo_[2] + 1;
  if (nx <= 0 || ny <= 0 || nz <= 0) {
    CCTK_VERROR("Invalid source box extents: lo=(%d,%d,%d), hi=(%d,%d,%d)", lo_[0],
                lo_[1], lo_[2], hi_[0], hi_[1], hi_[2]);
  }

  const long long cells_with_ghost =
      static_cast<long long>(values_in_box / static_cast<size_t>(ncomp_));

  bool found_ghost = false;
  for (int g = 0; g <= 64; ++g) {
    const long long candidate =
        static_cast<long long>(nx + 2 * g) * static_cast<long long>(ny + 2 * g) *
        static_cast<long long>(nz + 2 * g);
    if (candidate == cells_with_ghost) {
      nghost_ = g;
      found_ghost = true;
      break;
    }
  }
  if (!found_ghost) {
    CCTK_VERROR(
        "Could not infer ghost-zone width from source data layout: nx=%d ny=%d nz=%d "
        "cells_with_ghost=%lld ncomp=%d",
        nx, ny, nz, cells_with_ghost, ncomp_);
  }

  n_with_ghost_ = {{nx + 2 * nghost_, ny + 2 * nghost_, nz + 2 * nghost_}};
  data_.assign(raw_data.begin() + static_cast<long long>(begin),
               raw_data.begin() + static_cast<long long>(end));

  if (config_.use_source_center) {
    center_ = config_.source_center;
  } else {
    for (int d = 0; d < 3; ++d) {
      center_[d] = 0.5 * static_cast<double>(lo_[d] + hi_[d] + 1) * dx_;
    }
  }

  has_matter_data_ =
      (component_index("phi", false) >= 0 && component_index("pi", false) >= 0);

  if (H5Gclose(level_group) < 0 || H5Fclose(file_id) < 0) {
    CCTK_VERROR("Failed to close HDF5 handles while finishing source read");
  }

  loaded_ = true;

  if (config_.verbosity >= 1) {
    CCTK_VINFO(
        "GRTresnaIDX loaded '%s': dx=%e, box=[(%d,%d,%d) -> (%d,%d,%d)], nghost=%d, "
        "ncomp=%d, basis=%s",
        filename.c_str(), dx_, lo_[0], lo_[1], lo_[2], hi_[0], hi_[1], hi_[2],
        nghost_, ncomp_, (basis_ == VariableBasis::adm ? "ADM" : "BSSN-like"));
  }
}

double GRTresnaReader::cell_value(const int i, const int j, const int k,
                                  const int comp, bool &ok) const {
  const int il = i - lo_[0] + nghost_;
  const int jl = j - lo_[1] + nghost_;
  const int kl = k - lo_[2] + nghost_;

  if (il < 0 || il >= n_with_ghost_[0] || jl < 0 || jl >= n_with_ghost_[1] ||
      kl < 0 || kl >= n_with_ghost_[2]) {
    ok = false;
    return 0.0;
  }
  if (comp < 0 || comp >= ncomp_) {
    ok = false;
    return 0.0;
  }

  const size_t idx_cell =
      static_cast<size_t>(il) +
      static_cast<size_t>(n_with_ghost_[0]) *
          (static_cast<size_t>(jl) +
           static_cast<size_t>(n_with_ghost_[1]) * static_cast<size_t>(kl));
  const size_t idx = idx_cell * static_cast<size_t>(ncomp_) +
                     static_cast<size_t>(comp);
  if (idx >= data_.size()) {
    ok = false;
    return 0.0;
  }
  return data_[idx];
}

double GRTresnaReader::sample_component(const int comp, const double x,
                                        const double y, const double z,
                                        const InterpolationMethod method,
                                        const OutOfBoundsPolicy oob_policy,
                                        bool &ok) const {
  if (!loaded_) {
    ok = false;
    return 0.0;
  }

  const double gx = (x + center_[0]) / dx_ - 0.5;
  const double gy = (y + center_[1]) / dx_ - 0.5;
  const double gz = (z + center_[2]) / dx_ - 0.5;

  if (method == InterpolationMethod::nearest) {
    int ix = static_cast<int>(std::llround(gx));
    int iy = static_cast<int>(std::llround(gy));
    int iz = static_cast<int>(std::llround(gz));

    if (oob_policy == OutOfBoundsPolicy::clamp) {
      ix = clamp_int(ix, lo_[0], hi_[0]);
      iy = clamp_int(iy, lo_[1], hi_[1]);
      iz = clamp_int(iz, lo_[2], hi_[2]);
    } else {
      if (ix < lo_[0] || ix > hi_[0] || iy < lo_[1] || iy > hi_[1] || iz < lo_[2] ||
          iz > hi_[2]) {
        ok = false;
        return 0.0;
      }
    }

    return cell_value(ix, iy, iz, comp, ok);
  }

  int ix0 = static_cast<int>(std::floor(gx));
  int iy0 = static_cast<int>(std::floor(gy));
  int iz0 = static_cast<int>(std::floor(gz));
  double wx = gx - static_cast<double>(ix0);
  double wy = gy - static_cast<double>(iy0);
  double wz = gz - static_cast<double>(iz0);

  if (oob_policy == OutOfBoundsPolicy::clamp) {
    if (hi_[0] <= lo_[0]) {
      ix0 = lo_[0];
      wx = 0.0;
    } else {
      if (ix0 < lo_[0]) {
        ix0 = lo_[0];
        wx = 0.0;
      }
      if (ix0 >= hi_[0]) {
        ix0 = hi_[0] - 1;
        wx = 1.0;
      }
    }

    if (hi_[1] <= lo_[1]) {
      iy0 = lo_[1];
      wy = 0.0;
    } else {
      if (iy0 < lo_[1]) {
        iy0 = lo_[1];
        wy = 0.0;
      }
      if (iy0 >= hi_[1]) {
        iy0 = hi_[1] - 1;
        wy = 1.0;
      }
    }

    if (hi_[2] <= lo_[2]) {
      iz0 = lo_[2];
      wz = 0.0;
    } else {
      if (iz0 < lo_[2]) {
        iz0 = lo_[2];
        wz = 0.0;
      }
      if (iz0 >= hi_[2]) {
        iz0 = hi_[2] - 1;
        wz = 1.0;
      }
    }
  } else {
    if (ix0 < lo_[0] || ix0 >= hi_[0] || iy0 < lo_[1] || iy0 >= hi_[1] ||
        iz0 < lo_[2] || iz0 >= hi_[2]) {
      ok = false;
      return 0.0;
    }
  }

  const int ix1 = std::min(ix0 + 1, hi_[0]);
  const int iy1 = std::min(iy0 + 1, hi_[1]);
  const int iz1 = std::min(iz0 + 1, hi_[2]);

  const double c000 = cell_value(ix0, iy0, iz0, comp, ok);
  const double c100 = cell_value(ix1, iy0, iz0, comp, ok);
  const double c010 = cell_value(ix0, iy1, iz0, comp, ok);
  const double c110 = cell_value(ix1, iy1, iz0, comp, ok);
  const double c001 = cell_value(ix0, iy0, iz1, comp, ok);
  const double c101 = cell_value(ix1, iy0, iz1, comp, ok);
  const double c011 = cell_value(ix0, iy1, iz1, comp, ok);
  const double c111 = cell_value(ix1, iy1, iz1, comp, ok);
  if (!ok) {
    return 0.0;
  }

  const double c00 = lerp(c000, c100, wx);
  const double c10 = lerp(c010, c110, wx);
  const double c01 = lerp(c001, c101, wx);
  const double c11 = lerp(c011, c111, wx);
  const double c0 = lerp(c00, c10, wy);
  const double c1 = lerp(c01, c11, wy);
  return lerp(c0, c1, wz);
}

bool GRTresnaReader::sample_adm(const double x, const double y, const double z,
                                const InterpolationMethod method,
                                const OutOfBoundsPolicy oob_policy,
                                ADMSample &out) const {
  bool ok = true;

  if (basis_ == VariableBasis::adm) {
    out.gxx = sample_component(idx_gxx_, x, y, z, method, oob_policy, ok);
    out.gxy = sample_component(idx_gxy_, x, y, z, method, oob_policy, ok);
    out.gxz = sample_component(idx_gxz_, x, y, z, method, oob_policy, ok);
    out.gyy = sample_component(idx_gyy_, x, y, z, method, oob_policy, ok);
    out.gyz = sample_component(idx_gyz_, x, y, z, method, oob_policy, ok);
    out.gzz = sample_component(idx_gzz_, x, y, z, method, oob_policy, ok);

    out.kxx = sample_component(idx_kxx_, x, y, z, method, oob_policy, ok);
    out.kxy = sample_component(idx_kxy_, x, y, z, method, oob_policy, ok);
    out.kxz = sample_component(idx_kxz_, x, y, z, method, oob_policy, ok);
    out.kyy = sample_component(idx_kyy_, x, y, z, method, oob_policy, ok);
    out.kyz = sample_component(idx_kyz_, x, y, z, method, oob_policy, ok);
    out.kzz = sample_component(idx_kzz_, x, y, z, method, oob_policy, ok);

    out.alp = sample_component(idx_alp_, x, y, z, method, oob_policy, ok);
    out.betax = sample_component(idx_betax_, x, y, z, method, oob_policy, ok);
    out.betay = sample_component(idx_betay_, x, y, z, method, oob_policy, ok);
    out.betaz = sample_component(idx_betaz_, x, y, z, method, oob_policy, ok);
    return ok;
  }

  const double chi_raw =
      sample_component(idx_chi_, x, y, z, method, oob_policy, ok);
  const double h11 =
      sample_component(idx_h11_, x, y, z, method, oob_policy, ok);
  const double h12 =
      sample_component(idx_h12_, x, y, z, method, oob_policy, ok);
  const double h13 =
      sample_component(idx_h13_, x, y, z, method, oob_policy, ok);
  const double h22 =
      sample_component(idx_h22_, x, y, z, method, oob_policy, ok);
  const double h23 =
      sample_component(idx_h23_, x, y, z, method, oob_policy, ok);
  const double h33 =
      sample_component(idx_h33_, x, y, z, method, oob_policy, ok);

  const double ktrace =
      sample_component(idx_ktrace_, x, y, z, method, oob_policy, ok);
  const double A11 =
      sample_component(idx_A11_, x, y, z, method, oob_policy, ok);
  const double A12 =
      sample_component(idx_A12_, x, y, z, method, oob_policy, ok);
  const double A13 =
      sample_component(idx_A13_, x, y, z, method, oob_policy, ok);
  const double A22 =
      sample_component(idx_A22_, x, y, z, method, oob_policy, ok);
  const double A23 =
      sample_component(idx_A23_, x, y, z, method, oob_policy, ok);
  const double A33 =
      sample_component(idx_A33_, x, y, z, method, oob_policy, ok);

  out.alp = sample_component(idx_lapse_, x, y, z, method, oob_policy, ok);
  out.betax = sample_component(idx_shift1_, x, y, z, method, oob_policy, ok);
  out.betay = sample_component(idx_shift2_, x, y, z, method, oob_policy, ok);
  out.betaz = sample_component(idx_shift3_, x, y, z, method, oob_policy, ok);
  if (!ok) {
    return false;
  }

  const double chi = std::max(chi_raw, config_.chi_floor);
  if (!(std::isfinite(chi) && chi > 0.0)) {
    return false;
  }

  const double inv_chi = 1.0 / chi;
  out.gxx = h11 * inv_chi;
  out.gxy = h12 * inv_chi;
  out.gxz = h13 * inv_chi;
  out.gyy = h22 * inv_chi;
  out.gyz = h23 * inv_chi;
  out.gzz = h33 * inv_chi;

  const double one_third_k = ktrace / 3.0;
  out.kxx = (A11 + one_third_k * h11) * inv_chi;
  out.kxy = (A12 + one_third_k * h12) * inv_chi;
  out.kxz = (A13 + one_third_k * h13) * inv_chi;
  out.kyy = (A22 + one_third_k * h22) * inv_chi;
  out.kyz = (A23 + one_third_k * h23) * inv_chi;
  out.kzz = (A33 + one_third_k * h33) * inv_chi;

  return true;
}

} // namespace GRTresnaIDX
