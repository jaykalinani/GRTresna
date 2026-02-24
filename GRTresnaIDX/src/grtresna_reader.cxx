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

std::vector<std::array<int, 6>> read_boxes_dataset(const hid_t level_group) {
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
  std::vector<std::array<int, 6>> boxes(nboxes);

  if (dims[1] == 6) {
    for (size_t b = 0; b < nboxes; ++b) {
      for (int a = 0; a < 6; ++a) {
        boxes[b][a] = raw[b * 6 + static_cast<size_t>(a)];
      }
    }
  } else {
    for (size_t b = 0; b < nboxes; ++b) {
      for (int a = 0; a < 6; ++a) {
        boxes[b][a] = raw[static_cast<size_t>(a) * nboxes + b];
      }
    }
  }

  if (H5Sclose(space) < 0 || H5Dclose(dset) < 0) {
    CCTK_VERROR("Failed to close HDF5 handles for 'boxes' dataset");
  }
  return boxes;
}

inline int clamp_int(const int v, const int lo, const int hi) {
  return std::max(lo, std::min(v, hi));
}

inline double lerp(const double a, const double b, const double w) {
  return a + w * (b - a);
}

} // namespace

GRTresnaReader::GRTresnaReader()
    : loaded_(false), center_{{0.0, 0.0, 0.0}}, num_levels_(0),
      global_lo_{{0, 0, 0}}, global_hi_{{-1, -1, -1}}, ncomp_(0),
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

bool GRTresnaReader::point_in_box(const SourceBox &box, const int i, const int j,
                                  const int k) const {
  return i >= box.lo[0] && i <= box.hi[0] && j >= box.lo[1] &&
         j <= box.hi[1] && k >= box.lo[2] && k <= box.hi[2];
}

const GRTresnaReader::SourceBox *
GRTresnaReader::find_box_containing(const SourceLevel &level, const int i,
                                    const int j, const int k) const {
  for (const auto &box : level.boxes) {
    if (point_in_box(box, i, j, k)) {
      return &box;
    }
  }
  return nullptr;
}

bool GRTresnaReader::sample_component_at_level(const int level_idx, const int i,
                                               const int j, const int k,
                                               const int comp,
                                               double &out) const {
  if (level_idx < 0 || level_idx >= num_levels_) {
    return false;
  }
  if (comp < 0 || comp >= ncomp_) {
    return false;
  }

  const auto &level = levels_[level_idx];
  const auto *box = find_box_containing(level, i, j, k);
  if (box == nullptr) {
    return false;
  }

  const int il = i - box->lo[0] + box->nghost;
  const int jl = j - box->lo[1] + box->nghost;
  const int kl = k - box->lo[2] + box->nghost;
  if (il < 0 || il >= box->n_with_ghost[0] || jl < 0 ||
      jl >= box->n_with_ghost[1] || kl < 0 || kl >= box->n_with_ghost[2]) {
    return false;
  }

  const size_t idx_cell =
      static_cast<size_t>(il) +
      static_cast<size_t>(box->n_with_ghost[0]) *
          (static_cast<size_t>(jl) +
           static_cast<size_t>(box->n_with_ghost[1]) * static_cast<size_t>(kl));
  const size_t idx = box->begin + idx_cell * static_cast<size_t>(ncomp_) +
                     static_cast<size_t>(comp);

  if (idx >= box->end || idx >= level.data.size()) {
    return false;
  }

  out = level.data[idx];
  return true;
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
  levels_.clear();

  const hid_t file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file_id < 0) {
    CCTK_VERROR("Could not open GRTresna input file '%s'", filename.c_str());
  }

  int num_levels = read_int_attribute(file_id, "num_levels", false, 0);
  if (num_levels <= 0) {
    num_levels = 0;
    for (int lev = 0;; ++lev) {
      const std::string group_name = "level_" + std::to_string(lev);
      if (H5Lexists(file_id, group_name.c_str(), H5P_DEFAULT) <= 0) {
        break;
      }
      ++num_levels;
    }
  }
  if (num_levels <= 0) {
    CCTK_VERROR("Could not determine num_levels from source metadata");
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
    const std::string var_name =
        normalize_name(read_string_attribute(file_id, attr_name));
    comp_to_index_[var_name] = i;
  }

  detect_variable_basis();

  levels_.reserve(static_cast<size_t>(num_levels));
  for (int lev = 0; lev < num_levels; ++lev) {
    const std::string group_name = "level_" + std::to_string(lev);
    const hid_t level_group = H5Gopen2(file_id, group_name.c_str(), H5P_DEFAULT);
    if (level_group < 0) {
      CCTK_VERROR("Missing required HDF5 group '%s' in source file",
                  group_name.c_str());
    }

    SourceLevel level;
    level.dx = read_real_attribute(level_group, "dx", true, 0.0);
    if (!(level.dx > 0.0 && std::isfinite(level.dx))) {
      CCTK_VERROR("Invalid or missing positive dx on group '%s'",
                  group_name.c_str());
    }
    level.lo_union = {{std::numeric_limits<int>::max(),
                       std::numeric_limits<int>::max(),
                       std::numeric_limits<int>::max()}};
    level.hi_union = {{std::numeric_limits<int>::lowest(),
                       std::numeric_limits<int>::lowest(),
                       std::numeric_limits<int>::lowest()}};

    const auto boxes_raw = read_boxes_dataset(level_group);
    if (boxes_raw.empty()) {
      CCTK_VERROR("No boxes found in HDF5 group '%s'", group_name.c_str());
    }

    const auto offsets =
        read_int64_dataset(level_group, {"data:offsets=0", "data:offsets"});
    if (offsets.size() != boxes_raw.size() + 1) {
      CCTK_VERROR(
          "Expected %llu offsets for %llu boxes in group '%s', found %llu",
          static_cast<unsigned long long>(boxes_raw.size() + 1),
          static_cast<unsigned long long>(boxes_raw.size()), group_name.c_str(),
          static_cast<unsigned long long>(offsets.size()));
    }

    const auto raw_data =
        read_real_dataset(level_group,
                          {"data:datatype=0", "data:datatype=1", "data"});

    level.boxes.reserve(boxes_raw.size());
    for (size_t bi = 0; bi < boxes_raw.size(); ++bi) {
      const long long begin_ll = offsets[bi];
      const long long end_ll = offsets[bi + 1];
      if (begin_ll < 0 || end_ll <= begin_ll) {
        CCTK_VERROR(
            "Invalid offsets [%lld, %lld] for box %llu in group '%s'",
            static_cast<long long>(begin_ll), static_cast<long long>(end_ll),
            static_cast<unsigned long long>(bi), group_name.c_str());
      }
      if (static_cast<size_t>(end_ll) > raw_data.size()) {
        CCTK_VERROR(
            "Source data too short in group '%s': end=%llu data_size=%llu",
            group_name.c_str(), static_cast<unsigned long long>(end_ll),
            static_cast<unsigned long long>(raw_data.size()));
      }

      SourceBox box;
      box.lo = {{boxes_raw[bi][0], boxes_raw[bi][1], boxes_raw[bi][2]}};
      box.hi = {{boxes_raw[bi][3], boxes_raw[bi][4], boxes_raw[bi][5]}};
      box.begin = static_cast<size_t>(begin_ll);
      box.end = static_cast<size_t>(end_ll);

      const int nx = box.hi[0] - box.lo[0] + 1;
      const int ny = box.hi[1] - box.lo[1] + 1;
      const int nz = box.hi[2] - box.lo[2] + 1;
      if (nx <= 0 || ny <= 0 || nz <= 0) {
        CCTK_VERROR(
            "Invalid source box extents in '%s': lo=(%d,%d,%d), hi=(%d,%d,%d)",
            group_name.c_str(), box.lo[0], box.lo[1], box.lo[2], box.hi[0],
            box.hi[1], box.hi[2]);
      }

      const size_t values_in_box = box.end - box.begin;
      if (values_in_box % static_cast<size_t>(ncomp_) != 0) {
        CCTK_VERROR(
            "Data length for box %llu in '%s' (%llu) is not divisible by num_components (%d)",
            static_cast<unsigned long long>(bi), group_name.c_str(),
            static_cast<unsigned long long>(values_in_box), ncomp_);
      }
      const long long cells_with_ghost =
          static_cast<long long>(values_in_box / static_cast<size_t>(ncomp_));

      bool found_ghost = false;
      for (int g = 0; g <= 64; ++g) {
        const long long candidate =
            static_cast<long long>(nx + 2 * g) *
            static_cast<long long>(ny + 2 * g) *
            static_cast<long long>(nz + 2 * g);
        if (candidate == cells_with_ghost) {
          box.nghost = g;
          box.n_with_ghost = {{nx + 2 * g, ny + 2 * g, nz + 2 * g}};
          found_ghost = true;
          break;
        }
      }
      if (!found_ghost) {
        CCTK_VERROR(
            "Could not infer ghost-zone width for box %llu in '%s': nx=%d ny=%d nz=%d cells_with_ghost=%lld",
            static_cast<unsigned long long>(bi), group_name.c_str(), nx, ny,
            nz, cells_with_ghost);
      }

      level.lo_union[0] = std::min(level.lo_union[0], box.lo[0]);
      level.lo_union[1] = std::min(level.lo_union[1], box.lo[1]);
      level.lo_union[2] = std::min(level.lo_union[2], box.lo[2]);
      level.hi_union[0] = std::max(level.hi_union[0], box.hi[0]);
      level.hi_union[1] = std::max(level.hi_union[1], box.hi[1]);
      level.hi_union[2] = std::max(level.hi_union[2], box.hi[2]);

      level.boxes.push_back(box);
    }

    level.data = raw_data;
    levels_.push_back(level);

    if (H5Gclose(level_group) < 0) {
      CCTK_VERROR("Failed to close HDF5 group '%s'", group_name.c_str());
    }
  }

  if (H5Fclose(file_id) < 0) {
    CCTK_VERROR("Failed to close HDF5 file '%s'", filename.c_str());
  }

  num_levels_ = static_cast<int>(levels_.size());
  global_lo_ = levels_[0].lo_union;
  global_hi_ = levels_[0].hi_union;

  if (config_.use_source_center) {
    center_ = config_.source_center;
  } else {
    const double dx0 = levels_[0].dx;
    for (int d = 0; d < 3; ++d) {
      center_[d] =
          0.5 * static_cast<double>(global_lo_[d] + global_hi_[d] + 1) * dx0;
    }
  }

  has_matter_data_ = (component_index("phi", false) >= 0 &&
                      component_index("pi", false) >= 0);

  loaded_ = true;

  if (config_.verbosity >= 1) {
    size_t total_boxes = 0;
    for (const auto &level : levels_) {
      total_boxes += level.boxes.size();
    }

    CCTK_VINFO(
        "GRTresnaIDX loaded '%s': levels=%d, boxes=%llu, ncomp=%d, dx0=%e, "
        "coarse_box=[(%d,%d,%d)->(%d,%d,%d)], basis=%s",
        filename.c_str(), num_levels_,
        static_cast<unsigned long long>(total_boxes), ncomp_, levels_[0].dx,
        global_lo_[0], global_lo_[1], global_lo_[2], global_hi_[0],
        global_hi_[1], global_hi_[2],
        (basis_ == VariableBasis::adm ? "ADM" : "BSSN-like"));
  }
}

double GRTresnaReader::sample_component_nearest(
    const int comp, const double x, const double y, const double z,
    const OutOfBoundsPolicy oob_policy, bool &ok) const {
  for (int lev = num_levels_ - 1; lev >= 0; --lev) {
    const auto &level = levels_[lev];
    const int ix = static_cast<int>(std::llround((x + center_[0]) / level.dx - 0.5));
    const int iy = static_cast<int>(std::llround((y + center_[1]) / level.dx - 0.5));
    const int iz = static_cast<int>(std::llround((z + center_[2]) / level.dx - 0.5));

    double value = 0.0;
    if (sample_component_at_level(lev, ix, iy, iz, comp, value)) {
      return value;
    }
  }

  if (oob_policy == OutOfBoundsPolicy::error) {
    ok = false;
    return 0.0;
  }

  // Clamp fallback
  for (int lev = num_levels_ - 1; lev >= 0; --lev) {
    const auto &level = levels_[lev];
    int ix = static_cast<int>(std::llround((x + center_[0]) / level.dx - 0.5));
    int iy = static_cast<int>(std::llround((y + center_[1]) / level.dx - 0.5));
    int iz = static_cast<int>(std::llround((z + center_[2]) / level.dx - 0.5));

    ix = clamp_int(ix, level.lo_union[0], level.hi_union[0]);
    iy = clamp_int(iy, level.lo_union[1], level.hi_union[1]);
    iz = clamp_int(iz, level.lo_union[2], level.hi_union[2]);

    double value = 0.0;
    if (sample_component_at_level(lev, ix, iy, iz, comp, value)) {
      return value;
    }
  }

  ok = false;
  return 0.0;
}

double GRTresnaReader::sample_component_trilinear(
    const int comp, const double x, const double y, const double z,
    const OutOfBoundsPolicy oob_policy, bool &ok) const {
  int lev_sel = -1;
  int ix0 = 0, iy0 = 0, iz0 = 0;
  double wx = 0.0, wy = 0.0, wz = 0.0;

  for (int lev = num_levels_ - 1; lev >= 0; --lev) {
    const auto &level = levels_[lev];
    const double gx = (x + center_[0]) / level.dx - 0.5;
    const double gy = (y + center_[1]) / level.dx - 0.5;
    const double gz = (z + center_[2]) / level.dx - 0.5;

    const int tx0 = static_cast<int>(std::floor(gx));
    const int ty0 = static_cast<int>(std::floor(gy));
    const int tz0 = static_cast<int>(std::floor(gz));

    if (find_box_containing(level, tx0, ty0, tz0) != nullptr) {
      lev_sel = lev;
      ix0 = tx0;
      iy0 = ty0;
      iz0 = tz0;
      wx = gx - static_cast<double>(ix0);
      wy = gy - static_cast<double>(iy0);
      wz = gz - static_cast<double>(iz0);
      break;
    }
  }

  if (lev_sel < 0) {
    if (oob_policy == OutOfBoundsPolicy::error) {
      ok = false;
      return 0.0;
    }

    lev_sel = 0;
    const auto &level = levels_[lev_sel];
    const double gx = (x + center_[0]) / level.dx - 0.5;
    const double gy = (y + center_[1]) / level.dx - 0.5;
    const double gz = (z + center_[2]) / level.dx - 0.5;

    ix0 = static_cast<int>(std::floor(gx));
    iy0 = static_cast<int>(std::floor(gy));
    iz0 = static_cast<int>(std::floor(gz));
    wx = gx - static_cast<double>(ix0);
    wy = gy - static_cast<double>(iy0);
    wz = gz - static_cast<double>(iz0);
  }

  const auto &level = levels_[lev_sel];

  if (oob_policy == OutOfBoundsPolicy::clamp) {
    if (level.hi_union[0] <= level.lo_union[0]) {
      ix0 = level.lo_union[0];
      wx = 0.0;
    } else {
      if (ix0 < level.lo_union[0]) {
        ix0 = level.lo_union[0];
        wx = 0.0;
      }
      if (ix0 >= level.hi_union[0]) {
        ix0 = level.hi_union[0] - 1;
        wx = 1.0;
      }
    }

    if (level.hi_union[1] <= level.lo_union[1]) {
      iy0 = level.lo_union[1];
      wy = 0.0;
    } else {
      if (iy0 < level.lo_union[1]) {
        iy0 = level.lo_union[1];
        wy = 0.0;
      }
      if (iy0 >= level.hi_union[1]) {
        iy0 = level.hi_union[1] - 1;
        wy = 1.0;
      }
    }

    if (level.hi_union[2] <= level.lo_union[2]) {
      iz0 = level.lo_union[2];
      wz = 0.0;
    } else {
      if (iz0 < level.lo_union[2]) {
        iz0 = level.lo_union[2];
        wz = 0.0;
      }
      if (iz0 >= level.hi_union[2]) {
        iz0 = level.hi_union[2] - 1;
        wz = 1.0;
      }
    }
  } else {
    if (ix0 < level.lo_union[0] || ix0 >= level.hi_union[0] ||
        iy0 < level.lo_union[1] || iy0 >= level.hi_union[1] ||
        iz0 < level.lo_union[2] || iz0 >= level.hi_union[2]) {
      ok = false;
      return 0.0;
    }
  }

  const int ix1 = std::min(ix0 + 1, level.hi_union[0]);
  const int iy1 = std::min(iy0 + 1, level.hi_union[1]);
  const int iz1 = std::min(iz0 + 1, level.hi_union[2]);

  auto fetch_value = [&](const int i, const int j, const int k,
                         double &value) {
    if (sample_component_at_level(lev_sel, i, j, k, comp, value)) {
      return true;
    }

    const double xc = (static_cast<double>(i) + 0.5) * level.dx - center_[0];
    const double yc = (static_cast<double>(j) + 0.5) * level.dx - center_[1];
    const double zc = (static_cast<double>(k) + 0.5) * level.dx - center_[2];

    for (int lev = lev_sel - 1; lev >= 0; --lev) {
      const auto &coarse = levels_[lev];
      const int ic =
          static_cast<int>(std::llround((xc + center_[0]) / coarse.dx - 0.5));
      const int jc =
          static_cast<int>(std::llround((yc + center_[1]) / coarse.dx - 0.5));
      const int kc =
          static_cast<int>(std::llround((zc + center_[2]) / coarse.dx - 0.5));
      if (sample_component_at_level(lev, ic, jc, kc, comp, value)) {
        return true;
      }
    }

    if (oob_policy == OutOfBoundsPolicy::clamp) {
      bool ok_local = true;
      value = sample_component_nearest(comp, xc, yc, zc,
                                       OutOfBoundsPolicy::clamp, ok_local);
      return ok_local;
    }

    return false;
  };

  double c000 = 0.0, c100 = 0.0, c010 = 0.0, c110 = 0.0;
  double c001 = 0.0, c101 = 0.0, c011 = 0.0, c111 = 0.0;

  if (!fetch_value(ix0, iy0, iz0, c000) || !fetch_value(ix1, iy0, iz0, c100) ||
      !fetch_value(ix0, iy1, iz0, c010) || !fetch_value(ix1, iy1, iz0, c110) ||
      !fetch_value(ix0, iy0, iz1, c001) || !fetch_value(ix1, iy0, iz1, c101) ||
      !fetch_value(ix0, iy1, iz1, c011) || !fetch_value(ix1, iy1, iz1, c111)) {
    ok = false;
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

double GRTresnaReader::sample_component(const int comp, const double x,
                                        const double y, const double z,
                                        const InterpolationMethod method,
                                        const OutOfBoundsPolicy oob_policy,
                                        bool &ok) const {
  if (!loaded_) {
    ok = false;
    return 0.0;
  }

  if (method == InterpolationMethod::nearest) {
    return sample_component_nearest(comp, x, y, z, oob_policy, ok);
  }
  return sample_component_trilinear(comp, x, y, z, oob_policy, ok);
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
