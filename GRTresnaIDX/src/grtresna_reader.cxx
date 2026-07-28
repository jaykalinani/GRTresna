#include "grtresna_reader.hxx"

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <string>
#include <utility>
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

bool read_intvect_attribute(const hid_t obj, const std::string &name,
                            std::array<int, 3> &value) {
  if (!has_attribute(obj, name)) {
    return false;
  }

  struct IntVect3 {
    int i, j, k;
  };

  const hid_t attr = H5Aopen(obj, name.c_str(), H5P_DEFAULT);
  if (attr < 0) {
    CCTK_VERROR("Failed to open HDF5 IntVect attribute '%s'", name.c_str());
  }

  const hid_t memtype = H5Tcreate(H5T_COMPOUND, sizeof(IntVect3));
  if (memtype < 0) {
    CCTK_VERROR("Failed to create memory type for IntVect attribute '%s'",
                name.c_str());
  }
  if (H5Tinsert(memtype, "intvecti", HOFFSET(IntVect3, i), H5T_NATIVE_INT) < 0 ||
      H5Tinsert(memtype, "intvectj", HOFFSET(IntVect3, j), H5T_NATIVE_INT) < 0 ||
      H5Tinsert(memtype, "intvectk", HOFFSET(IntVect3, k), H5T_NATIVE_INT) < 0) {
    CCTK_VERROR("Failed to define memory type for IntVect attribute '%s'",
                name.c_str());
  }

  IntVect3 raw{0, 0, 0};
  if (H5Aread(attr, memtype, &raw) < 0) {
    CCTK_VERROR("Failed to read HDF5 IntVect attribute '%s'", name.c_str());
  }

  if (H5Tclose(memtype) < 0 || H5Aclose(attr) < 0) {
    CCTK_VERROR("Failed to close HDF5 handles for IntVect attribute '%s'",
                name.c_str());
  }

  value = {{raw.i, raw.j, raw.k}};
  return true;
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

std::size_t
real_dataset_size(const hid_t group,
                  const std::initializer_list<const char *> names) {
  const hid_t dset = open_dataset_any(group, names);
  const hid_t space = H5Dget_space(dset);
  if (space < 0) {
    CCTK_VERROR("Failed to get dataspace for real dataset");
  }

  const int ndims = H5Sget_simple_extent_ndims(space);
  if (ndims < 0) {
    CCTK_VERROR("Failed to get rank of real dataset");
  }

  std::vector<hsize_t> dims(ndims > 0 ? ndims : 1, 1);
  if (ndims > 0 && H5Sget_simple_extent_dims(space, dims.data(), nullptr) < 0) {
    CCTK_VERROR("Failed to get dimensions of real dataset");
  }

  std::size_t total = 1;
  for (int d = 0; d < ndims; ++d) {
    total *= static_cast<std::size_t>(dims[d]);
  }

  if (H5Sclose(space) < 0 || H5Dclose(dset) < 0) {
    CCTK_VERROR("Failed to close HDF5 handles for real dataset");
  }
  return total;
}

std::vector<double>
read_real_dataset_slice_from_open_dataset(const hid_t dset,
                                          const std::size_t begin,
                                          const std::size_t count) {
  const hid_t filespace = H5Dget_space(dset);
  if (filespace < 0) {
    CCTK_VERROR("Failed to get dataspace for real dataset slice");
  }

  const int ndims = H5Sget_simple_extent_ndims(filespace);
  if (ndims != 1) {
    CCTK_VERROR(
        "GRTresnaIDX local source-box loading expects a rank-1 data dataset; found rank %d",
        ndims);
  }

  hsize_t dims[1] = {0};
  if (H5Sget_simple_extent_dims(filespace, dims, nullptr) < 0) {
    CCTK_VERROR("Failed to get dimensions of real dataset slice");
  }
  if (begin + count > static_cast<std::size_t>(dims[0])) {
    CCTK_VERROR(
        "Requested source-box slice [%llu,%llu) exceeds data dataset length %llu",
        static_cast<unsigned long long>(begin),
        static_cast<unsigned long long>(begin + count),
        static_cast<unsigned long long>(dims[0]));
  }

  hsize_t start[1] = {static_cast<hsize_t>(begin)};
  hsize_t hcount[1] = {static_cast<hsize_t>(count)};
  if (H5Sselect_hyperslab(filespace, H5S_SELECT_SET, start, nullptr, hcount,
                          nullptr) < 0) {
    CCTK_VERROR("Failed to select source-box hyperslab");
  }

  const hid_t memspace = H5Screate_simple(1, hcount, nullptr);
  if (memspace < 0) {
    CCTK_VERROR("Failed to create source-box memory dataspace");
  }

  std::vector<double> out(count);
  if (H5Dread(dset, H5T_NATIVE_DOUBLE, memspace, filespace, H5P_DEFAULT,
              out.data()) < 0) {
    CCTK_VERROR("Failed to read source-box data slice");
  }

  if (H5Sclose(memspace) < 0 || H5Sclose(filespace) < 0) {
    CCTK_VERROR("Failed to close HDF5 handles for source-box data slice");
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
  if (ndims != 1 && ndims != 2) {
    CCTK_VERROR("Unsupported 'boxes' rank %d; expected rank-1 compound array "
                "or rank-2 integer array",
                ndims);
  }

  hsize_t dims[2] = {0, 0};
  if (H5Sget_simple_extent_dims(space, dims, nullptr) < 0) {
    CCTK_VERROR("Failed to read dimensions of 'boxes' dataset");
  }

  if (ndims == 1) {
    struct ChomboBox {
      int lo_i, lo_j, lo_k;
      int hi_i, hi_j, hi_k;
    };

    const hsize_t nboxes_h = dims[0];
    const size_t nboxes = static_cast<size_t>(nboxes_h);
    std::vector<ChomboBox> raw(nboxes);

    const hid_t memtype = H5Tcreate(H5T_COMPOUND, sizeof(ChomboBox));
    if (memtype < 0) {
      CCTK_VERROR("Failed to create memory type for compound 'boxes' dataset");
    }
    const auto insert_member = [&](const char *name, const size_t offset) {
      if (H5Tinsert(memtype, name, offset, H5T_NATIVE_INT) < 0) {
        CCTK_VERROR("Failed to define compound 'boxes' member '%s'", name);
      }
    };
    insert_member("lo_i", HOFFSET(ChomboBox, lo_i));
    insert_member("lo_j", HOFFSET(ChomboBox, lo_j));
    insert_member("lo_k", HOFFSET(ChomboBox, lo_k));
    insert_member("hi_i", HOFFSET(ChomboBox, hi_i));
    insert_member("hi_j", HOFFSET(ChomboBox, hi_j));
    insert_member("hi_k", HOFFSET(ChomboBox, hi_k));

    if (H5Dread(dset, memtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, raw.data()) < 0) {
      CCTK_VERROR("Failed to read compound 'boxes' dataset");
    }

    std::vector<std::array<int, 6>> boxes(nboxes);
    for (size_t b = 0; b < nboxes; ++b) {
      boxes[b] = {{raw[b].lo_i, raw[b].lo_j, raw[b].lo_k, raw[b].hi_i,
                   raw[b].hi_j, raw[b].hi_k}};
    }

    if (H5Tclose(memtype) < 0 || H5Sclose(space) < 0 || H5Dclose(dset) < 0) {
      CCTK_VERROR("Failed to close HDF5 handles for compound 'boxes' dataset");
    }
    return boxes;
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

std::vector<std::array<std::size_t, 2>>
decode_box_offsets(const std::vector<long long> &offsets, const size_t nboxes,
                   const std::string &group_name) {
  if (!(offsets.size() == nboxes + 1 || offsets.size() == 2 * nboxes)) {
    CCTK_VERROR(
        "Unsupported offsets layout in group '%s': found %llu entries for %llu boxes "
        "(expected nboxes+1 or 2*nboxes)",
        group_name.c_str(), static_cast<unsigned long long>(offsets.size()),
        static_cast<unsigned long long>(nboxes));
  }

  std::vector<std::array<std::size_t, 2>> box_offsets(nboxes);
  for (size_t bi = 0; bi < nboxes; ++bi) {
    const long long begin_ll =
        (offsets.size() == nboxes + 1) ? offsets[bi] : offsets[2 * bi];
    const long long end_ll =
        (offsets.size() == nboxes + 1) ? offsets[bi + 1] : offsets[2 * bi + 1];

    if (begin_ll < 0 || end_ll <= begin_ll) {
      CCTK_VERROR(
          "Invalid offsets [%lld, %lld] for box %llu in group '%s'",
          static_cast<long long>(begin_ll), static_cast<long long>(end_ll),
          static_cast<unsigned long long>(bi), group_name.c_str());
    }

    box_offsets[bi] = {{static_cast<std::size_t>(begin_ll),
                        static_cast<std::size_t>(end_ll)}};
  }

  return box_offsets;
}

inline int clamp_int(const int v, const int lo, const int hi) {
  return std::max(lo, std::min(v, hi));
}

inline double lerp(const double a, const double b, const double w) {
  return a + w * (b - a);
}

double walltime_seconds() {
  using clock = std::chrono::steady_clock;
  static const auto t0 = clock::now();
  const auto now = clock::now();
  return std::chrono::duration<double>(now - t0).count();
}

bool all_finite(const std::initializer_list<double> values) {
  for (const double value : values) {
    if (!std::isfinite(value)) {
      return false;
    }
  }
  return true;
}

bool adm_sample_is_valid(const ADMSample &s, const bool need_metric_curv,
                         const bool need_lapse, const bool need_shift) {
  if (need_metric_curv) {
    if (!all_finite({s.gxx, s.gxy, s.gxz, s.gyy, s.gyz, s.gzz, s.kxx,
                     s.kxy, s.kxz, s.kyy, s.kyz, s.kzz})) {
      return false;
    }

    const double det =
        s.gxx * (s.gyy * s.gzz - s.gyz * s.gyz) -
        s.gxy * (s.gxy * s.gzz - s.gyz * s.gxz) +
        s.gxz * (s.gxy * s.gyz - s.gyy * s.gxz);
    if (!(std::isfinite(det) && det > 0.0)) {
      return false;
    }
  }

  if (need_lapse && !std::isfinite(s.alp)) {
    return false;
  }
  if (need_shift && !all_finite({s.betax, s.betay, s.betaz})) {
    return false;
  }
  return true;
}

} // namespace

GRTresnaReader::GRTresnaReader()
    : loaded_(false), center_{{0.0, 0.0, 0.0}}, num_levels_(0),
      global_lo_{{0, 0, 0}}, global_hi_{{-1, -1, -1}}, ncomp_(0),
      basis_(VariableBasis::adm), idx_gxx_(-1), idx_gxy_(-1), idx_gxz_(-1),
      idx_gyy_(-1), idx_gyz_(-1), idx_gzz_(-1), idx_kxx_(-1), idx_kxy_(-1),
      idx_kxz_(-1), idx_kyy_(-1), idx_kyz_(-1), idx_kzz_(-1), idx_alp_(-1),
      idx_betax_(-1), idx_betay_(-1), idx_betaz_(-1), idx_chi_(-1),
      idx_h11_(-1), idx_h12_(-1), idx_h13_(-1), idx_h22_(-1), idx_h23_(-1),
      idx_h33_(-1), idx_ktrace_(-1), idx_A11_(-1), idx_A12_(-1),
      idx_A13_(-1), idx_A22_(-1), idx_A23_(-1), idx_A33_(-1), idx_lapse_(-1),
      idx_shift1_(-1), idx_shift2_(-1), idx_shift3_(-1) {}

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

bool GRTresnaReader::point_in_box_data(const SourceBox &box, const int i,
                                       const int j, const int k) const {
  return i >= box.lo[0] - box.nghost && i <= box.hi[0] + box.nghost &&
         j >= box.lo[1] - box.nghost && j <= box.hi[1] + box.nghost &&
         k >= box.lo[2] - box.nghost && k <= box.hi[2] + box.nghost;
}

bool GRTresnaReader::box_intersects_index_range(
    const SourceBox &box, const std::array<int, 3> &ilo,
    const std::array<int, 3> &ihi) const {
  return ihi[0] >= box.lo[0] - box.nghost &&
         ilo[0] <= box.hi[0] + box.nghost &&
         ihi[1] >= box.lo[1] - box.nghost &&
         ilo[1] <= box.hi[1] + box.nghost &&
         ihi[2] >= box.lo[2] - box.nghost &&
         ilo[2] <= box.hi[2] + box.nghost;
}

std::uint64_t GRTresnaReader::pack_index3(const int i, const int j,
                                          const int k) {
  // Source indices in known GRTresna/Chombo files are non-negative and small;
  // keep this signed-bias packing robust enough for shifted domains too.
  constexpr int bias = 1 << 20;
  constexpr std::uint64_t mask = (std::uint64_t(1) << 21) - 1;
  return ((static_cast<std::uint64_t>(i + bias) & mask) << 42) |
         ((static_cast<std::uint64_t>(j + bias) & mask) << 21) |
         (static_cast<std::uint64_t>(k + bias) & mask);
}

int GRTresnaReader::floor_div(const int a, const int b) {
  if (b <= 0) {
    return 0;
  }
  int q = a / b;
  const int r = a % b;
  if (r != 0 && ((r < 0) != (b < 0))) {
    --q;
  }
  return q;
}

std::size_t GRTresnaReader::find_box_containing_index(
    const SourceLevel &level, const int i, const int j, const int k) const {
  if (i < level.lo_union[0] || i > level.hi_union[0] || j < level.lo_union[1] ||
      j > level.hi_union[1] || k < level.lo_union[2] ||
      k > level.hi_union[2]) {
    return invalid_box_index;
  }

  if (level.uniform_box_size && level.box_size[0] > 0 &&
      level.box_size[1] > 0 && level.box_size[2] > 0) {
    const int blo0 =
        level.lo_union[0] +
        floor_div(i - level.lo_union[0], level.box_size[0]) *
            level.box_size[0];
    const int blo1 =
        level.lo_union[1] +
        floor_div(j - level.lo_union[1], level.box_size[1]) *
            level.box_size[1];
    const int blo2 =
        level.lo_union[2] +
        floor_div(k - level.lo_union[2], level.box_size[2]) *
            level.box_size[2];
    const auto it =
        level.interior_box_index.find(pack_index3(blo0, blo1, blo2));
    if (it != level.interior_box_index.end() &&
        it->second < level.boxes.size() &&
        point_in_box(level.boxes[it->second], i, j, k)) {
      return it->second;
    }
  }

  for (std::size_t bi = 0; bi < level.boxes.size(); ++bi) {
    if (point_in_box(level.boxes[bi], i, j, k)) {
      return bi;
    }
  }
  return invalid_box_index;
}

std::size_t GRTresnaReader::find_box_with_data_index(
    const SourceLevel &level, const int i, const int j, const int k) const {
  if (i < level.data_lo_union[0] || i > level.data_hi_union[0] ||
      j < level.data_lo_union[1] || j > level.data_hi_union[1] ||
      k < level.data_lo_union[2] || k > level.data_hi_union[2]) {
    return invalid_box_index;
  }

  const std::size_t interior = find_box_containing_index(level, i, j, k);
  if (interior != invalid_box_index) {
    return interior;
  }

  if (level.uniform_box_size && level.box_size[0] > 0 &&
      level.box_size[1] > 0 && level.box_size[2] > 0) {
    const int base0 =
        level.lo_union[0] +
        floor_div(i - level.lo_union[0], level.box_size[0]) *
            level.box_size[0];
    const int base1 =
        level.lo_union[1] +
        floor_div(j - level.lo_union[1], level.box_size[1]) *
            level.box_size[1];
    const int base2 =
        level.lo_union[2] +
        floor_div(k - level.lo_union[2], level.box_size[2]) *
            level.box_size[2];
    for (int dk = -1; dk <= 1; ++dk) {
      for (int dj = -1; dj <= 1; ++dj) {
        for (int di = -1; di <= 1; ++di) {
          const int blo0 = base0 + di * level.box_size[0];
          const int blo1 = base1 + dj * level.box_size[1];
          const int blo2 = base2 + dk * level.box_size[2];
          const auto it =
              level.interior_box_index.find(pack_index3(blo0, blo1, blo2));
          if (it != level.interior_box_index.end() &&
              it->second < level.boxes.size() &&
              point_in_box_data(level.boxes[it->second], i, j, k)) {
            return it->second;
          }
        }
      }
    }
  }

  for (std::size_t bi = 0; bi < level.boxes.size(); ++bi) {
    if (point_in_box_data(level.boxes[bi], i, j, k)) {
      return bi;
    }
  }
  return invalid_box_index;
}

const GRTresnaReader::SourceBox *
GRTresnaReader::find_box_with_data(const SourceLevel &level, const int i,
                                   const int j, const int k) const {
  const std::size_t bi = find_box_with_data_index(level, i, j, k);
  return bi == invalid_box_index ? nullptr : &level.boxes[bi];
}

void GRTresnaReader::load_box_data_locked(const int level_idx,
                                          const std::size_t box_idx,
                                          const hid_t data_dset) const {
  if (level_idx < 0 || level_idx >= num_levels_) {
    CCTK_VERROR("Invalid source level %d while loading local source box",
                level_idx);
  }
  const auto &level = levels_[level_idx];
  if (box_idx >= level.boxes.size()) {
    CCTK_VERROR("Invalid source box %llu on level %d",
                static_cast<unsigned long long>(box_idx), level_idx);
  }
  const auto &box = level.boxes[box_idx];
  if (box.data_loaded.load(std::memory_order_acquire)) {
    return;
  }

  box.data = read_real_dataset_slice_from_open_dataset(
      data_dset, box.begin, box.end - box.begin);
  box.data_loaded.store(true, std::memory_order_release);
}

void GRTresnaReader::load_box_data(const int level_idx,
                                   const std::size_t box_idx) const {
  std::lock_guard<std::mutex> lock(data_mutex_);

  if (level_idx < 0 || level_idx >= num_levels_) {
    CCTK_VERROR("Invalid source level %d while loading local source box",
                level_idx);
  }
  const auto &level = levels_[level_idx];
  if (box_idx >= level.boxes.size()) {
    CCTK_VERROR("Invalid source box %llu on level %d",
                static_cast<unsigned long long>(box_idx), level_idx);
  }
  const auto &box = level.boxes[box_idx];
  if (box.data_loaded.load(std::memory_order_acquire)) {
    return;
  }

  const hid_t file_id = H5Fopen(filename_.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file_id < 0) {
    CCTK_VERROR("Could not reopen GRTresna input file '%s'",
                filename_.c_str());
  }

  const std::string group_name = "level_" + std::to_string(level_idx);
  const hid_t level_group = H5Gopen2(file_id, group_name.c_str(), H5P_DEFAULT);
  if (level_group < 0) {
    CCTK_VERROR("Missing required HDF5 group '%s' in source file",
                group_name.c_str());
  }

  const hid_t data_dset =
      open_dataset_any(level_group, {"data:datatype=0", "data:datatype=1", "data"});
  if (data_dset < 0) {
    CCTK_VERROR("Could not open source data dataset in group '%s'",
                group_name.c_str());
  }

  load_box_data_locked(level_idx, box_idx, data_dset);

  if (H5Dclose(data_dset) < 0 || H5Gclose(level_group) < 0 ||
      H5Fclose(file_id) < 0) {
    CCTK_VERROR("Failed to close HDF5 handles after loading local source box");
  }
}

void GRTresnaReader::prefetch_region(
    const double xmin, const double ymin, const double zmin, const double xmax,
    const double ymax, const double zmax,
    const InterpolationMethod method, const int preferred_source_level) const {
  if (!loaded_) {
    return;
  }

  const double t0 = walltime_seconds();
  const std::array<double, 3> xlo{{std::min(xmin, xmax), std::min(ymin, ymax),
                                   std::min(zmin, zmax)}};
  const std::array<double, 3> xhi{{std::max(xmin, xmax), std::max(ymin, ymax),
                                   std::max(zmin, zmax)}};
  const int margin = method == InterpolationMethod::trilinear ? 2 : 1;

  std::vector<std::vector<std::size_t>> boxes_to_load(
      static_cast<std::size_t>(num_levels_));
  const int lev_begin =
      preferred_source_level >= 0 && preferred_source_level < num_levels_
          ? preferred_source_level
          : 0;
  const int lev_end =
      preferred_source_level >= 0 && preferred_source_level < num_levels_
          ? preferred_source_level + 1
          : num_levels_;

  for (int lev = lev_begin; lev < lev_end; ++lev) {
    const auto &level = levels_[lev];
    std::array<int, 3> ilo;
    std::array<int, 3> ihi;
    for (int d = 0; d < 3; ++d) {
      const double glo = (xlo[d] + center_[d]) / level.dx - 0.5;
      const double ghi = (xhi[d] + center_[d]) / level.dx - 0.5;
      ilo[d] = static_cast<int>(std::floor(std::min(glo, ghi))) - margin;
      ihi[d] = static_cast<int>(std::ceil(std::max(glo, ghi))) + margin;
    }

    if (level.uniform_box_size && level.box_size[0] > 0 &&
        level.box_size[1] > 0 && level.box_size[2] > 0 &&
        !level.interior_box_index.empty()) {
      std::array<int, 3> blo_min;
      std::array<int, 3> blo_max;
      for (int d = 0; d < 3; ++d) {
        blo_min[d] =
            level.lo_union[d] +
            floor_div(ilo[d] - level.lo_union[d], level.box_size[d]) *
                level.box_size[d];
        blo_max[d] =
            level.lo_union[d] +
            floor_div(ihi[d] - level.lo_union[d], level.box_size[d]) *
                level.box_size[d];
        blo_min[d] -= level.box_size[d];
        blo_max[d] += level.box_size[d];
      }

      for (int bk = blo_min[2]; bk <= blo_max[2]; bk += level.box_size[2]) {
        for (int bj = blo_min[1]; bj <= blo_max[1]; bj += level.box_size[1]) {
          for (int bi0 = blo_min[0]; bi0 <= blo_max[0];
               bi0 += level.box_size[0]) {
            const auto it =
                level.interior_box_index.find(pack_index3(bi0, bj, bk));
            if (it == level.interior_box_index.end() ||
                it->second >= level.boxes.size()) {
              continue;
            }
            const auto &box = level.boxes[it->second];
            if (!box.data_loaded.load(std::memory_order_acquire) &&
                box_intersects_index_range(box, ilo, ihi)) {
              boxes_to_load[static_cast<std::size_t>(lev)].push_back(
                  it->second);
            }
          }
        }
      }
    } else {
      for (std::size_t bi = 0; bi < level.boxes.size(); ++bi) {
        const auto &box = level.boxes[bi];
        if (!box.data_loaded.load(std::memory_order_acquire) &&
            box_intersects_index_range(box, ilo, ihi)) {
          boxes_to_load[static_cast<std::size_t>(lev)].push_back(bi);
        }
      }
    }
  }

  std::size_t requested = 0;
  for (const auto &v : boxes_to_load) {
    requested += v.size();
  }
  if (requested == 0) {
    return;
  }

  const double t_scan = walltime_seconds();
  std::size_t newly_loaded = 0;
  {
    std::lock_guard<std::mutex> lock(data_mutex_);

    const hid_t file_id =
        H5Fopen(filename_.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0) {
      CCTK_VERROR("Could not reopen GRTresna input file '%s'",
                  filename_.c_str());
    }

    for (int lev = lev_begin; lev < lev_end; ++lev) {
      const auto &level_boxes = boxes_to_load[static_cast<std::size_t>(lev)];
      if (level_boxes.empty()) {
        continue;
      }

      const std::string group_name = "level_" + std::to_string(lev);
      const hid_t level_group =
          H5Gopen2(file_id, group_name.c_str(), H5P_DEFAULT);
      if (level_group < 0) {
        CCTK_VERROR("Missing required HDF5 group '%s' in source file",
                    group_name.c_str());
      }

      const hid_t data_dset = open_dataset_any(
          level_group, {"data:datatype=0", "data:datatype=1", "data"});
      if (data_dset < 0) {
        CCTK_VERROR("Could not open source data dataset in group '%s'",
                    group_name.c_str());
      }

      for (const std::size_t bi : level_boxes) {
        if (!levels_[lev].boxes[bi].data_loaded.load(
                std::memory_order_acquire)) {
          load_box_data_locked(lev, bi, data_dset);
          ++newly_loaded;
        }
      }

      if (H5Dclose(data_dset) < 0 || H5Gclose(level_group) < 0) {
        CCTK_VERROR("Failed to close HDF5 handles after batch source loading");
      }
    }

    if (H5Fclose(file_id) < 0) {
      CCTK_VERROR("Failed to close HDF5 file after batch source loading");
    }
  }

  if (config_.verbosity >= 2 && newly_loaded > 0) {
    const double t_done = walltime_seconds();
    CCTK_VINFO(
        "GRTresnaIDX prefetched %llu source boxes for local patch (scan %.3fs, read %.3fs)",
        static_cast<unsigned long long>(newly_loaded), t_scan - t0,
        t_done - t_scan);
  }
}

bool GRTresnaReader::cell_overlaps_level(const double x, const double y,
                                         const double z, const double dx,
                                         const double dy, const double dz,
                                         const int level_idx) const {
  if (!loaded_ || level_idx < 0 || level_idx >= num_levels_) {
    return false;
  }

  const auto &level = levels_[level_idx];
  const std::array<double, 3> xc{{x, y, z}};
  const std::array<double, 3> dxc{{dx, dy, dz}};
  std::array<int, 3> ilo;
  std::array<int, 3> ihi;
  for (int d = 0; d < 3; ++d) {
    const double xlo = xc[d] - 0.5 * dxc[d];
    const double xhi = xc[d] + 0.5 * dxc[d];
    ilo[d] = static_cast<int>(std::floor((xlo + center_[d]) / level.dx));
    ihi[d] = static_cast<int>(std::ceil((xhi + center_[d]) / level.dx)) - 1;
  }

  for (const auto &box : level.boxes) {
    if (ihi[0] >= box.lo[0] && ilo[0] <= box.hi[0] &&
        ihi[1] >= box.lo[1] && ilo[1] <= box.hi[1] &&
        ihi[2] >= box.lo[2] && ilo[2] <= box.hi[2]) {
      return true;
    }
  }

  return false;
}

double GRTresnaReader::base_domain_radius() const {
  if (!loaded_ || num_levels_ <= 0) {
    return 0.0;
  }

  const auto &level = levels_[0];
  double radius = 0.0;
  for (int d = 0; d < 3; ++d) {
    const double lower_face =
        static_cast<double>(level.lo_union[d]) * level.dx - center_[d];
    const double upper_face =
        static_cast<double>(level.hi_union[d] + 1) * level.dx - center_[d];
    radius = std::max(radius, std::max(std::abs(lower_face),
                                       std::abs(upper_face)));
  }
  return radius;
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
  const std::size_t box_idx = find_box_with_data_index(level, i, j, k);
  if (box_idx == invalid_box_index) {
    return false;
  }
  const SourceBox *box = &level.boxes[box_idx];
  if (!box->data_loaded.load(std::memory_order_acquire)) {
    load_box_data(level_idx, box_idx);
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
  const size_t npts =
      static_cast<size_t>(box->n_with_ghost[0]) *
      static_cast<size_t>(box->n_with_ghost[1]) *
      static_cast<size_t>(box->n_with_ghost[2]);
  const size_t idx = static_cast<size_t>(comp) * npts + idx_cell;

  if (idx >= box->data.size()) {
    return false;
  }

  out = box->data[idx];
  if (!std::isfinite(out)) {
    return false;
  }
  return true;
}

bool GRTresnaReader::sample_components_at_level(
    const int level_idx, const int i, const int j, const int k,
    const int *components, const int component_count, double *out) const {
  if (level_idx < 0 || level_idx >= num_levels_ || component_count < 0 ||
      components == nullptr || out == nullptr) {
    return false;
  }

  const auto &level = levels_[level_idx];
  const std::size_t box_idx = find_box_with_data_index(level, i, j, k);
  if (box_idx == invalid_box_index) {
    return false;
  }
  const SourceBox *box = &level.boxes[box_idx];
  if (!box->data_loaded.load(std::memory_order_acquire)) {
    load_box_data(level_idx, box_idx);
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
  const size_t npts =
      static_cast<size_t>(box->n_with_ghost[0]) *
      static_cast<size_t>(box->n_with_ghost[1]) *
      static_cast<size_t>(box->n_with_ghost[2]);

  for (int n = 0; n < component_count; ++n) {
    const int comp = components[n];
    if (comp < 0 || comp >= ncomp_) {
      return false;
    }
    const size_t idx = static_cast<size_t>(comp) * npts + idx_cell;
    if (idx >= box->data.size()) {
      return false;
    }
    out[n] = box->data[idx];
    if (!std::isfinite(out[n])) {
      return false;
    }
  }

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
  filename_ = filename;
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
    level.data_size = 0;
    level.box_size = {{0, 0, 0}};
    level.uniform_box_size = true;
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
    level.data_lo_union = {{std::numeric_limits<int>::max(),
                            std::numeric_limits<int>::max(),
                            std::numeric_limits<int>::max()}};
    level.data_hi_union = {{std::numeric_limits<int>::lowest(),
                            std::numeric_limits<int>::lowest(),
                            std::numeric_limits<int>::lowest()}};

    const auto boxes_raw = read_boxes_dataset(level_group);
    if (boxes_raw.empty()) {
      CCTK_VERROR("No boxes found in HDF5 group '%s'", group_name.c_str());
    }

    int metadata_ghost = -1;
    const hid_t data_attr_group =
        H5Gopen2(level_group, "data_attributes", H5P_DEFAULT);
    if (data_attr_group >= 0) {
      const int level_ncomp =
          read_int_attribute(data_attr_group, "comps", false, ncomp_);
      if (level_ncomp != ncomp_) {
        CCTK_VERROR(
            "Component count mismatch in '%s/data_attributes': root num_components=%d, level comps=%d",
            group_name.c_str(), ncomp_, level_ncomp);
      }

      std::array<int, 3> ghost{{0, 0, 0}};
      if (read_intvect_attribute(data_attr_group, "ghost", ghost) ||
          read_intvect_attribute(data_attr_group, "outputGhost", ghost)) {
        if (ghost[0] != ghost[1] || ghost[0] != ghost[2]) {
          CCTK_VERROR(
              "GRTresnaIDX expects isotropic source ghost zones, but group '%s' has ghost=(%d,%d,%d)",
              group_name.c_str(), ghost[0], ghost[1], ghost[2]);
        }
        if (ghost[0] < 0) {
          CCTK_VERROR("Invalid negative source ghost width %d in group '%s'",
                      ghost[0], group_name.c_str());
        }
        metadata_ghost = ghost[0];
      }
      if (H5Gclose(data_attr_group) < 0) {
        CCTK_VERROR("Failed to close HDF5 group '%s/data_attributes'",
                    group_name.c_str());
      }
    }

    const auto offsets =
        read_int64_dataset(level_group, {"data:offsets=0", "data:offsets"});
    const auto box_offsets =
        decode_box_offsets(offsets, boxes_raw.size(), group_name);

    level.data_size =
        real_dataset_size(level_group,
                          {"data:datatype=0", "data:datatype=1", "data"});

    level.boxes.reserve(boxes_raw.size());
    for (size_t bi = 0; bi < boxes_raw.size(); ++bi) {
      const std::size_t begin = box_offsets[bi][0];
      const std::size_t end = box_offsets[bi][1];
      if (end > level.data_size) {
        CCTK_VERROR(
            "Source data too short in group '%s': end=%llu data_size=%llu",
            group_name.c_str(), static_cast<unsigned long long>(end),
            static_cast<unsigned long long>(level.data_size));
      }

      SourceBox box;
      box.lo = {{boxes_raw[bi][0], boxes_raw[bi][1], boxes_raw[bi][2]}};
      box.hi = {{boxes_raw[bi][3], boxes_raw[bi][4], boxes_raw[bi][5]}};
      box.begin = begin;
      box.end = end;
      box.data_loaded.store(false, std::memory_order_release);

      const int nx = box.hi[0] - box.lo[0] + 1;
      const int ny = box.hi[1] - box.lo[1] + 1;
      const int nz = box.hi[2] - box.lo[2] + 1;
      if (nx <= 0 || ny <= 0 || nz <= 0) {
        CCTK_VERROR(
            "Invalid source box extents in '%s': lo=(%d,%d,%d), hi=(%d,%d,%d)",
            group_name.c_str(), box.lo[0], box.lo[1], box.lo[2], box.hi[0],
            box.hi[1], box.hi[2]);
      }
      const std::array<int, 3> this_box_size{{nx, ny, nz}};
      if (bi == 0) {
        level.box_size = this_box_size;
      } else if (level.box_size != this_box_size) {
        level.uniform_box_size = false;
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

      const auto set_ghost_width = [&](const int g) {
        const long long candidate =
            static_cast<long long>(nx + 2 * g) *
            static_cast<long long>(ny + 2 * g) *
            static_cast<long long>(nz + 2 * g);
        if (candidate != cells_with_ghost) {
          return false;
        }
        box.nghost = g;
        box.n_with_ghost = {{nx + 2 * g, ny + 2 * g, nz + 2 * g}};
        return true;
      };

      bool found_ghost = false;
      if (metadata_ghost >= 0) {
        found_ghost = set_ghost_width(metadata_ghost);
        if (!found_ghost) {
          CCTK_VERROR(
              "HDF5 metadata ghost=%d does not match data length for box %llu in '%s': nx=%d ny=%d nz=%d cells_with_ghost=%lld",
              metadata_ghost, static_cast<unsigned long long>(bi),
              group_name.c_str(), nx, ny, nz, cells_with_ghost);
        }
      } else {
        for (int g = 0; g <= 64; ++g) {
          if (set_ghost_width(g)) {
            found_ghost = true;
            break;
          }
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
      level.data_lo_union[0] =
          std::min(level.data_lo_union[0], box.lo[0] - box.nghost);
      level.data_lo_union[1] =
          std::min(level.data_lo_union[1], box.lo[1] - box.nghost);
      level.data_lo_union[2] =
          std::min(level.data_lo_union[2], box.lo[2] - box.nghost);
      level.data_hi_union[0] =
          std::max(level.data_hi_union[0], box.hi[0] + box.nghost);
      level.data_hi_union[1] =
          std::max(level.data_hi_union[1], box.hi[1] + box.nghost);
      level.data_hi_union[2] =
          std::max(level.data_hi_union[2], box.hi[2] + box.nghost);

      level.boxes.push_back(box);
    }

    if (level.uniform_box_size && level.box_size[0] > 0 &&
        level.box_size[1] > 0 && level.box_size[2] > 0) {
      level.interior_box_index.reserve(level.boxes.size());
      for (std::size_t bi = 0; bi < level.boxes.size(); ++bi) {
        const auto &box = level.boxes[bi];
        level.interior_box_index.emplace(
            pack_index3(box.lo[0], box.lo[1], box.lo[2]), bi);
      }
    }

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

  loaded_ = true;

  if (config_.verbosity >= 1) {
    size_t total_boxes = 0;
    std::size_t total_values = 0;
    for (const auto &level : levels_) {
      total_boxes += level.boxes.size();
      total_values += level.data_size;
    }

    CCTK_VINFO(
        "GRTresnaIDX loaded metadata for '%s': levels=%d, boxes=%llu, ncomp=%d, dx0=%e, source_values=%llu, "
        "coarse_box=[(%d,%d,%d)->(%d,%d,%d)], basis=%s",
        filename.c_str(), num_levels_,
        static_cast<unsigned long long>(total_boxes), ncomp_, levels_[0].dx,
        static_cast<unsigned long long>(total_values),
        global_lo_[0], global_lo_[1], global_lo_[2], global_hi_[0],
        global_hi_[1], global_hi_[2],
        (basis_ == VariableBasis::adm ? "ADM" : "BSSN-like"));
  }
}

bool GRTresnaReader::sample_adm_nearest(
    const double x, const double y, const double z,
    const OutOfBoundsPolicy oob_policy, ADMSample &out,
    const bool need_metric_curv, const bool need_lapse, const bool need_shift,
    const int preferred_source_level) const {
  const auto try_level = [&](const int lev, const bool allow_clamp) {
    const auto &level = levels_[lev];
    int ix = static_cast<int>(std::llround((x + center_[0]) / level.dx - 0.5));
    int iy = static_cast<int>(std::llround((y + center_[1]) / level.dx - 0.5));
    int iz = static_cast<int>(std::llround((z + center_[2]) / level.dx - 0.5));

    if (allow_clamp) {
      ix = clamp_int(ix, level.lo_union[0], level.hi_union[0]);
      iy = clamp_int(iy, level.lo_union[1], level.hi_union[1]);
      iz = clamp_int(iz, level.lo_union[2], level.hi_union[2]);
    } else if (ix < level.lo_union[0] || ix > level.hi_union[0] ||
               iy < level.lo_union[1] || iy > level.hi_union[1] ||
               iz < level.lo_union[2] || iz > level.hi_union[2]) {
      return false;
    }

    std::array<int, 18> components{};
    std::array<double, 18> values{};
    int ncomp = 0;
    const auto add = [&](const int comp) {
      if (ncomp >= static_cast<int>(components.size())) {
        return false;
      }
      components[ncomp++] = comp;
      return true;
    };

    if (basis_ == VariableBasis::adm) {
      if (need_metric_curv) {
        if (!add(idx_gxx_) || !add(idx_gxy_) || !add(idx_gxz_) ||
            !add(idx_gyy_) || !add(idx_gyz_) || !add(idx_gzz_) ||
            !add(idx_kxx_) || !add(idx_kxy_) || !add(idx_kxz_) ||
            !add(idx_kyy_) || !add(idx_kyz_) || !add(idx_kzz_)) {
          return false;
        }
      }
      if (need_lapse && !add(idx_alp_)) {
        return false;
      }
      if (need_shift &&
          (!add(idx_betax_) || !add(idx_betay_) || !add(idx_betaz_))) {
        return false;
      }
    } else {
      if (need_lapse && !add(idx_lapse_)) {
        return false;
      }
      if (need_shift &&
          (!add(idx_shift1_) || !add(idx_shift2_) || !add(idx_shift3_))) {
        return false;
      }
      if (need_metric_curv) {
        if (!add(idx_chi_) || !add(idx_h11_) || !add(idx_h12_) ||
            !add(idx_h13_) || !add(idx_h22_) || !add(idx_h23_) ||
            !add(idx_h33_) || !add(idx_ktrace_) || !add(idx_A11_) ||
            !add(idx_A12_) || !add(idx_A13_) || !add(idx_A22_) ||
            !add(idx_A23_) || !add(idx_A33_)) {
          return false;
        }
      }
    }

    if (!sample_components_at_level(lev, ix, iy, iz, components.data(), ncomp,
                                    values.data())) {
      return false;
    }

    int pos = 0;
    if (basis_ == VariableBasis::adm) {
      if (need_metric_curv) {
        out.gxx = values[pos++];
        out.gxy = values[pos++];
        out.gxz = values[pos++];
        out.gyy = values[pos++];
        out.gyz = values[pos++];
        out.gzz = values[pos++];
        out.kxx = values[pos++];
        out.kxy = values[pos++];
        out.kxz = values[pos++];
        out.kyy = values[pos++];
        out.kyz = values[pos++];
        out.kzz = values[pos++];
      }
      if (need_lapse) {
        out.alp = values[pos++];
      }
      if (need_shift) {
        out.betax = values[pos++];
        out.betay = values[pos++];
        out.betaz = values[pos++];
      }
    } else {
      if (need_lapse) {
        out.alp = values[pos++];
      }
      if (need_shift) {
        out.betax = values[pos++];
        out.betay = values[pos++];
        out.betaz = values[pos++];
      }
      if (need_metric_curv) {
        const double chi_raw = values[pos++];
        const double h11 = values[pos++];
        const double h12 = values[pos++];
        const double h13 = values[pos++];
        const double h22 = values[pos++];
        const double h23 = values[pos++];
        const double h33 = values[pos++];
        const double ktrace = values[pos++];
        const double A11 = values[pos++];
        const double A12 = values[pos++];
        const double A13 = values[pos++];
        const double A22 = values[pos++];
        const double A23 = values[pos++];
        const double A33 = values[pos++];

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
      }
    }

    return adm_sample_is_valid(out, need_metric_curv, need_lapse, need_shift);
  };

  if (preferred_source_level >= 0 && preferred_source_level < num_levels_ &&
      try_level(preferred_source_level, false)) {
    return true;
  }

  for (int lev = num_levels_ - 1; lev >= 0; --lev) {
    if (lev == preferred_source_level) {
      continue;
    }
    if (try_level(lev, false)) {
      return true;
    }
  }

  if (oob_policy == OutOfBoundsPolicy::error) {
    return false;
  }

  for (int lev = num_levels_ - 1; lev >= 0; --lev) {
    if (try_level(lev, true)) {
      return true;
    }
  }

  return false;
}

bool GRTresnaReader::sample_adm_asymptotic(
    const double x, const double y, const double z,
    const InterpolationMethod method, ADMSample &out,
    const bool need_metric_curv, const bool need_lapse, const bool need_shift,
    const int preferred_source_level) const {
  if (!loaded_ || levels_.empty()) {
    return false;
  }

  const auto &base = levels_[0];
  const double pos[3] = {x, y, z};
  double lambda = 1.0;
  for (int d = 0; d < 3; ++d) {
    const double lower =
        static_cast<double>(base.lo_union[d]) * base.dx - center_[d];
    const double upper =
        static_cast<double>(base.hi_union[d] + 1) * base.dx - center_[d];

    if (pos[d] > upper) {
      if (!(pos[d] > 0.0)) {
        return false;
      }
      lambda = std::min(lambda, upper / pos[d]);
    } else if (pos[d] < lower) {
      if (!(pos[d] < 0.0)) {
        return false;
      }
      lambda = std::min(lambda, lower / pos[d]);
    }
  }

  if (!(std::isfinite(lambda) && lambda > 0.0 && lambda <= 1.0)) {
    return false;
  }

  const double xb = lambda * x;
  const double yb = lambda * y;
  const double zb = lambda * z;
  const double r = std::sqrt(x * x + y * y + z * z);
  const double rb = std::sqrt(xb * xb + yb * yb + zb * zb);
  const double falloff =
      (r > 0.0 && rb > 0.0) ? std::min(1.0, rb / r) : 1.0;
  if (!(std::isfinite(falloff) && falloff >= 0.0 && falloff <= 1.0)) {
    return false;
  }

  ADMSample boundary{};
  const int boundary_level =
      preferred_source_level >= 0 && preferred_source_level < num_levels_
          ? preferred_source_level
          : 0;
  if (!sample_adm(xb, yb, zb, method, OutOfBoundsPolicy::clamp, boundary,
                  need_metric_curv, need_lapse, need_shift, boundary_level)) {
    return false;
  }

  const double metric_falloff = falloff;
  const double curv_falloff = falloff * falloff;
  const double shift_falloff = falloff * falloff;

  if (need_metric_curv) {
    out.gxx = 1.0 + (boundary.gxx - 1.0) * metric_falloff;
    out.gxy = boundary.gxy * metric_falloff;
    out.gxz = boundary.gxz * metric_falloff;
    out.gyy = 1.0 + (boundary.gyy - 1.0) * metric_falloff;
    out.gyz = boundary.gyz * metric_falloff;
    out.gzz = 1.0 + (boundary.gzz - 1.0) * metric_falloff;

    out.kxx = boundary.kxx * curv_falloff;
    out.kxy = boundary.kxy * curv_falloff;
    out.kxz = boundary.kxz * curv_falloff;
    out.kyy = boundary.kyy * curv_falloff;
    out.kyz = boundary.kyz * curv_falloff;
    out.kzz = boundary.kzz * curv_falloff;
  }

  if (need_lapse) {
    out.alp = 1.0 + (boundary.alp - 1.0) * metric_falloff;
  }
  if (need_shift) {
    out.betax = boundary.betax * shift_falloff;
    out.betay = boundary.betay * shift_falloff;
    out.betaz = boundary.betaz * shift_falloff;
  }

  return adm_sample_is_valid(out, need_metric_curv, need_lapse, need_shift);
}

double GRTresnaReader::sample_component_nearest(
    const int comp, const double x, const double y, const double z,
    const OutOfBoundsPolicy oob_policy, bool &ok,
    const int preferred_source_level) const {
  if (preferred_source_level >= 0 && preferred_source_level < num_levels_) {
    double value = 0.0;
    const auto try_preferred = [&]() -> bool {
      const auto &level = levels_[preferred_source_level];
      const int ix = static_cast<int>(
          std::llround((x + center_[0]) / level.dx - 0.5));
      const int iy = static_cast<int>(
          std::llround((y + center_[1]) / level.dx - 0.5));
      const int iz = static_cast<int>(
          std::llround((z + center_[2]) / level.dx - 0.5));
      if (ix < level.lo_union[0] || ix > level.hi_union[0] ||
          iy < level.lo_union[1] || iy > level.hi_union[1] ||
          iz < level.lo_union[2] || iz > level.hi_union[2]) {
        return false;
      }
      return sample_component_at_level(preferred_source_level, ix, iy, iz,
                                       comp, value);
    };
    if (try_preferred()) {
      return value;
    }
  }

  for (int lev = num_levels_ - 1; lev >= 0; --lev) {
    if (lev == preferred_source_level) {
      continue;
    }
    const auto &level = levels_[lev];
    const int ix = static_cast<int>(
        std::llround((x + center_[0]) / level.dx - 0.5));
    const int iy = static_cast<int>(
        std::llround((y + center_[1]) / level.dx - 0.5));
    const int iz = static_cast<int>(
        std::llround((z + center_[2]) / level.dx - 0.5));

    if (ix < level.lo_union[0] || ix > level.hi_union[0] ||
        iy < level.lo_union[1] || iy > level.hi_union[1] ||
        iz < level.lo_union[2] || iz > level.hi_union[2]) {
      continue;
    }

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
    const OutOfBoundsPolicy oob_policy, bool &ok,
    const int preferred_source_level) const {
  int lev_sel = -1;
  int ix0 = 0, iy0 = 0, iz0 = 0;
  double wx = 0.0, wy = 0.0, wz = 0.0;

  const auto set_axis_stencil = [](const double g, const int lo, const int hi,
                                   const bool allow_clamp, int &i0,
                                   double &w) {
    if (!allow_clamp && (g < static_cast<double>(lo) ||
                         g > static_cast<double>(hi))) {
      return false;
    }

    i0 = static_cast<int>(std::floor(g));
    w = g - static_cast<double>(i0);

    if (hi <= lo) {
      i0 = lo;
      w = 0.0;
      return allow_clamp || g == static_cast<double>(lo);
    }

    if (i0 < lo) {
      if (!allow_clamp) {
        return false;
      }
      i0 = lo;
      w = 0.0;
    }
    if (i0 >= hi) {
      if (!allow_clamp && g > static_cast<double>(hi)) {
        return false;
      }
      i0 = hi - 1;
      w = 1.0;
    }

    return true;
  };

  const auto stencil_available = [&](const SourceLevel &level, const int i0,
                                     const int j0, const int k0,
                                     const double twx, const double twy,
                                     const double twz) {
    constexpr double exact_eps = 1.0e-12;
    const bool exact_x = twx <= exact_eps || twx >= 1.0 - exact_eps;
    const bool exact_y = twy <= exact_eps || twy >= 1.0 - exact_eps;
    const bool exact_z = twz <= exact_eps || twz >= 1.0 - exact_eps;
    if (exact_x && exact_y && exact_z) {
      const int ie = twx >= 1.0 - exact_eps ? i0 + 1 : i0;
      const int je = twy >= 1.0 - exact_eps ? j0 + 1 : j0;
      const int ke = twz >= 1.0 - exact_eps ? k0 + 1 : k0;
      return find_box_with_data(level, ie, je, ke) != nullptr;
    }

    for (int dk = 0; dk < 2; ++dk) {
      for (int dj = 0; dj < 2; ++dj) {
        for (int di = 0; di < 2; ++di) {
          if (find_box_with_data(level, i0 + di, j0 + dj, k0 + dk) ==
              nullptr) {
            return false;
          }
        }
      }
    }
    return true;
  };

  const auto try_select_level = [&](const int lev, const bool allow_clamp) {
    const auto &level = levels_[lev];
    const double gx = (x + center_[0]) / level.dx - 0.5;
    const double gy = (y + center_[1]) / level.dx - 0.5;
    const double gz = (z + center_[2]) / level.dx - 0.5;
    int tx0 = 0, ty0 = 0, tz0 = 0;
    double twx = 0.0, twy = 0.0, twz = 0.0;
    if (!set_axis_stencil(gx, level.lo_union[0], level.hi_union[0],
                          allow_clamp, tx0, twx) ||
        !set_axis_stencil(gy, level.lo_union[1], level.hi_union[1],
                          allow_clamp, ty0, twy) ||
        !set_axis_stencil(gz, level.lo_union[2], level.hi_union[2],
                          allow_clamp, tz0, twz)) {
      return false;
    }

    if (!stencil_available(level, tx0, ty0, tz0, twx, twy, twz)) {
      return false;
    }

    lev_sel = lev;
    ix0 = tx0;
    iy0 = ty0;
    iz0 = tz0;
    wx = twx;
    wy = twy;
    wz = twz;
    return true;
  };

  if (preferred_source_level >= 0 && preferred_source_level < num_levels_) {
    try_select_level(preferred_source_level, false);
  }

  if (lev_sel < 0) {
    for (int lev = num_levels_ - 1; lev >= 0; --lev) {
      if (lev == preferred_source_level) {
        continue;
      }
      if (try_select_level(lev, false)) {
        break;
      }
    }
  }

  if (lev_sel < 0) {
    if (oob_policy == OutOfBoundsPolicy::error) {
      ok = false;
      return 0.0;
    }

    if (!try_select_level(0, true)) {
      ok = false;
      return 0.0;
    }
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
    return sample_component_at_level(lev_sel, i, j, k, comp, value);
  };

  constexpr double exact_eps = 1.0e-12;
  const bool exact_x = wx <= exact_eps || wx >= 1.0 - exact_eps;
  const bool exact_y = wy <= exact_eps || wy >= 1.0 - exact_eps;
  const bool exact_z = wz <= exact_eps || wz >= 1.0 - exact_eps;
  if (exact_x && exact_y && exact_z) {
    const int ie = wx >= 1.0 - exact_eps ? ix1 : ix0;
    const int je = wy >= 1.0 - exact_eps ? iy1 : iy0;
    const int ke = wz >= 1.0 - exact_eps ? iz1 : iz0;
    double value = 0.0;
    if (!fetch_value(ie, je, ke, value)) {
      return sample_component_nearest(comp, x, y, z, oob_policy, ok,
                                      preferred_source_level);
    }
    return value;
  }

  double c000 = 0.0, c100 = 0.0, c010 = 0.0, c110 = 0.0;
  double c001 = 0.0, c101 = 0.0, c011 = 0.0, c111 = 0.0;

  if (!fetch_value(ix0, iy0, iz0, c000) || !fetch_value(ix1, iy0, iz0, c100) ||
      !fetch_value(ix0, iy1, iz0, c010) || !fetch_value(ix1, iy1, iz0, c110) ||
      !fetch_value(ix0, iy0, iz1, c001) || !fetch_value(ix1, iy0, iz1, c101) ||
      !fetch_value(ix0, iy1, iz1, c011) || !fetch_value(ix1, iy1, iz1, c111)) {
    return sample_component_nearest(comp, x, y, z, oob_policy, ok,
                                    preferred_source_level);
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
                                        bool &ok,
                                        const int preferred_source_level) const {
  if (!loaded_) {
    ok = false;
    return 0.0;
  }

  if (method == InterpolationMethod::nearest) {
    return sample_component_nearest(comp, x, y, z, oob_policy, ok,
                                    preferred_source_level);
  }
  return sample_component_trilinear(comp, x, y, z, oob_policy, ok,
                                    preferred_source_level);
}

bool GRTresnaReader::sample_adm(const double x, const double y, const double z,
                                const InterpolationMethod method,
                                const OutOfBoundsPolicy oob_policy,
                                ADMSample &out,
                                const bool need_metric_curv,
                                const bool need_lapse,
                                const bool need_shift,
                                const int preferred_source_level) const {
  bool ok = true;

  if (!(need_metric_curv || need_lapse || need_shift)) {
    return true;
  }

  if (oob_policy == OutOfBoundsPolicy::asymptotic) {
    bool outside_base_domain = false;
    if (loaded_ && !levels_.empty()) {
      const auto &base = levels_[0];
      const double pos[3] = {x, y, z};
      for (int d = 0; d < 3; ++d) {
        const double lower =
            static_cast<double>(base.lo_union[d]) * base.dx - center_[d];
        const double upper =
            static_cast<double>(base.hi_union[d] + 1) * base.dx - center_[d];
        outside_base_domain |= pos[d] < lower || pos[d] > upper;
      }
    }
    if (outside_base_domain) {
      return sample_adm_asymptotic(x, y, z, method, out, need_metric_curv,
                                   need_lapse, need_shift,
                                   preferred_source_level);
    }

    if (sample_adm(x, y, z, method, OutOfBoundsPolicy::error, out,
                   need_metric_curv, need_lapse, need_shift,
                   preferred_source_level)) {
      return true;
    }
    return sample_adm_asymptotic(x, y, z, method, out, need_metric_curv,
                                 need_lapse, need_shift,
                                 preferred_source_level);
  }

  if (method == InterpolationMethod::nearest) {
    return sample_adm_nearest(x, y, z, oob_policy, out, need_metric_curv,
                              need_lapse, need_shift, preferred_source_level);
  }

  const auto sample = [&](const int comp) {
    return sample_component(comp, x, y, z, method, oob_policy, ok,
                            preferred_source_level);
  };

  if (basis_ == VariableBasis::adm) {
    if (need_metric_curv) {
      out.gxx = sample(idx_gxx_);
      out.gxy = sample(idx_gxy_);
      out.gxz = sample(idx_gxz_);
      out.gyy = sample(idx_gyy_);
      out.gyz = sample(idx_gyz_);
      out.gzz = sample(idx_gzz_);

      out.kxx = sample(idx_kxx_);
      out.kxy = sample(idx_kxy_);
      out.kxz = sample(idx_kxz_);
      out.kyy = sample(idx_kyy_);
      out.kyz = sample(idx_kyz_);
      out.kzz = sample(idx_kzz_);
    }

    if (need_lapse) {
      out.alp = sample(idx_alp_);
    }
    if (need_shift) {
      out.betax = sample(idx_betax_);
      out.betay = sample(idx_betay_);
      out.betaz = sample(idx_betaz_);
    }
    if (!ok) {
      return false;
    }
    if (!adm_sample_is_valid(out, need_metric_curv, need_lapse, need_shift)) {
      if (method == InterpolationMethod::trilinear) {
        return sample_adm(x, y, z, InterpolationMethod::nearest, oob_policy,
                          out, need_metric_curv, need_lapse, need_shift,
                          preferred_source_level);
      }
      return false;
    }
    return true;
  }

  if (need_lapse) {
    out.alp = sample(idx_lapse_);
  }
  if (need_shift) {
    out.betax = sample(idx_shift1_);
    out.betay = sample(idx_shift2_);
    out.betaz = sample(idx_shift3_);
  }
  if (!ok) {
    return false;
  }

  if (need_metric_curv) {
    const double chi_raw = sample(idx_chi_);
    const double h11 = sample(idx_h11_);
    const double h12 = sample(idx_h12_);
    const double h13 = sample(idx_h13_);
    const double h22 = sample(idx_h22_);
    const double h23 = sample(idx_h23_);
    const double h33 = sample(idx_h33_);

    const double ktrace = sample(idx_ktrace_);
    const double A11 = sample(idx_A11_);
    const double A12 = sample(idx_A12_);
    const double A13 = sample(idx_A13_);
    const double A22 = sample(idx_A22_);
    const double A23 = sample(idx_A23_);
    const double A33 = sample(idx_A33_);
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
  }

  if (!ok) {
    return false;
  }
  if (!adm_sample_is_valid(out, need_metric_curv, need_lapse, need_shift)) {
    if (method == InterpolationMethod::trilinear) {
      return sample_adm(x, y, z, InterpolationMethod::nearest, oob_policy, out,
                        need_metric_curv, need_lapse, need_shift,
                        preferred_source_level);
    }
    return false;
  }

  return true;
}

} // namespace GRTresnaIDX
