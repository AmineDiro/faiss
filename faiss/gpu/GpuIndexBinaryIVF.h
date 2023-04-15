/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/Clustering.h>
#include <faiss/IndexBinary.h>
#include <faiss/gpu/GpuIndex.h>
#include <faiss/gpu/GpuResources.h>
#include <faiss/invlists/DirectMap.h>
#include <faiss/invlists/InvertedLists.h>
#include <faiss/utils/Heap.h>
#include <memory>

namespace faiss {
namespace gpu {

class BinaryFlatIndex;

struct GpuIndexBinaryIVFConfig : public GpuIndexConfig {};

/** GPU version of IndexBinaryIVF
 *
 * In the inverted file, the quantizer (an IndexBinary instance) provides a
 * quantization index for each vector to be added. The quantization
 * index maps to a list (aka inverted list or posting list), where the
 * id of the vector is stored.
 *
 * Otherwise the object is similar to the IndexIVF
 */
class GpuIndexBinaryIVF : public IndexBinary {
   public:
    /// Version that takes a coarse quantizer instance. The GpuIndexIVF does not
    /// own the coarseQuantizer instance by default (functions like IndexIVF).
    GpuIndexBinaryIVF(
            GpuResourcesProvider* provider,
            IndexBinary* coarseQuantizer,
            size_t dims,
            idx_t nlist,
            GpuIndexBinaryIVFConfig config = GpuIndexBinaryIVFConfig());

    ~GpuIndexBinaryIVF() override;

    // TODO : Basically a Level1Quantizer
    InvertedLists* invlists = nullptr;
    bool own_invlists = true;

    size_t nprobe = 1;    ///< number of probes at query time
    size_t max_codes = 0; ///< max nb of codes to visit to do a query

    /** Select between using a heap or counting to select the k smallest values
     * when scanning inverted lists.
     */
    bool use_heap = true;

    /** collect computations per batch */
    bool per_invlist_search = false;

    /// map for direct access to the elements. Enables reconstruct().
    DirectMap direct_map;

    /// quantizer that maps vectors to inverted lists
    IndexBinary* quantizer = nullptr;

    /// number of possible key values
    size_t nlist = 0;

    /// whether object owns the quantizer
    bool own_fields = false;

    ClusteringParameters cp; ///< to override default clustering params

    /// to override index used during clustering
    Index* clustering_index = nullptr;

    /// Returns the device that this index is resident on
    int getDevice() const;

    /// Returns a reference to our GpuResources object that manages memory,
    /// stream and handle resources on the GPU
    std::shared_ptr<GpuResources> getResources();

    void add(faiss::idx_t n, const uint8_t* x) override;

    void reset() override;

    void search(
            idx_t n,
            const uint8_t* x,
            // faiss::IndexBinary has idx_t for k
            idx_t k,
            int32_t* distances,
            faiss::idx_t* labels,
            const faiss::SearchParameters* params = nullptr) const override;

   protected:
    /// Called from search when the input data is on the CPU;
    /// potentially allows for pinned memory usage
    void searchFromCpuPaged_(
            idx_t n,
            const uint8_t* x,
            int k,
            int32_t* outDistancesData,
            idx_t* outIndicesData) const;

    void searchNonPaged_(
            idx_t n,
            const uint8_t* x,
            int k,
            int32_t* outDistancesData,
            idx_t* outIndicesData) const;

   protected:
    /// Manages streans, cuBLAS handles and scratch memory for devices
    std::shared_ptr<GpuResources> resources_;

    /// Configuration options
    const GpuIndexBinaryIVFConfig binaryIVFConfig_;

    /// Holds our GPU data containing the list of vectors
    std::unique_ptr<BinaryFlatIndex> data_;
};

} // namespace gpu
} // namespace faiss
