/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*
 * Top-k value+index selection along the innermost axis, batched over any
 * leading dimensions. Backs relax.topk for the c_static/C7x backend, which
 * has no TVM runtime linked in and so cannot use topi's default
 * tvm.contrib.sort.topk (a packed-function call) -- see
 * ti_c7x_topk_legalize.py for how this kernel is wired in via call_extern.
 *
 * Algorithm ported from TI's Relay-era C7x runtime support
 * (topk_impl in tvm_tidl_sort.cc, TI's neo-tvm tree), simplified for this
 * use: a single dtype pair (float32 values / int64 indices -- the
 * quantizer fix in c7x_mma_quantizer.py keeps the entire region feeding
 * relax.topk in float, so no other dtype is needed here), a single
 * innermost axis (every real call site uses axis=-1), descending order
 * only (largest=True), and this runtime's own DDR scratch allocator
 * (TVMBackendAllocWorkspace) in place of the original AllocDDRContext.
 *
 * Maintains a size-k min-heap of (value, index) pairs seeded with the
 * first k elements of each row, then scans the remaining n-k elements,
 * replacing the heap's root whenever a larger value is found -- O(n log k)
 * per row, well within budget even for the largest real shape here
 * (n=24000, k=300).
 */

#include <stdint.h>

#include <algorithm>
#include <utility>

extern "C" void* TVMBackendAllocWorkspace(int device_type, int device_id, uint64_t nbytes,
                                           int dtype_code_hint, int dtype_bits_hint);
extern "C" int TVMBackendFreeWorkspace(int device_type, int device_id, void* ptr);

namespace {

struct Workspace {
  void* ptr = nullptr;
  ~Workspace() {
    if (ptr) TVMBackendFreeWorkspace(1, 0, ptr);
  }
  void* alloc(int64_t nbytes) {
    ptr = TVMBackendAllocWorkspace(1, 0, static_cast<uint64_t>(nbytes), 0, 8);
    return ptr;
  }
};

using ValIdx = std::pair<float, int64_t>;

/* Top-k of one row of `n` (value, index) pairs, in place: on return,
 * row[0..k) holds the k largest pairs, sorted descending by value. */
void TopkOneRow(ValIdx* row, int32_t n, int32_t k) {
  auto cmp = [](const ValIdx& a, const ValIdx& b) { return a.first > b.first; };

  std::make_heap(row, row + k, cmp);
  float curr_min = row[0].first;

  for (int32_t i = k; i < n; i++) {
    float val = row[i].first;
    if (val > curr_min) {
      std::pop_heap(row, row + k, cmp);
      row[k - 1] = ValIdx(val, i);
      std::push_heap(row, row + k, cmp);
      curr_min = row[0].first;
    }
  }

  std::sort_heap(row, row + k, cmp);
}

}  // namespace

extern "C" int32_t c7x_topk(const void* data_ptr, void* out_val_ptr, void* out_idx_ptr,
                             int32_t batch, int32_t n, int32_t k) {
  const float* data = static_cast<const float*>(data_ptr);
  float* out_val = static_cast<float*>(out_val_ptr);
  int64_t* out_idx = static_cast<int64_t*>(out_idx_ptr);

  if (!data || !out_val || !out_idx || k <= 0 || k > n) {
    return -1;
  }

  Workspace ws;
  ValIdx* scratch = static_cast<ValIdx*>(ws.alloc(static_cast<int64_t>(n) * sizeof(ValIdx)));
  if (!scratch) {
    return -1;
  }

  for (int32_t b = 0; b < batch; b++) {
    const float* row = data + static_cast<int64_t>(b) * n;
    for (int32_t i = 0; i < n; i++) {
      scratch[i] = ValIdx(row[i], i);
    }
    TopkOneRow(scratch, n, k);
    float* val_row = out_val + static_cast<int64_t>(b) * k;
    int64_t* idx_row = out_idx + static_cast<int64_t>(b) * k;
    for (int32_t i = 0; i < k; i++) {
      val_row[i] = scratch[i].first;
      idx_row[i] = scratch[i].second;
    }
  }

  return 0;
}
