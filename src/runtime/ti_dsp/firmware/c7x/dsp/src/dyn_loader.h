/*
 * C7x Compute Service - Dynamic ELF Loader
 *
 * Wraps TI's DLOAD library to load dynamically-linked .out files
 * on the C7x DSP at runtime. Loaded modules resolve symbols against
 * the firmware's export table (TVM runtime, libc, MCU+ SDK).
 */

#ifndef DYN_LOADER_H
#define DYN_LOADER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Initialize the dynamic loader subsystem.
 *
 * Must be called once at startup before any load/unload operations.
 *
 * @return 0 on success, negative error code on failure
 */
int32_t dyn_loader_init(void);

/**
 * Load an ELF shared object from memory.
 *
 * The ELF data must be accessible at elf_addr (typically the shared
 * input buffer). Segments are allocated in DDR and relocated.
 * Symbols are resolved against the firmware export table.
 *
 * @param elf_addr   Address of the ELF data in memory
 * @param elf_size   Size of the ELF data in bytes
 * @param handle_out Output: opaque handle for the loaded module
 *
 * @return 0 on success, negative error code on failure
 */
int32_t dyn_loader_load(uint64_t elf_addr, uint32_t elf_size,
                        uint32_t *handle_out);

/**
 * Look up a symbol in a loaded module.
 *
 * @param handle   Module handle from dyn_loader_load()
 * @param name     Symbol name (e.g. "cg_main_dsp")
 * @param addr_out Output: address of the symbol
 *
 * @return 0 on success, negative error code if not found
 */
int32_t dyn_loader_query_symbol(uint32_t handle, const char *name,
                                uint64_t *addr_out);

/**
 * Unload a previously loaded module.
 *
 * Frees all segments allocated for the module.
 *
 * @param handle  Module handle from dyn_loader_load()
 *
 * @return 0 on success, negative error code on failure
 */
int32_t dyn_loader_unload(uint32_t handle);

/**
 * Deinitialize the dynamic loader subsystem.
 *
 * Unloads all modules and frees resources.
 */
void dyn_loader_deinit(void);

#ifdef __cplusplus
}
#endif

#endif /* DYN_LOADER_H */
