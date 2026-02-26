/*
 * C7x Compute Service - Command Line Tool
 *
 * Usage:
 *   c7x_compute ping                                  Test connectivity
 *   c7x_compute status                                Get service status
 *   c7x_compute model-load <weights.bin>               Load TVM model weights
 *   c7x_compute model-unload <model_id>                Unload model weights
 *   c7x_compute load <lib0.out>                        Load dynamic module
 *   c7x_compute unload <handle>                        Unload dynamic module
 *   c7x_compute infer <handle> <model_id> --input <in> --output <out>
 *   c7x_compute run --module <lib0.out> --input <in>   Load+infer+unload (JSON)
 *   c7x_compute trace                                  Monitor trace buffer
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <memory>
#include <getopt.h>
#include <signal.h>
#include <unistd.h>
#include <fcntl.h>
#include <dirent.h>
#include <sys/mman.h>

#include "c7x_compute_client.h"
#include "c7x_compute_protocol.h"
#include "raii.h"

#define C7X_DSP_DEVICE_ADDR "7e000000.dsp"

static volatile int g_running = 1;

static void signal_handler(int sig)
{
    (void)sig;
    g_running = 0;
}

static void print_usage(const char *prog)
{
    printf("C7x Compute Service - Command Line Tool\n");
    printf("\n");
    printf("Usage: %s <command> [options]\n", prog);
    printf("\n");
    printf("Commands:\n");
    printf("  ping                              Test connectivity with DSP\n");
    printf("  status                            Get service status\n");
    printf("  model-load <weights.bin>          Load TVM model weights\n");
    printf("  model-unload <model_id>           Unload model weights\n");
    printf("  load <lib0.out>                   Load dynamic ELF module\n");
    printf("  unload <handle>                   Unload dynamic module\n");
    printf("  infer <handle> <model_id>         Run TVM inference\n");
    printf("  run                               Load, infer, and unload (JSON output)\n");
    printf("  trace                             Monitor DSP trace buffer\n");
    printf("\n");
    printf("Infer options:\n");
    printf("  --input <file>         Input tensor data file\n");
    printf("  --output <file>        Output tensor data file\n");
    printf("  --shape <dims>         Input shape (e.g. 1,3,224,224)\n");
    printf("  --dtype <type>         Data type: float32, float16, int8 (default: float32)\n");
    printf("\n");
    printf("Run options:\n");
    printf("  --module <file>        ELF module (lib0.out, with embedded weights)\n");
    printf("  --input <file>         Input tensor binary\n");
    printf("  --output <file>        Output tensor binary\n");
    printf("  --shape <dims>         Input shape (e.g. 1,3,224,224)\n");
    printf("  --dtype <type>         float32, float16, int8 (default: float32)\n");
    printf("\n");
    printf("Examples:\n");
    printf("  %s ping\n", prog);
    printf("  %s status\n", prog);
    printf("  %s model-load weights.bin\n", prog);
    printf("  %s load lib0.out\n", prog);
    printf("  %s infer 1 1 --input input.bin --output output.bin --shape 1,3,224,224\n", prog);
    printf("  %s run --module lib0.out --input input.bin --output output.bin --shape 1,8,1\n", prog);
    printf("  %s unload 1\n", prog);
    printf("  %s model-unload 1\n", prog);
    printf("  %s trace\n", prog);
    printf("\n");
}

static int cmd_ping(void)
{
    uint32_t version, uptime;

    c7x_client_t *client = c7x_client_open();
    if (!client) {
        return 1;
    }

    int ret = c7x_client_ping(client, &version, &uptime);
    if (ret == 0) {
        printf("PING successful!\n");
        printf("  Version: %d.%d.%d\n",
               C7X_VERSION_MAJOR(version),
               C7X_VERSION_MINOR(version),
               C7X_VERSION_PATCH(version));
        printf("  Uptime:  %u.%03u seconds\n", uptime / 1000, uptime % 1000);
    } else {
        printf("PING failed: %s\n", c7x_strerror(ret));
    }

    c7x_client_close(client);
    return (ret == 0) ? 0 : 1;
}

static int cmd_status(void)
{
    c7x_status_t status;

    c7x_client_t *client = c7x_client_open();
    if (!client) {
        return 1;
    }

    int ret = c7x_client_get_status(client, &status);
    if (ret == 0) {
        printf("C7x Compute Service Status:\n");
        printf("  Version:        %d.%d.%d\n",
               C7X_VERSION_MAJOR(status.version),
               C7X_VERSION_MINOR(status.version),
               C7X_VERSION_PATCH(status.version));
        printf("  Uptime:         %u.%03u seconds\n",
               status.uptime_ms / 1000, status.uptime_ms % 1000);
        printf("  Jobs completed: %u\n", status.jobs_completed);
        printf("  Jobs failed:    %u\n", status.jobs_failed);
    } else {
        printf("Failed to get status: %s\n", c7x_strerror(ret));
    }

    c7x_client_close(client);
    return (ret == 0) ? 0 : 1;
}

/*
 * Find the remoteproc index for our DSP by matching the device tree address
 * in sysfs.  Returns the index (e.g. 0 for remoteproc0), or -1 on failure.
 */
static int find_remoteproc_index(const char *device_addr)
{
    char path[256], link[512];

    for (int i = 0; i < 16; i++) {
        snprintf(path, sizeof(path),
                 "/sys/class/remoteproc/remoteproc%d/device", i);
        ssize_t len = readlink(path, link, sizeof(link) - 1);
        if (len < 0) continue;
        link[len] = '\0';
        if (strstr(link, device_addr))
            return i;
    }
    return -1;
}

static int cmd_trace(void)
{
    /* Discover which remoteproc owns the C7x DSP */
    int rproc_idx = find_remoteproc_index(C7X_DSP_DEVICE_ADDR);
    if (rproc_idx < 0) {
        fprintf(stderr, "Could not find remoteproc for %s\n",
                C7X_DSP_DEVICE_ADDR);
        return 1;
    }

    char trace_path[256];
    snprintf(trace_path, sizeof(trace_path),
             "/sys/kernel/debug/remoteproc/remoteproc%d/trace0", rproc_idx);

    printf("Monitoring DSP trace buffer (Ctrl+C to stop)...\n");
    printf("Source: %s\n", trace_path);
    printf("----------------------------------------\n");

    /* Set up signal handler for clean exit */
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    size_t last_len = 0;

    /* Poll for new data */
    while (g_running) {
        FILE *fp = fopen(trace_path, "r");
        if (!fp) {
            perror("Failed to open trace file");
            return 1;
        }

        /* Read entire trace buffer contents */
        char buf[4096];
        size_t total = 0;
        size_t n;
        /* Skip what we already printed */
        while (total < last_len) {
            size_t want = last_len - total;
            if (want > sizeof(buf)) want = sizeof(buf);
            n = fread(buf, 1, want, fp);
            if (n == 0) break;
            total += n;
        }

        /* Print new data */
        while ((n = fread(buf, 1, sizeof(buf), fp)) > 0) {
            fwrite(buf, 1, n, stdout);
            total += n;
        }
        if (total > last_len)
            fflush(stdout);
        last_len = total;

        fclose(fp);
        usleep(500000);  /* 500ms polling interval */
    }

    printf("\n----------------------------------------\n");
    printf("Trace monitoring stopped.\n");

    return 0;
}

/*
 * =============================================================================
 * Dynamic Loading & TVM Inference Commands
 * =============================================================================
 */

static int cmd_model_load(const char *weights_file)
{
    uint32_t model_id;

    if (!weights_file) {
        fprintf(stderr, "Error: weights file required\n");
        return 1;
    }

    c7x_client_t *client = c7x_client_open();
    if (!client) return 1;

    int ret = c7x_client_model_load(client, weights_file, &model_id);
    if (ret == 0) {
        printf("Model loaded successfully: model_id=%u\n", model_id);
    } else {
        printf("Model load failed: %s\n", c7x_strerror(ret));
    }

    c7x_client_close(client);
    return (ret == 0) ? 0 : 1;
}

static int cmd_model_unload(uint32_t model_id)
{
    c7x_client_t *client = c7x_client_open();
    if (!client) return 1;

    int ret = c7x_client_model_unload(client, model_id);
    if (ret == 0) {
        printf("Model unloaded: model_id=%u\n", model_id);
    } else {
        printf("Model unload failed: %s\n", c7x_strerror(ret));
    }

    c7x_client_close(client);
    return (ret == 0) ? 0 : 1;
}

static int cmd_load(const char *elf_file)
{
    uint32_t handle;

    if (!elf_file) {
        fprintf(stderr, "Error: ELF file required\n");
        return 1;
    }

    c7x_client_t *client = c7x_client_open();
    if (!client) return 1;

    int ret = c7x_client_dyn_load(client, elf_file, &handle);
    if (ret == 0) {
        printf("Module loaded successfully: handle=%u\n", handle);
    } else {
        printf("Module load failed: %s\n", c7x_strerror(ret));
    }

    c7x_client_close(client);
    return (ret == 0) ? 0 : 1;
}

static int cmd_unload(uint32_t handle)
{
    c7x_client_t *client = c7x_client_open();
    if (!client) return 1;

    int ret = c7x_client_dyn_unload(client, handle);
    if (ret == 0) {
        printf("Module unloaded: handle=%u\n", handle);
    } else {
        printf("Module unload failed: %s\n", c7x_strerror(ret));
    }

    c7x_client_close(client);
    return (ret == 0) ? 0 : 1;
}

/**
 * Parse a comma-separated shape string like "1,3,224,224" into an int64_t array.
 */
static int parse_shape(const char *str, int64_t *shape, int max_ndim)
{
    int ndim = 0;
    const char *p = str;

    while (*p && ndim < max_ndim) {
        char *end;
        shape[ndim++] = strtoll(p, &end, 10);
        p = end;
        if (*p == ',') p++;
    }
    return ndim;
}

/**
 * Parse dtype string to (code, bits) pair.
 */
static int parse_dtype(const char *str, int32_t *code, int32_t *bits)
{
    if (!str || strcmp(str, "float32") == 0) {
        *code = 2; *bits = 32;
    } else if (strcmp(str, "float16") == 0) {
        *code = 2; *bits = 16;
    } else if (strcmp(str, "int32") == 0) {
        *code = 0; *bits = 32;
    } else if (strcmp(str, "int8") == 0) {
        *code = 0; *bits = 8;
    } else if (strcmp(str, "uint8") == 0) {
        *code = 1; *bits = 8;
    } else {
        fprintf(stderr, "Unknown dtype: %s\n", str);
        return -1;
    }
    return 0;
}

static int cmd_infer(uint32_t module_handle, uint32_t model_id,
                     const char *input_file, const char *output_file,
                     const char *shape_str, const char *dtype_str)
{
    c7x_tensor_desc_t input, output;
    int num_outputs = 0;
    uint64_t cycles = 0;

    if (!input_file) {
        fprintf(stderr, "Error: --input file required\n");
        return 1;
    }

    /* Parse dtype */
    int32_t dtype_code = 2, dtype_bits = 32;
    if (parse_dtype(dtype_str, &dtype_code, &dtype_bits) < 0) return 1;

    /* Read input file */
    UniqueFile f(fopen(input_file, "rb"));
    if (!f) {
        fprintf(stderr, "Failed to open %s: %s\n", input_file, strerror(errno));
        return 1;
    }
    fseek(f.get(), 0, SEEK_END);
    size_t file_size = ftell(f.get());
    fseek(f.get(), 0, SEEK_SET);

    auto input_data = std::make_unique<uint8_t[]>(file_size);
    if (fread(input_data.get(), 1, file_size, f.get()) != file_size) {
        fprintf(stderr, "Failed to read input file\n");
        return 1;
    }
    f = {};  /* close input file early */

    /* Build input tensor descriptor */
    memset(&input, 0, sizeof(input));
    input.data = input_data.get();
    input.data_size = file_size;
    input.dtype_code = dtype_code;
    input.dtype_bits = dtype_bits;

    if (shape_str) {
        input.ndim = parse_shape(shape_str, input.shape, C7X_TENSOR_MAX_NDIM);
    } else {
        /* Default: 1D tensor of appropriate type */
        input.ndim = 1;
        input.shape[0] = static_cast<int64_t>(file_size / (dtype_bits / 8));
    }

    memset(&output, 0, sizeof(output));

    c7x_client_t *client = c7x_client_open();
    if (!client) {
        return 1;
    }

    int ret = c7x_client_infer(client, module_handle, model_id,
                               &input, 1, &output, &num_outputs, &cycles);

    if (ret == 0) {
        printf("Inference complete: %llu cycles\n", (unsigned long long)cycles);
        printf("  Outputs: %d\n", num_outputs);
        if (num_outputs > 0) {
            printf("  Output[0]: ndim=%d, dtype=%d/%d, size=%zu bytes\n",
                   output.ndim, output.dtype_code, output.dtype_bits,
                   output.data_size);
            printf("  Shape: [");
            for (int i = 0; i < output.ndim; i++) {
                printf("%lld%s", static_cast<long long>(output.shape[i]),
                       (i < output.ndim - 1) ? "," : "");
            }
            printf("]\n");

            /* Write output file if requested */
            if (output_file && output.data && output.data_size > 0) {
                UniqueFile of(fopen(output_file, "wb"));
                if (of) {
                    fwrite(output.data, 1, output.data_size, of.get());
                    printf("  Written %zu bytes to %s\n", output.data_size, output_file);
                } else {
                    fprintf(stderr, "Failed to open output file: %s\n", output_file);
                }
            }
        }
    } else {
        printf("Inference failed: %s\n", c7x_strerror(ret));
    }

    c7x_client_close(client);
    return (ret == 0) ? 0 : 1;
}

static int cmd_run(const char *module_file,
                   const char *input_file, const char *output_file,
                   const char *shape_str, const char *dtype_str)
{
    c7x_tensor_desc_t input, output;
    int num_outputs = 0;
    uint64_t cycles = 0;
    c7x_client_t *client = nullptr;
    uint32_t handle = 0;
    bool module_loaded = false;
    int ret;
    const char *error_stage = nullptr;

    if (!module_file) {
        fprintf(stderr, "Error: --module file required\n");
        return 1;
    }
    if (!input_file) {
        fprintf(stderr, "Error: --input file required\n");
        return 1;
    }

    /* Parse dtype */
    int32_t dtype_code = 2, dtype_bits = 32;
    if (parse_dtype(dtype_str, &dtype_code, &dtype_bits) < 0) return 1;

    /* Read input file */
    UniqueFile f(fopen(input_file, "rb"));
    if (!f) {
        fprintf(stderr, "Failed to open %s: %s\n", input_file, strerror(errno));
        return 1;
    }
    fseek(f.get(), 0, SEEK_END);
    size_t file_size = ftell(f.get());
    fseek(f.get(), 0, SEEK_SET);

    auto input_data = std::make_unique<uint8_t[]>(file_size);
    if (fread(input_data.get(), 1, file_size, f.get()) != file_size) {
        fprintf(stderr, "Failed to read input file\n");
        return 1;
    }
    f = {};  /* close input file early */

    /* Build input tensor descriptor */
    memset(&input, 0, sizeof(input));
    input.data = input_data.get();
    input.data_size = file_size;
    input.dtype_code = dtype_code;
    input.dtype_bits = dtype_bits;

    if (shape_str) {
        input.ndim = parse_shape(shape_str, input.shape, C7X_TENSOR_MAX_NDIM);
    } else {
        input.ndim = 1;
        input.shape[0] = static_cast<int64_t>(file_size / (dtype_bits / 8));
    }

    memset(&output, 0, sizeof(output));

    /* Open client connection */
    client = c7x_client_open();
    if (!client) {
        printf("{\"status\":\"error\",\"stage\":\"open\","
               "\"error\":\"Failed to open client\"}\n");
        return 1;
    }

    /* LOAD */
    ret = c7x_client_dyn_load(client, module_file, &handle);
    if (ret != 0) {
        error_stage = "load";
        goto cleanup;
    }
    module_loaded = true;

    /* INFER — model_id=0 triggers embedded-weights fallback */
    ret = c7x_client_infer(client, handle, 0,
                           &input, 1, &output, &num_outputs, &cycles);
    if (ret != 0) {
        error_stage = "infer";
        goto cleanup;
    }

    /* Write output file if requested */
    if (output_file && output.data && output.data_size > 0) {
        UniqueFile of(fopen(output_file, "wb"));
        if (of) {
            fwrite(output.data, 1, output.data_size, of.get());
        } else {
            fprintf(stderr, "Failed to open output file: %s\n", output_file);
        }
    }

cleanup:
    /* UNLOAD — always attempt if module was loaded */
    if (module_loaded) {
        int unload_ret = c7x_client_dyn_unload(client, handle);
        if (unload_ret != 0 && ret == 0) {
            /* Only report unload failure if everything else succeeded */
            ret = unload_ret;
            error_stage = "unload";
        }
    }

    c7x_client_close(client);

    /* Print JSON result */
    if (ret == 0) {
        printf("{\"status\":\"ok\",\"cycles\":%llu,\"num_outputs\":%d,\"outputs\":[",
               (unsigned long long)cycles, num_outputs);
        for (int i = 0; i < num_outputs && i < 1; i++) {
            printf("{\"index\":%d,\"ndim\":%d,"
                   "\"dtype_code\":%d,\"dtype_bits\":%d,"
                   "\"data_size\":%zu,\"shape\":[",
                   i, output.ndim,
                   output.dtype_code, output.dtype_bits,
                   output.data_size);
            for (int j = 0; j < output.ndim; j++) {
                printf("%lld%s", static_cast<long long>(output.shape[j]),
                       (j < output.ndim - 1) ? "," : "");
            }
            printf("]}");
        }
        printf("]}\n");
    } else {
        printf("{\"status\":\"error\",\"stage\":\"%s\","
               "\"error\":\"%s\"}\n",
               error_stage ? error_stage : "unknown",
               c7x_strerror(ret));
    }

    return (ret == 0) ? 0 : 1;
}

int main(int argc, char *argv[])
{
    const char *command;
    const char *input_file = nullptr;
    const char *output_file = nullptr;
    const char *module_file = nullptr;
    int opt;

    const char *shape_str = nullptr;
    const char *dtype_str = nullptr;

    static struct option long_options[] = {
        {"input",  required_argument, 0, 'i'},
        {"output", required_argument, 0, 'o'},
        {"shape",  required_argument, 0, 's'},
        {"dtype",  required_argument, 0, 'd'},
        {"module", required_argument, 0, 'm'},
        {"help",   no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    command = argv[1];

    /* Parse options */
    optind = 2;  /* Start parsing after command */
    while ((opt = getopt_long(argc, argv, "i:o:s:d:m:h", long_options, nullptr)) != -1) {
        switch (opt) {
        case 'i':
            input_file = optarg;
            break;
        case 'o':
            output_file = optarg;
            break;
        case 's':
            shape_str = optarg;
            break;
        case 'd':
            dtype_str = optarg;
            break;
        case 'm':
            module_file = optarg;
            break;
        case 'h':
            print_usage(argv[0]);
            return 0;
        default:
            print_usage(argv[0]);
            return 1;
        }
    }

    /* Dispatch command */
    if (strcmp(command, "ping") == 0) {
        return cmd_ping();
    } else if (strcmp(command, "status") == 0) {
        return cmd_status();
    } else if (strcmp(command, "model-load") == 0) {
        /* c7x_compute model-load <weights.bin> */
        if (optind > 2) {
            return cmd_model_load(argv[optind - 1]);
        } else if (argc > 2 && argv[2][0] != '-') {
            return cmd_model_load(argv[2]);
        } else {
            fprintf(stderr, "Error: weights file required\n");
            return 1;
        }
    } else if (strcmp(command, "model-unload") == 0) {
        /* c7x_compute model-unload <model_id> */
        if (argc > 2) {
            return cmd_model_unload(strtoul(argv[2], nullptr, 0));
        } else {
            fprintf(stderr, "Error: model_id required\n");
            return 1;
        }
    } else if (strcmp(command, "load") == 0) {
        /* c7x_compute load <lib0.out> */
        if (optind > 2) {
            return cmd_load(argv[optind - 1]);
        } else if (argc > 2 && argv[2][0] != '-') {
            return cmd_load(argv[2]);
        } else {
            fprintf(stderr, "Error: ELF file required\n");
            return 1;
        }
    } else if (strcmp(command, "unload") == 0) {
        /* c7x_compute unload <handle> */
        if (argc > 2) {
            return cmd_unload(strtoul(argv[2], nullptr, 0));
        } else {
            fprintf(stderr, "Error: module handle required\n");
            return 1;
        }
    } else if (strcmp(command, "infer") == 0) {
        /* c7x_compute infer <handle> <model_id> --input <in> --output <out> ...
         * Options already parsed above; positional args are at argv[optind..] */
        uint32_t handle_arg = 0, model_arg = 0;
        if (argc - optind >= 2) {
            handle_arg = strtoul(argv[optind], nullptr, 0);
            model_arg = strtoul(argv[optind + 1], nullptr, 0);
        } else {
            fprintf(stderr, "Error: handle and model_id required\n");
            return 1;
        }
        return cmd_infer(handle_arg, model_arg,
                         input_file, output_file, shape_str, dtype_str);
    } else if (strcmp(command, "run") == 0) {
        return cmd_run(module_file, input_file, output_file,
                       shape_str, dtype_str);
    } else if (strcmp(command, "trace") == 0) {
        return cmd_trace();
    } else if (strcmp(command, "help") == 0 || strcmp(command, "-h") == 0 ||
               strcmp(command, "--help") == 0) {
        print_usage(argv[0]);
        return 0;
    } else {
        fprintf(stderr, "Unknown command: %s\n", command);
        print_usage(argv[0]);
        return 1;
    }
}
