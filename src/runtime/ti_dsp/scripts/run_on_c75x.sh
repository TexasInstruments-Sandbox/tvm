#!/bin/bash
#
# Load and run a program on J722S C75x DSP
#
# Usage:
#   ./run_on_c75x.sh [program.out] [options]
#
# Examples:
#   ./run_on_c75x.sh                                    # Run default hello world
#   ./run_on_c75x.sh ../hello_world_standalone/build/c75_hello_world.out
#   ./run_on_c75x.sh --timeout 120000                   # Extended timeout
#   ./run_on_c75x.sh --output results.log               # Capture to file
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default CCS installation path
CCS_ROOT="${CCS_ROOT:-/home/a0323430/ti/ccs2040/ccs}"

# DSS executable
DSS_CMD="${CCS_ROOT}/ccs_base/scripting/bin/dss.sh"

# Default program to run
DEFAULT_PROGRAM="${PROJECT_ROOT}/hello_world_standalone/build/c75_hello_world.out"

# Default CCXML configuration
DEFAULT_CCXML="${HOME}/ti/CCSTargetConfigurations/J722S_560v2.ccxml"

# Default core pattern for C75x_0 (regex pattern)
DEFAULT_CORE=".*C75X_0"

# Print help
print_help() {
    cat << EOF
J722S C75x DSP Program Loader

Usage: $(basename "$0") [program.out] [options]

Arguments:
  program.out          Program to load and run (ELF format)
                       Default: hello_world_standalone/build/c75_hello_world.out

Options:
  --timeout <ms>       Execution timeout in milliseconds (default: 60000)
  --output <file>      File to capture console output
  --no-wait            Don't wait for program completion
  --reset              Reset target before loading
  --force              Force connect (recover from HPI/page fault state)
  --ccxml <file>       Custom CCXML configuration file
                       Default: $DEFAULT_CCXML
  --core <pattern>     Core name pattern (default: $DEFAULT_CORE)
  --ccs-root <path>    CCS installation root (default: $CCS_ROOT)
  -h, --help           Show this help message

Environment Variables:
  CCS_ROOT             CCS installation directory

Examples:
  # Run the default hello world program
  $(basename "$0")

  # Run with extended timeout (2 minutes)
  $(basename "$0") --timeout 120000

  # Run a specific program and save output
  $(basename "$0") my_program.out --output results.log

  # Run on C75x_1 instead of C75x_0
  $(basename "$0") --core ".*C75X_1"

  # Run without waiting for completion
  $(basename "$0") --no-wait

Prerequisites:
  - J722S EVM connected via JTAG debug probe
  - Code Composer Studio installed at CCS_ROOT
  - CCXML target configuration file for J722S

EOF
}

# Check prerequisites
check_prerequisites() {
    # Check CCS installation
    if [[ ! -d "$CCS_ROOT" ]]; then
        echo "ERROR: CCS not found at: $CCS_ROOT"
        echo "Set CCS_ROOT environment variable to your CCS installation."
        exit 1
    fi

    # Check DSS executable
    if [[ ! -x "$DSS_CMD" ]]; then
        echo "ERROR: DSS executable not found: $DSS_CMD"
        exit 1
    fi

    # Check JavaScript file
    if [[ ! -f "${SCRIPT_DIR}/load_and_run_c75x.js" ]]; then
        echo "ERROR: JavaScript file not found: ${SCRIPT_DIR}/load_and_run_c75x.js"
        exit 1
    fi
}

# Parse arguments and build DSS command
main() {
    local program=""
    local dss_args=()
    local ccxml="$DEFAULT_CCXML"

    # Parse command line
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -h|--help)
                print_help
                exit 0
                ;;
            --timeout|--output|--core)
                dss_args+=("$1" "$2")
                shift 2
                ;;
            --ccxml)
                ccxml="$2"
                dss_args+=("$1" "$2")
                shift 2
                ;;
            --ccs-root)
                CCS_ROOT="$2"
                DSS_CMD="${CCS_ROOT}/ccs_base/scripting/bin/dss.sh"
                dss_args+=("$1" "$2")
                shift 2
                ;;
            --no-wait|--reset|--force)
                dss_args+=("$1")
                shift
                ;;
            -*)
                echo "Unknown option: $1"
                print_help
                exit 1
                ;;
            *)
                if [[ -z "$program" ]]; then
                    program="$1"
                else
                    echo "Unexpected argument: $1"
                    print_help
                    exit 1
                fi
                shift
                ;;
        esac
    done

    check_prerequisites

    # Use default program if not specified
    if [[ -z "$program" ]]; then
        program="$DEFAULT_PROGRAM"
    fi

    # Make program path absolute if relative
    if [[ "$program" != /* ]]; then
        program="$(pwd)/$program"
    fi

    # Check program exists
    if [[ ! -f "$program" ]]; then
        echo "ERROR: Program file not found: $program"
        echo ""
        echo "Build the program first with:"
        echo "  cd ${PROJECT_ROOT}/hello_world_standalone && ./build.sh"
        exit 1
    fi

    # Check CCXML exists
    if [[ ! -f "$ccxml" ]]; then
        echo "ERROR: CCXML file not found: $ccxml"
        echo ""
        echo "Create a target configuration in CCS for J722S EVM."
        exit 1
    fi

    echo "=============================================="
    echo "  J722S C75x DSP Program Loader"
    echo "=============================================="
    echo "CCS Root:  $CCS_ROOT"
    echo "Program:   $program"
    echo "CCXML:     $ccxml"
    echo "=============================================="
    echo ""

    # Run DSS script (always pass ccxml to override JS default)
    exec "$DSS_CMD" "${SCRIPT_DIR}/load_and_run_c75x.js" "$program" --ccxml "$ccxml" "${dss_args[@]}"
}

main "$@"
