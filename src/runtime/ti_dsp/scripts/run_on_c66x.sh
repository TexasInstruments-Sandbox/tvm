#!/bin/bash
#
# Load and run a program on AWRL6844 C66x DSP
#
# Usage:
#   ./run_on_c66x.sh [program.out] [options]
#
# Examples:
#   ./run_on_c66x.sh                                    # Run default test executable
#   ./run_on_c66x.sh ../build-awrl6844/my_program.out   # Run specific program
#   ./run_on_c66x.sh --timeout 120000                   # Extended timeout
#   ./run_on_c66x.sh --output results.log               # Capture to file
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LAYER_TESTS_DIR="${PROJECT_ROOT}/layer_tests"

# Default CCS installation path (detect Mac vs Linux)
if [[ "$(uname)" == "Darwin" ]]; then
    CCS_ROOT="${CCS_ROOT:-$HOME/ti/ccs2050/ccs}"
else
    CCS_ROOT="${CCS_ROOT:-$HOME/ti/ccs2040/ccs}"
fi

# DSS executable
DSS_CMD="${CCS_ROOT}/ccs_base/scripting/bin/dss.sh"

# Default program to run
DEFAULT_PROGRAM="${LAYER_TESTS_DIR}/build-awrl6844/run_layer_tests.out"

# Default CCXML configuration (in scripts directory)
DEFAULT_CCXML="${SCRIPT_DIR}/AWRL68xx.ccxml"

# Print help
print_help() {
    cat << EOF
AWRL6844 C66x DSP Program Loader

Usage: $(basename "$0") [program.out] [options]

Arguments:
  program.out          Program to load and run (ELF format)
                       Default: build-awrl6844/run_layer_tests.out

Options:
  --timeout <ms>       Execution timeout in milliseconds (default: 60000)
  --output <file>      File to capture console output
  --no-wait            Don't wait for program completion
  --reset              Reset target before loading
  --ccxml <file>       Custom CCXML configuration file
  --ccs-root <path>    CCS installation root (default: $CCS_ROOT)
  -h, --help           Show this help message

Environment Variables:
  CCS_ROOT             CCS installation directory

Examples:
  # Run the default layer tests
  $(basename "$0")

  # Run with extended timeout (2 minutes)
  $(basename "$0") --timeout 120000

  # Run a specific program and save output
  $(basename "$0") my_program.out --output results.log

  # Run without waiting for completion
  $(basename "$0") --no-wait

Prerequisites:
  - AWRL6844 hardware connected via XDS110 USB debug probe
  - Code Composer Studio installed at CCS_ROOT

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
    if [[ ! -f "${SCRIPT_DIR}/load_and_run_c66x.js" ]]; then
        echo "ERROR: JavaScript file not found: ${SCRIPT_DIR}/load_and_run_c66x.js"
        exit 1
    fi
}

# Parse arguments and build DSS command
main() {
    local program=""
    local dss_args=()
    local ccxml_specified=false

    # Parse command line
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -h|--help)
                print_help
                exit 0
                ;;
            --timeout|--output|--ccs-root)
                dss_args+=("$1" "$2")
                if [[ "$1" == "--ccs-root" ]]; then
                    CCS_ROOT="$2"
                    DSS_CMD="${CCS_ROOT}/ccs_base/scripting/bin/dss.sh"
                fi
                shift 2
                ;;
            --ccxml)
                dss_args+=("$1" "$2")
                ccxml_specified=true
                shift 2
                ;;
            --no-wait|--reset)
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

    # Add default CCXML if not specified
    if [[ "$ccxml_specified" == "false" ]]; then
        dss_args+=("--ccxml" "$DEFAULT_CCXML")
    fi

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
        echo "  cd ${LAYER_TESTS_DIR}/build-awrl6844 && cmake --build ."
        exit 1
    fi

    echo "=============================================="
    echo "  AWRL6844 C66x DSP Program Loader"
    echo "=============================================="
    echo "CCS Root:  $CCS_ROOT"
    echo "Program:   $program"
    echo "=============================================="
    echo ""

    # Run DSS script
    exec "$DSS_CMD" "${SCRIPT_DIR}/load_and_run_c66x.js" "$program" "${dss_args[@]}"
}

main "$@"
