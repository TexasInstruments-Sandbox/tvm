/**
 * CCS Debug Server Scripting (DSS) script to load and run programs
 * on the J722S C75x DSP.
 *
 * Usage:
 *   <CCS_INSTALL>/ccs/ccs_base/scripting/bin/dss.sh load_and_run_c75x.js [program.out] [options]
 *
 * Options:
 *   --timeout <ms>     Execution timeout in milliseconds (default: 60000)
 *   --output <file>    File to capture console output (default: stdout)
 *   --no-wait          Don't wait for program to halt (run in background)
 *   --reset            Reset target before loading
 *   --help             Show this help message
 *
 * Examples:
 *   dss.sh load_and_run_c75x.js ../hello_world_standalone/build/c75_hello_world.out
 *   dss.sh load_and_run_c75x.js program.out --timeout 120000
 *   dss.sh load_and_run_c75x.js program.out --output test_results.log
 */

// Get script directory for relative paths
var scriptDir = new java.io.File(java.lang.System.getProperty("user.dir")).getCanonicalPath();

// Print usage help
function printUsage() {
    print("");
    print("Usage: dss.sh load_and_run_c75x.js [program.out] [options]");
    print("");
    print("Options:");
    print("  --timeout <ms>     Execution timeout in milliseconds (default: 60000)");
    print("  --output <file>    File to capture console output");
    print("  --no-wait          Don't wait for program to halt");
    print("  --reset            Reset target before loading");
    print("  --force            Force connect (recover from HPI/page fault state)");
    print("  --ccxml <file>     CCXML configuration file");
    print("  --ccs-root <path>  CCS installation root");
    print("  --core <pattern>   Core name pattern (default: C75SS0_0)");
    print("  --help             Show this help message");
    print("");
}

// Default configuration
var config = {
    ccsRoot: java.lang.System.getenv("CCS_ROOT") || "/home/a0323430/ti/ccs2040/ccs",
    ccxmlFile: scriptDir + "/J722S_560v2.ccxml",
    corePattern: ".*C75X_0",
    programFile: null,
    timeout: 60000,
    outputFile: null,
    waitForHalt: true,
    resetTarget: false,
    forceConnect: false
};

// Parse command line arguments
for (var i = 0; i < arguments.length; i++) {
    var arg = String(arguments[i]);

    if (arg == "--help" || arg == "-h") {
        printUsage();
        quit(0);
    } else if (arg == "--timeout") {
        i++;
        config.timeout = parseInt(String(arguments[i]));
    } else if (arg == "--output") {
        i++;
        config.outputFile = String(arguments[i]);
    } else if (arg == "--no-wait") {
        config.waitForHalt = false;
    } else if (arg == "--reset") {
        config.resetTarget = true;
    } else if (arg == "--force") {
        config.forceConnect = true;
    } else if (arg == "--ccxml") {
        i++;
        config.ccxmlFile = String(arguments[i]);
    } else if (arg == "--ccs-root") {
        i++;
        config.ccsRoot = String(arguments[i]);
    } else if (arg == "--core") {
        i++;
        config.corePattern = String(arguments[i]);
    } else if (arg.indexOf("-") != 0 && config.programFile == null) {
        config.programFile = arg;
    }
}

function log(msg) {
    var sdf = new java.text.SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSS");
    var timestamp = sdf.format(new java.util.Date());
    print("[" + timestamp + "] " + msg);
}

function main() {
    // Validate program file
    if (config.programFile == null) {
        // Default to hello_world output if not specified
        config.programFile = scriptDir + "/../hello_world_standalone/build/c75_hello_world.out";
    }

    // Resolve relative paths
    var programPath = new java.io.File(config.programFile);
    if (!programPath.isAbsolute()) {
        programPath = new java.io.File(scriptDir, config.programFile);
    }
    config.programFile = programPath.getCanonicalPath();

    var ccxmlPath = new java.io.File(config.ccxmlFile);
    if (!ccxmlPath.isAbsolute()) {
        ccxmlPath = new java.io.File(scriptDir, config.ccxmlFile);
    }
    config.ccxmlFile = ccxmlPath.getCanonicalPath();

    // Verify files exist
    if (!new java.io.File(config.programFile).exists()) {
        log("ERROR: Program file not found: " + config.programFile);
        quit(1);
    }

    if (!new java.io.File(config.ccxmlFile).exists()) {
        log("ERROR: CCXML file not found: " + config.ccxmlFile);
        quit(1);
    }

    log("=== J722S C75x DSP Loader ===");
    log("Program: " + config.programFile);
    log("CCXML:   " + config.ccxmlFile);
    log("Core:    " + config.corePattern);
    log("Timeout: " + config.timeout + " ms");
    log("");

    // Create scripting environment
    var script = new Packages.com.ti.ccstudio.scripting.environment.ScriptingEnvironment.instance();
    script.traceSetConsoleLevel(Packages.com.ti.ccstudio.scripting.environment.TraceLevel.INFO);

    var debugServer = null;
    var debugSession = null;
    var captureFile = null;

    try {
        // Set timeout
        script.setScriptTimeout(config.timeout);

        // Create debug server
        log("Creating debug server...");
        debugServer = script.getServer("DebugServer.1");

        // Configure target
        log("Configuring target with CCXML...");
        debugServer.setConfig(config.ccxmlFile);

        // Open session to C75x DSP
        log("Opening session to " + config.corePattern + "...");
        debugSession = debugServer.openSession(config.corePattern);

        // Connect to target
        log("Connecting to target...");
        if (config.forceConnect) {
            log("Using FORCE connect mode...");
            // Set connection options to force recovery from HPI state
            try {
                // Try setting force-related options before connect
                debugSession.options.setBoolean("AutoRunToLabelOnRestart", false);
                debugSession.options.setBoolean("ResetOnRestart", false);
            } catch (optErr) {
                log("Could not set options: " + optErr.message);
            }

            try {
                // First try normal connect
                debugSession.target.connect();
            } catch (connectError) {
                log("Normal connect failed: " + connectError.message);
                log("");
                log("=== FORCE RECOVERY REQUIRED ===");
                log("The target is stuck in a High Priority Interrupt (Double Page Fault).");
                log("Automatic recovery via DSS scripting is limited.");
                log("");
                log("To recover, use one of these methods:");
                log("  1. Power cycle the EVM board");
                log("  2. In CCS GUI: Target → Connect Target, then click 'Force' when prompted");
                log("  3. In CCS GUI: Run → Reset → CPU Reset");
                log("");
                throw connectError;
            }
        } else {
            debugSession.target.connect();
        }

        // Optional reset
        if (config.resetTarget) {
            log("Resetting target...");
            debugSession.target.reset();
        }

        // Load program
        log("Loading program...");
        debugSession.memory.loadProgram(config.programFile);

        // Note: CIO capture (beginCIOCapture) is not available for C7x sessions
        // File I/O must be handled through memory-mapped approach or avoided
        if (config.outputFile != null) {
            log("Output capture requested but CIO not available for C7x");
            log("Output file will not be captured: " + config.outputFile);
        }

        // Run the program
        log("Running program...");
        log("");
        log("=== Program Output ===");

        if (config.waitForHalt) {
            debugSession.target.run();
            log("=== Program Completed ===");
        } else {
            debugSession.target.runAsynch();
            log("Program started (not waiting for completion)");
        }

    } catch (e) {
        log("ERROR: " + e.message);
        if (e.javaException) {
            log("Java Exception: " + e.javaException.getMessage());
        }
        quit(1);
    } finally {
        // Cleanup
        log("Cleaning up...");

        // Note: CIO capture not available for C7x, nothing to clean up

        if (debugSession != null) {
            try {
                debugSession.target.disconnect();
            } catch (e) {}
            try {
                debugSession.terminate();
            } catch (e) {}
        }

        if (debugServer != null) {
            try {
                debugServer.stop();
            } catch (e) {}
        }
    }

    log("Done.");
}

// Run main
main();
