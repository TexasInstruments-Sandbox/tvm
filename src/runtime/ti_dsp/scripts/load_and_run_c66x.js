/**
 * CCS Debug Server Scripting (DSS) script to load and run programs
 * on the AWRL6844 C66x DSP.
 *
 * Usage:
 *   <CCS_INSTALL>/ccs/ccs_base/scripting/bin/dss.sh load_and_run_c66x.js [program.out] [options]
 *
 * Options:
 *   --timeout <ms>     Execution timeout in milliseconds (default: 60000)
 *   --output <file>    File to capture console output (default: stdout)
 *   --no-wait          Don't wait for program to halt (run in background)
 *   --reset            Reset target before loading
 *   --help             Show this help message
 *
 * Examples:
 *   dss.sh load_and_run_c66x.js ../build-awrl6844/run_layer_tests.out
 *   dss.sh load_and_run_c66x.js ../build-awrl6844/run_layer_tests.out --timeout 120000
 *   dss.sh load_and_run_c66x.js ../build-awrl6844/run_layer_tests.out --output test_results.log
 */

// Get script directory for relative paths
// DSS runs scripts from their location, so we can find the script directory
// by walking up from user.dir looking for the scripts directory with this file
var projectRoot = null;
var searchDir = new java.io.File(java.lang.System.getProperty("user.dir"));

// First, try to find the project root by looking for layer_tests directory
while (searchDir != null) {
    var layerTestsDir = new java.io.File(searchDir, "layer_tests");
    var scriptsDir = new java.io.File(searchDir, "scripts");
    if (layerTestsDir.exists() && layerTestsDir.isDirectory() &&
        scriptsDir.exists() && scriptsDir.isDirectory()) {
        projectRoot = searchDir.getCanonicalPath();
        break;
    }
    searchDir = searchDir.getParentFile();
}

// Fallback: check environment variable or common paths
if (projectRoot == null) {
    var envRoot = java.lang.System.getenv("TIDL_PROJECT_ROOT");
    if (envRoot != null && new java.io.File(envRoot).exists()) {
        projectRoot = new java.io.File(envRoot).getCanonicalPath();
    }
}

if (projectRoot == null) {
    // Last resort: assume we're in the project root
    projectRoot = java.lang.System.getProperty("user.dir");
}

var scriptDir = projectRoot + "/scripts";

// Detect SDK path (Mac vs Linux)
var homeDir = java.lang.System.getProperty("user.home");
var sdkPath = homeDir + "/ti/MMWAVE_L_SDK_06_01_00_05";
if (!new java.io.File(sdkPath).exists()) {
    // Try alternate location
    sdkPath = "/home/" + java.lang.System.getProperty("user.name") + "/ti/MMWAVE_L_SDK_06_01_00_05";
}

// Print usage help
function printUsage() {
    print("");
    print("Usage: dss.sh load_and_run_c66x.js [program.out] [options]");
    print("");
    print("Options:");
    print("  --timeout <ms>     Execution timeout in milliseconds (default: 60000)");
    print("  --output <file>    File to capture console output");
    print("  --no-wait          Don't wait for program to halt");
    print("  --reset            Reset target before loading");
    print("  --ccxml <file>     CCXML configuration file");
    print("  --ccs-root <path>  CCS installation root");
    print("  --help             Show this help message");
    print("");
}

// Default configuration
var config = {
    ccsRoot: java.lang.System.getenv("CCS_ROOT") || "/home/a0323430/ti/ccs2040/ccs",
    //ccxmlFile: "/home/a0323430/ti/CCSTargetConfigurations/xwrl68xx_xds110.ccxml",
    ccxmlFile: scriptDir + "/../layer_tests/awrl6844/targetConfigs/AWRL68xx.ccxml",
    corePattern: "Texas Instruments XDS110 USB Debug Probe/C66xx_DSP",
    programFile: null,
    timeout: 60000,
    outputFile: null,
    waitForHalt: true,
    resetTarget: false
};

// Parse command line arguments at top level (global 'arguments' from DSS)
// Note: DSS does NOT include script name in arguments - arguments[0] is the first user arg
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
    } else if (arg == "--ccxml") {
        i++;
        config.ccxmlFile = String(arguments[i]);
    } else if (arg == "--ccs-root") {
        i++;
        config.ccsRoot = String(arguments[i]);
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
        // Default to run_layer_tests.out if not specified
        config.programFile = scriptDir + "/../layer_tests/build-awrl6844/run_layer_tests.out";
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

    log("=== AWRL6844 C66x DSP Loader ===");
    log("Program: " + config.programFile);
    log("CCXML:   " + config.ccxmlFile);
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

        // Open session to C66x DSP
        log("Opening session to " + config.corePattern + "...");
        debugSession = debugServer.openSession(config.corePattern);
        // Load GEL file (optional - provides additional target setup)
        var gelFile = sdkPath + "/tools/gelfile/xwrl68xx.gel";
        if (new java.io.File(gelFile).exists()) {
            try {
                debugSession.expression.evaluate('GEL_LoadGel("' + gelFile + '")');
            } catch (err) {
                log("Warning: Could not load GEL file: " + err);
            }
        } else {
            log("Note: GEL file not found at " + gelFile + " (optional)");
        }
        //debugSession.expression.evaluate("dsp_wakeup_unhalt()");

        // Connect to target
        log("Connecting to target...");
        debugSession.target.connect();


        // Optional reset
        if (config.resetTarget) {
            log("Resetting target...");
            debugSession.target.reset();
        }

        // Load program
        log("Loading program...");
        debugSession.memory.loadProgram(config.programFile);

        // Start capturing console output
        if (config.outputFile != null) {
            log("Capturing output to: " + config.outputFile);
            captureFile = config.outputFile;
            debugSession.beginCIOCapture(captureFile);
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

        if (captureFile != null) {
            try {
                debugSession.endCIOCapture();
            } catch (e) {}
        }

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
