// TVM C7x Contributor Docs Build Pipeline
//
// Builds the docs-c7x/ MkDocs Material site with `mkdocs build --strict`
// (fails the build on any broken link, missing nav target, or other
// mkdocs warning -- see AGENTS.md's Documentation section) and publishes
// the generated HTML as a Jenkins artifact + HTML Publisher report.
//
// This pipeline does NOT push to gh-pages. Deploying the site is a
// separate, manual step (mkdocs gh-deploy --remote-name gh-origin
// --remote-branch gh-pages, per AGENTS.md) -- intentionally not
// automated here since it pushes to a shared branch.
//
// Jenkins setup:
//   1. Install plugins: Pipeline, HTML Publisher, Timestamper,
//      Build Timeout, Pipeline Stage View
//      (HTML Publisher is already required by tests/ti-dsp-runtime's
//      cycles report, so it's likely already installed on this node.)
//   2. Create the job as: Pipeline job, SCM, script path
//      "docs-c7x/Jenkinsfile". Set the SCM's Branch Specifier to a FIXED
//      branch (e.g. "*/relax-c7x-mma") -- NOT "*/${BRANCH}". Jenkins must
//      fetch a real branch to read this Jenkinsfile before it has parsed
//      the parameters{} block below, so a literal "${BRANCH}" here fails
//      with "Couldn't find remote ref refs/heads/${BRANCH}". The BRANCH
//      parameter is consumed by this pipeline's own Checkout stage
//      instead (see below), so it can still build docs from a different
//      branch on demand without touching the job config.
//   3. Label the build node "relax-c7x" (or any x86 node with `uv`
//      installed -- see https://docs.astral.sh/uv/getting-started/installation/)
//   4. Configure git HTTP proxy on the Jenkins node (as the jenkins user),
//      same as the other C7x pipelines:
//        git config --global http.proxy http://webproxy.ext.ti.com:80
//        git config --global https.proxy http://webproxy.ext.ti.com:80
//   5. This job builds no TVM/DSP artifacts and touches no hardware --
//      it's safe to trigger frequently (SCM polling below) without
//      contending with the "am67a-dsp-board" lockable resource used by
//      tests/ti-dsp-runtime.
//   6. Jenkins' default Content-Security-Policy locks down JS/CSS for
//      anything served via archived artifacts or HTML Publisher, and
//      mkdocs-material needs its own CSS/JS (search, nav, theming) to
//      render correctly. Without loosening it, the published report will
//      load as broken/unstyled HTML. One-time, INSTANCE-WIDE fix (Manage
//      Jenkins -> Script Console), not something this Jenkinsfile can set
//      itself -- this relaxes CSP for every job's archived/published HTML
//      on this Jenkins instance, not just this one:
//        System.setProperty("hudson.model.DirectoryBrowserSupport.CSP",
//          "default-src 'self'; style-src 'self' 'unsafe-inline'; " +
//          "script-src 'self' 'unsafe-inline'; img-src 'self' data:; " +
//          "font-src 'self' data:;")
//      To persist across Jenkins restarts, set it as a JVM argument
//      instead (e.g. in /etc/sysconfig/jenkins or the service's JAVA_OPTS):
//        -Dhudson.model.DirectoryBrowserSupport.CSP="default-src 'self'; ..."

pipeline {
    agent { label 'relax-c7x' }

    triggers {
        // No inbound GitHub webhook from this corporate network; poll
        // instead. Docs builds are cheap, so a short interval is fine.
        pollSCM('H/15 * * * *')
    }

    options {
        timestamps()
        timeout(time: 15, unit: 'MINUTES')
        buildDiscarder(logRotator(numToKeepStr: '20'))
        disableConcurrentBuilds()
    }

    parameters {
        string(
            name: 'BRANCH',
            defaultValue: 'relax-c7x-mma',
            description: 'Branch to build docs from'
        )
    }

    environment {
        NO_MKDOCS_2_WARNING = '1'
        http_proxy          = 'http://webproxy.ext.ti.com:80'
        https_proxy         = 'http://webproxy.ext.ti.com:80'
        no_proxy            = 'ti.com,.ti.com,localhost,127.0.0.1'
    }

    stages {
        stage('Checkout') {
            steps {
                // Re-checkout params.BRANCH explicitly, reusing the remote
                // URL/credentials already configured on this job (via the
                // implicit `scm` variable) rather than the fixed branch
                // Jenkins used to load this Jenkinsfile. This is what
                // actually makes the BRANCH parameter do something -- see
                // "Jenkins setup" step 2 above for why the job's own SCM
                // branch specifier can't be "${BRANCH}" itself.
                checkout([
                    $class: 'GitSCM',
                    branches: [[name: params.BRANCH]],
                    userRemoteConfigs: scm.userRemoteConfigs
                ])
            }
        }

        stage('Build Docs') {
            steps {
                sh '''
                    rm -rf site
                    uvx --with mkdocs-material \
                        --with-requirements docs-c7x/requirements.txt \
                        mkdocs build --strict
                '''
            }
        }
    }

    post {
        always {
            archiveArtifacts artifacts: 'site/**', allowEmptyArchive: true
            // If this renders as unstyled/broken HTML, see "Jenkins setup"
            // step 6 above -- it's almost certainly the instance's default
            // Content-Security-Policy blocking mkdocs-material's CSS/JS.
            publishHTML([
                allowMissing: true,
                alwaysLinkToLastBuild: true,
                keepAll: true,
                reportDir: 'site',
                reportFiles: 'index.html',
                reportName: 'C7x Contributor Docs'
            ])
        }
    }
}
