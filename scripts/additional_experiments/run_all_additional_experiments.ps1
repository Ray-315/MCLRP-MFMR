param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$ResultsRoot = Join-Path $ProjectRoot "results\additional_experiments"
$ShardRoot = Join-Path $ResultsRoot "_shards"
$LogRoot = Join-Path $ResultsRoot "logs"
New-Item -ItemType Directory -Force $ShardRoot, $LogRoot | Out-Null

function Start-ExperimentProcess {
    param([string]$Name, [string[]]$Arguments)
    $stdout = Join-Path $LogRoot "$Name.stdout.log"
    $stderr = Join-Path $LogRoot "$Name.stderr.log"
    Start-Process -FilePath $Python -WorkingDirectory $ProjectRoot -ArgumentList $Arguments `
        -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru
}

function Wait-ExperimentProcesses {
    param([System.Diagnostics.Process[]]$Processes, [string[]]$Names)
    for ($index = 0; $index -lt $Processes.Count; $index++) {
        $Processes[$index].WaitForExit()
        if ($Processes[$index].ExitCode -ne 0) {
            throw "$($Names[$index]) failed with exit code $($Processes[$index].ExitCode). See results/additional_experiments/logs."
        }
    }
}

$datasets = @("CCLE", "ERKAUC30", "ERKIC50", "PI3KAUC", "PI3KIC50")
$adaptiveProcesses = @()
$adaptiveNames = @()
foreach ($dataset in $datasets) {
    $name = "adaptive_$dataset"
    $output = Join-Path $ShardRoot "adaptive\$dataset"
    $arguments = @(
        "scripts/additional_experiments/run_adaptive_fusion.py",
        "--datasets", $dataset,
        "--seeds", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
        "--outer-folds", "10", "--inner-folds", "5",
        "--output-dir", $output,
        "--resume"
    )
    $adaptiveProcesses += Start-ExperimentProcess $name $arguments
    $adaptiveNames += $name
}

$prismBundle = Join-Path $ProjectRoot "data\standardized\PRISM19Q4_independent"
$prismSplit = Join-Path $ProjectRoot "splits\additional_experiments\PRISM19Q4_locked.npz"
if ((Test-Path $prismBundle) -and (Test-Path $prismSplit)) {
    $name = "adaptive_PRISM19Q4_independent"
    $output = Join-Path $ShardRoot "adaptive\PRISM19Q4_independent"
    $arguments = @(
        "scripts/additional_experiments/run_adaptive_fusion.py",
        "--datasets", "PRISM19Q4_independent",
        "--seeds", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
        "--outer-folds", "1", "--inner-folds", "5",
        "--prism-bundle-dir", $prismBundle,
        "--prism-split", $prismSplit,
        "--output-dir", $output,
        "--resume"
    )
    $adaptiveProcesses += Start-ExperimentProcess $name $arguments
    $adaptiveNames += $name
}
Wait-ExperimentProcesses $adaptiveProcesses $adaptiveNames

& $Python "scripts/additional_experiments/aggregate_shards.py" adaptive_fusion `
    --shard-dir (Join-Path $ShardRoot "adaptive") `
    --output-dir (Join-Path $ResultsRoot "adaptive_fusion")
if ($LASTEXITCODE -ne 0) { throw "Adaptive shard aggregation failed." }

$complementArguments = @(
    "scripts/additional_experiments/analyze_branch_complementarity.py",
    "--prediction-dir", (Join-Path $ResultsRoot "adaptive_fusion\predictions"),
    "--output-dir", (Join-Path $ResultsRoot "branch_complementarity"),
    "--prism-bundle-dir", $prismBundle
)
& $Python @complementArguments
if ($LASTEXITCODE -ne 0) { throw "Branch complementarity analysis failed." }

$maskProcesses = @()
$maskNames = @()
foreach ($dataset in $datasets) {
    $name = "mask_$dataset"
    $output = Join-Path $ShardRoot "mask\$dataset"
    $arguments = @(
        "scripts/additional_experiments/run_mask_sensitivity.py",
        "--datasets", $dataset,
        "--ratios", "0.10", "0.15", "0.20", "0.25", "0.30",
        "--seeds", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
        "--methods", "original_mclrp", "mfmr_base",
        "--output-dir", $output,
        "--resume"
    )
    $maskProcesses += Start-ExperimentProcess $name $arguments
    $maskNames += $name
}
Wait-ExperimentProcesses $maskProcesses $maskNames

& $Python "scripts/additional_experiments/aggregate_shards.py" mask_sensitivity `
    --shard-dir (Join-Path $ShardRoot "mask") `
    --output-dir (Join-Path $ResultsRoot "mask_sensitivity")
if ($LASTEXITCODE -ne 0) { throw "Mask-sensitivity shard aggregation failed." }

& $Python "scripts/additional_experiments/summarize_additional_experiments.py" --results-dir $ResultsRoot
if ($LASTEXITCODE -ne 0) { throw "Final additional-experiment summary failed." }

Write-Output "ALL_ADDITIONAL_EXPERIMENTS_COMPLETE"
