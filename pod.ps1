param(
    [Parameter(Mandatory=$true)][ValidateSet('upload','run','download')][string]$Action,
    [Parameter(Mandatory=$true)][string]$PodHost,
    [Parameter(Mandatory=$true)][ValidateRange(1,65535)][int]$Port,
    [string]$KeyPath = "$HOME\.ssh\id_ed25519"
)
$ErrorActionPreference = 'Stop'
if ($PodHost -notmatch '^[a-zA-Z0-9.-]+$') { throw 'Use an IP address or hostname.' }
$destination = "root@$PodHost"
$remotePath = '/workspace/ML_training'
switch ($Action) {
    'upload' {
        & ssh -p $Port -i $KeyPath $destination "mkdir -p $remotePath"
        if ($LASTEXITCODE -ne 0) { throw 'Remote directory creation failed.' }
        $files = @('train.py','README.md','pod.ps1') | ForEach-Object { Join-Path $PSScriptRoot $_ }
        & scp -P $Port -i $KeyPath @files "${destination}:${remotePath}/"
    }
    'run' {
        & ssh -p $Port -i $KeyPath $destination "cd $remotePath && python -u train.py"
    }
    'download' {
        $results = Join-Path $PSScriptRoot 'pod-results'
        New-Item -ItemType Directory -Force -Path $results | Out-Null
        & scp -P $Port -i $KeyPath -r "${destination}:${remotePath}/outputs" $results
    }
}
if ($LASTEXITCODE -ne 0) { throw "$Action failed: exit code $LASTEXITCODE" }
