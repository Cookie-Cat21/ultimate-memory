$ErrorActionPreference = "SilentlyContinue"

$repo = "C:\Users\Ovindu\Documents\Pet Projects\ultimate-memory"
$hookInput = [Console]::In.ReadToEnd()

if ([string]::IsNullOrWhiteSpace($hookInput)) {
    exit 0
}

try {
    $event = $hookInput | ConvertFrom-Json
} catch {
    exit 0
}

$sessionId = if ($event.session_id) { [string]$event.session_id } else { "claude-session" }
$transcriptPath = if ($event.transcript_path) { [string]$event.transcript_path } else { "" }
$cwd = if ($event.cwd) { [string]$event.cwd } else { "" }

if ([string]::IsNullOrWhiteSpace($transcriptPath) -or -not (Test-Path -LiteralPath $transcriptPath)) {
    exit 0
}

Push-Location $repo
try {
    if ([string]::IsNullOrWhiteSpace($cwd)) {
        uv run ultimate-memory ingest-log claude $sessionId $transcriptPath --tags claude --tags hook *> $null
    } else {
        uv run ultimate-memory ingest-log claude $sessionId $transcriptPath --project-path $cwd --tags claude --tags hook *> $null
    }
} finally {
    Pop-Location
}

