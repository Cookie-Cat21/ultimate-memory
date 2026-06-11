# Claude Code SessionEnd hook — ingests the session transcript into Ultimate Memory.
# Claude Code sends a JSON payload on stdin: {session_id, transcript_path, project_path, ...}
# This script pipes that into ingest_session.py which calls the MemoryRouter.
$ErrorActionPreference = "SilentlyContinue"

$repoRoot = "C:\Users\Ovindu\Documents\Pet Projects\ultimate-memory"
$scriptPath = Join-Path $repoRoot "scripts\ingest_session.py"

# Read stdin from the hook payload
$stdinData = [Console]::In.ReadToEnd()

if (-not $stdinData.Trim()) {
    exit 0
}

# Pipe through uv run python
$stdinData | & uv --directory $repoRoot run python $scriptPath 2>&1 | Out-Null
