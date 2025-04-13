Invoke-Expression .\.venv\Scripts\activate.ps1
$env:PYTHONPATH = "$(Get-Location);$env:PYTHONPATH"