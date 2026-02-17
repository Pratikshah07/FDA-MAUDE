# Script to set GROQ_API_KEY for current session
# Usage: .\set_groq_key.ps1

Write-Host "Setting GROQ_API_KEY environment variable..." -ForegroundColor Yellow
Write-Host ""
Write-Host "Please enter your Groq API key:" -ForegroundColor Cyan
Write-Host "(You can get one from https://console.groq.com/)" -ForegroundColor Gray
Write-Host ""

$apiKey = Read-Host "Enter GROQ_API_KEY" -AsSecureString
$BSTR = [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($apiKey)
$plainKey = [System.Runtime.InteropServices.Marshal]::PtrToStringAuto($BSTR)

$env:GROQ_API_KEY = $plainKey

Write-Host ""
Write-Host "GROQ_API_KEY has been set for this session!" -ForegroundColor Green
Write-Host ""
Write-Host "To set it permanently, run this command in PowerShell as Administrator:" -ForegroundColor Yellow
Write-Host '[System.Environment]::SetEnvironmentVariable("GROQ_API_KEY", "' + $plainKey + '", "User")' -ForegroundColor Cyan
Write-Host ""
Write-Host "Or manually:" -ForegroundColor Yellow
Write-Host "1. Press Win+R, type 'sysdm.cpl', press Enter" -ForegroundColor Cyan
Write-Host "2. Go to 'Advanced' tab -> 'Environment Variables'" -ForegroundColor Cyan
Write-Host "3. Under 'User variables', click 'New'" -ForegroundColor Cyan
Write-Host "4. Variable name: GROQ_API_KEY" -ForegroundColor Cyan
Write-Host "5. Variable value: (your API key)" -ForegroundColor Cyan
Write-Host ""
