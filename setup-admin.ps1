# Разовая настройка Windows для Vision
# ─────────────────────────────────────
# Запускать ОТ ИМЕНИ АДМИНИСТРАТОРА:
#   правый клик по файлу → «Выполнить с помощью PowerShell»
#
# Делает две вещи:
#   1) открывает порты 80, 443 и 8004 в брандмауэре ТОЛЬКО для локальной сети,
#      чтобы телефон мог достучаться (из интернета — нельзя);
#   2) прописывает имена vision.vlx и vision.local в файл hosts,
#      чтобы на этом компьютере они открывались в браузере.
#
# Телефону настройка не нужна: имя vision.local он узнаёт сам,
# через Bonjour — это встроено в iOS.

$ruleName = "Vision · камера и дашборд"
$hostsFile = "$env:SystemRoot\System32\drivers\etc\hosts"
$names = @("vision.vlx", "vision.local")

$id = [Security.Principal.WindowsIdentity]::GetCurrent()
if (-not (New-Object Security.Principal.WindowsPrincipal($id)).IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host "Нужны права администратора." -ForegroundColor Red
    Write-Host "Правый клик по файлу → «Выполнить с помощью PowerShell» от админа."
    Read-Host "Enter для выхода"
    exit 1
}

# ── 1. брандмауэр ──────────────────────────────────────────────────────────
Get-NetFirewallRule -DisplayName "Vision*" -ErrorAction SilentlyContinue | Remove-NetFirewallRule
# 8010 — http для компьютера, 8443 — https для телефона, 8090 — перенаправление.
# Порты 80, 443 и 8004 НЕ трогаем: их занимает MES-система в Docker.
New-NetFirewallRule -DisplayName $ruleName `
    -Direction Inbound -Action Allow -Protocol TCP `
    -LocalPort 8010, 8090, 8443 -Profile Any -RemoteAddress LocalSubnet | Out-Null
Write-Host "✓ Порты 8010, 8090 и 8443 открыты для локальной сети." -ForegroundColor Green

# ── 2. имена в файле hosts ────────────────────────────────────────────────
$lines = Get-Content $hostsFile -ErrorAction SilentlyContinue
$added = @()
foreach ($n in $names) {
    if ($lines -match "^\s*[\d\.]+\s+$([regex]::Escape($n))\s*$") { continue }
    Add-Content -Path $hostsFile -Value "127.0.0.1`t$n" -Encoding ASCII
    $added += $n
}
if ($added.Count) {
    Write-Host "✓ В hosts добавлено: $($added -join ', ')" -ForegroundColor Green
} else {
    Write-Host "✓ Имена в hosts уже были." -ForegroundColor Green
}

ipconfig /flushdns | Out-Null

Write-Host ""
Write-Host "Готово. Теперь:"
Write-Host "   на компьютере →  vision.vlx:8010"
Write-Host "   на телефоне   →  vision.local:8443"
Write-Host ""
Read-Host "Enter для выхода"
