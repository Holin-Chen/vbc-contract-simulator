# Download all 5 CMS datasets into data/raw/
# Run from project root: .\download_data.ps1

$rawDir = "data\raw"
New-Item -ItemType Directory -Force $rawDir | Out-Null

$files = @(
    @{
        Url  = "https://data.cms.gov/provider-data/sites/default/files/resources/893c372430d9d71a1c52737d01239d47_1777413958/Hospital_General_Information.csv"
        Dest = "$rawDir\Hospital_General_Information.csv"
        Name = "Hospital General Information"
    },
    @{
        Url  = "https://data.cms.gov/provider-data/sites/default/files/resources/69874ce604586980ac088283c1b35095_1777413962/Medicare_Hospital_Spending_Per_Patient-Hospital.csv"
        Dest = "$rawDir\Medicare_Spending_Per_Beneficiary_Hospital.csv"
        Name = "Medicare Spending Per Beneficiary (MSPB)"
    },
    @{
        Url  = "https://data.cms.gov/provider-data/sites/default/files/resources/5551d4839c1dd75e3f7fe1310a1e2369_1770163628/hvbp_tps.csv"
        Dest = "$rawDir\VBP_Hospital_TPS.csv"
        Name = "Hospital VBP Total Performance Score"
    },
    @{
        Url  = "https://data.cms.gov/provider-data/sites/default/files/resources/78a50346fbe828ea0ce2837847af6a7c_1777413952/HCAHPS-Hospital.csv"
        Dest = "$rawDir\HCAHPS_Hospital.csv"
        Name = "HCAHPS Patient Experience"
    },
    @{
        Url  = "https://data.cms.gov/provider-data/sites/default/files/resources/30edc1d0417a34b58affcc2495a02b0a_1777413968/Unplanned_Hospital_Visits-Hospital.csv"
        Dest = "$rawDir\Unplanned_Hospital_Visits_Provider_Data.csv"
        Name = "Unplanned Hospital Visits / Readmissions"
    }
)

foreach ($f in $files) {
    Write-Host "Downloading: $($f.Name)..." -ForegroundColor Cyan
    try {
        Invoke-WebRequest -Uri $f.Url -OutFile $f.Dest -UseBasicParsing
        $size = (Get-Item $f.Dest).Length / 1MB
        Write-Host "  Saved to $($f.Dest) ($([math]::Round($size,1)) MB)" -ForegroundColor Green
    } catch {
        Write-Host "  FAILED: $_" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "Done. Files in $rawDir :" -ForegroundColor Yellow
Get-ChildItem $rawDir | Select-Object Name, @{N="Size (KB)";E={[math]::Round($_.Length/1KB,0)}}
