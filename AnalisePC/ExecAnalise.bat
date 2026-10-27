@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8

set RODADAS=%1
if "%RODADAS%"=="" set RODADAS=20

set SEEDS_FIXAS=%2
if "%SEEDS_FIXAS%"=="" set SEEDS_FIXAS=1

powershell -NoProfile -Command ^
  "$rodadas = %RODADAS%; " ^
  "$seedsFixas = %SEEDS_FIXAS%; " ^
  "$ts = Get-Date -Format 'yyyyMMdd_HHmmss'; " ^
  "$logFile = \"relatorio_$ts.txt\"; " ^
  "$logConsolidado = \"relatorio_consolidado_$ts.txt\"; " ^
  "function Log($txt) { $txt | Tee-Object -FilePath $logFile -Append }; " ^
  "Log \"================ PREPARACAO E GERACAO DOS SETTINGS ($rodadas RODADAS - SEEDS FIXAS: $seedsFixas) ================\"; " ^
  "python -u gerar_settings.py --grupos $rodadas --usar_seeds_fixas $seedsFixas | Tee-Object -FilePath $logFile -Append; " ^
  "if ($LASTEXITCODE -ne 0) { Write-Host '`n[ERRO] Falha no gerar_settings.py! Abortando execucao.' -ForegroundColor Red; exit $LASTEXITCODE }; " ^
  "$arquivos = Get-ChildItem -Path 'settings\setting*-train.txt' | Sort-Object { [int]($_.BaseName -replace '^setting(\d+)\.(\d+)-train$','$1') }, { [int]($_.BaseName -replace '^setting(\d+)\.(\d+)-train$','$2') }; " ^
  "if (-not $arquivos) { Write-Host '`n[ERRO] Nenhum arquivo setting*-train.txt encontrado na pasta settings!' -ForegroundColor Red; exit 1 }; " ^
  "function Rodar-Bloco($tituloBloco, $tagModelo, $cmdBase) { " ^
  "  Log \"`n================ $tituloBloco ================\"; " ^
  "  foreach ($arq in $arquivos) { " ^
  "    $id = $arq.BaseName -replace '^setting','' -replace '-train$',''; " ^
  "    $caminhoRel = 'settings/' + $arq.Name; " ^
  "    Log \"`n========================================\"; " ^
  "    Log \"   $tagModelo - INICIANDO SETTING $id   \"; " ^
  "    Log \"========================================\"; " ^
  "    $i = Get-Date; Log ('INICIO: ' + $i.ToString('HH:mm:ss.ff')); " ^
  "    Invoke-Expression \"$cmdBase --outfile strawberry_$id -i $caminhoRel\" | Tee-Object -FilePath $logFile -Append; " ^
  "    $f = Get-Date; Log ('FIM:    ' + $f.ToString('HH:mm:ss.ff')); " ^
  "    $seg = [math]::Round(($f - $i).TotalSeconds, 2); " ^
  "    Log (\"TEMPO TOTAL $tagModelo SETTING $id : \" + $seg.ToString([System.Globalization.CultureInfo]::InvariantCulture) + ' segundos'); " ^
  "    Log \"========================================`n\"; " ^
  "  } " ^
  "}; " ^
  "Rodar-Bloco 'BLOCO 1: A1 - LMFCN' 'LMFCN' 'python -u main_lmfcn.py --epoch 5 --relax 100 --lr 1.0 --feature 64 --svclose 5 --wrclose 5 --shclose 0 --width 275 --height 183 --outdir strawberry_out --batch 32'; " ^
  "Rodar-Bloco 'BLOCO 2: B1 - YOLO11 SCRATCH' 'YOLO11 SCRATCH' 'python -u main_baselines.py --model yolo11 --pretrained 0 --epoch 5 --width 275 --height 183 --outdir strawberry_out --batch 32'; " ^
  "Rodar-Bloco 'BLOCO 3: B2 - YOLO11 PRE-TREINADO' 'YOLO11 PRE-TREINADO' 'python -u main_baselines.py --model yolo11 --pretrained 1 --epoch 5 --width 275 --height 183 --outdir strawberry_out --batch 32'; " ^
  "Rodar-Bloco 'BLOCO 4: C1 - MOBILENET SCRATCH' 'MOBILENET SCRATCH' 'python -u main_baselines.py --model mobilenet --pretrained 0 --epoch 5 --width 275 --height 183 --outdir strawberry_out --batch 32'; " ^
  "Rodar-Bloco 'BLOCO 5: C2 - MOBILENET PRE-TREINADO' 'MOBILENET PRE-TREINADO' 'python -u main_baselines.py --model mobilenet --pretrained 1 --epoch 5 --width 275 --height 183 --outdir strawberry_out --batch 32'; " ^
  "python -u consolidar_relatorio.py $logFile $logConsolidado; " ^
  "Write-Host \"`n[OK] Relatorio completo salvo em: $logFile\" -ForegroundColor Green; " ^
  "Write-Host \"[OK] Relatorio consolidado salvo em: $logConsolidado\" -ForegroundColor Green"