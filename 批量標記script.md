使用範例:
  單一資料夾處理:
```zsh
python3 befor_processes.py --paths /path/to/dicom/folder
```
  多個資料夾處理:
```zsh
python3 befor_processes.py --paths /path/to/dicom1 /path/to/dicom2 /path/to/dicom3
```
  自動發現並批次處理:
```zsh
python3 befor_processes.py --batch /path/to/parent/folder
```
  從檔案讀取路徑列表:
```zsh
python3 befor_processes.py --input-list patients.txt
```
  平行處理:
```zsh
python3 befor_processes.py --parallel --max-workers 4 --paths /path/to/dicom1 /path/to/dicom2
```
  使用新模型:
```zsh
python3 befor_processes.py --use-new-models --paths /path/to/dicom/folder
```

其他選項:
```zsh
python3 befor_processes.py --help
```

我怎麼跑
```zsh
python3 befor_processes.py --batch "/Users/lujingyuan/Downloads/沒有顯影劑" --parallel --output-dir "results" --skip-existing  --use-new-models --max-workers 4
``` 