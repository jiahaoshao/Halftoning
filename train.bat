@echo off
cd /d D:\GraduationProject\Code\Halftoning
python train.py -c config\config.json -r halftoning_dev\model_last.pth.tar
pause