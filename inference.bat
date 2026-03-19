@echo off
cd /d D:\GraduationProject\Code\Halftoning
python inference.py --model halftoning_dev\model_best.pth.tar --input e:\SteamLibrary\f4f6c0c4fd989f3639ab757f5510d98178305ff12498d4b58.jpg --output dataset\HalftoneVOC2012\test_result
pause