rm log_files.txt
python src/pretrain.py -c config.conf
python src/finetune.py -c config.conf
