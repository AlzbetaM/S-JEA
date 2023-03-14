rm log_files.txt
python src/pretrain.py -c config.conf --tag=test_vicreg
python src/finetune.py -c config.conf --tag=test_vicreg
