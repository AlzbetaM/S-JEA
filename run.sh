rm log_files.txt
python src/pretrain.py -c ~/Documents/VICReg/config.conf --tag=test_vicreg
python src/finetune.py -c ~/Documents/VICReg/config.conf --tag=test_vicreg
