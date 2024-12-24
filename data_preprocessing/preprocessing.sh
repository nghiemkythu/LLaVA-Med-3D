TYPE="test"

python3 bao_preprocessing.py \
	--path /netscratch/duynguyen/Research/ \
	--dataset data_RAD \
	--original_file "${TYPE}set.json"

python3 bao_preprocessing.py \
	--path /netscratch/duynguyen/Research/ \
	--dataset Slake1.0 \
	--original_file "${TYPE}_raw.json" 

python3 bao_preprocessing.py \
	--path /netscratch/duynguyen/Research/ \
	--dataset pvqa \
	--original_file "qas/${TYPE}_vqa.pkl"
