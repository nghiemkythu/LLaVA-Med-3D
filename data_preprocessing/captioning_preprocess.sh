PATH="/netscratch/duynguyen/Research/Nghiem_LLaVA-Med/Data_Report_Gen"
FILE=test 

python3 captioning_preprocess.py \
	--path $PATH/iu_xray \
	--file $FILE \

python3 captioning_preprocess.py \
	--path $PATH/peir_gross \
	--file $FILE
