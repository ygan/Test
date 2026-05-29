from medcat.components.ner.trf.deid import DeIdModel

# This may be in a different place depending on your MedCAT version
from medcat.components.ner.trf.helpers import replace_entities_in_text
from tqdm import tqdm
import json


def main():

    # path to your DeID AnonCAT model
    # model_path = "AnonCAT"
    model_path = "/opt/notebooks/ERUK/SB/gstt_data/AnonCAT"

    data_path = "/opt/notebooks/ERUK/software/LLaMA-Factory/data/real_train_1481_COT_freq_number_xml4PPO.json"
    
    # text field in your dataset
    # text_field = "document_Content"
    text_field = "instruction"
    

    # whether you want [ADDRESS] or ************ in your document - we (ERUK project) have been using the former.
    redact = False
    output_dir = data_path + 'AnonCAT.json'
   
    # if you want to do a test of a few examples
    test_sample = False # set True if want examples
    length_of_sample = 50

    # if you want to use standard or modified version - True = Standard, False = Modified - you want False
    remove_dates = False

    # Read data from csv
    with open(data_path, 'r') as f:
        all_data = json.load(f)

    # create test subset if test_sample = True
    if test_sample:
        df = df.head(length_of_sample)

    # load AnonCAT model
    deid = DeIdModel.load_model_pack(model_path)

    # Main loop - On CPU this took 3-4 hours for ~30000 rows. 
    for d in tqdm(all_data):
        # Standard AnonCAT
        if remove_dates:
            d[text_field] = deid.deid_text(d[text_field], redact=redact)
        else:
            # Modified AnonCAT
            ents = deid.get_entities(d[text_field])['entities']
            new_ents = {k: v for k, v in ents.items() if v['cui'] != "D4000"}
            d[text_field] = replace_entities_in_text(d[text_field],  new_ents, 
                                                          deid.cat.cdb.get_name, 
                                                          redact=redact)
            
    with open(output_dir, 'w') as f:
       json.dump(all_data, f)


# go go go
if __name__ == '__main__':
    main()



# python - <<'PY'
# import urllib.request

# url = "https://github.com/ygan/Test/releases/download/tag/test_retriever_v4.bin"
# out = "test_retriever_v4.bin"

# urllib.request.urlretrieve(url, out)
# print("Downloaded:", out)
# PY