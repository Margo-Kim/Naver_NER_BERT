from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base")
MAX_TOKEN_LEN = 512

label_set = ["O", "B-Person", "I-Person", "[PAD]"]
cls_token = tokenizer.cls_token
sep_token = tokenizer.sep_token
unk_token = tokenizer.unk_token
pad_token_id = tokenizer.pad_token_id

sentences = [["마크크크", "주커버그는", "밥을", "먹었다"], ["허철훈은", "슬프다"]]
labels = [["B-Person", "I-Person", "O", "O"], ["B-Person", "O"]]

label_to_ids = {l:i for i, l in enumerate(label_set)}


for a_sentences, a_labels in zip(sentences, labels):
    
    word_tokens = []
    word_label = []
    for word, label in zip(a_sentences,a_labels):
        b = tokenizer.tokenize(word)
        word_tokens += tokenizer.tokenize(word)
        word_label.append(label)
        word_label+= ['[PAD]'] * (len(b)-1)
        
    word_tokens = [cls_token] + word_tokens + [sep_token] 
    word_label = ['[PAD]'] + word_label
    
    word_tokens += ['[PAD]'] * (MAX_TOKEN_LEN - len(word_tokens))
    word_label += ['[PAD]'] * (MAX_TOKEN_LEN - len(word_label))
    
    # label to idx
    label_ids = []
    for label in word_label:
        idx = label_to_ids[label]
        
        label_ids.append(idx)

input_ids = tokenizer.convert_tokens_to_ids(word_tokens)
attention_mask = [1 if token_id != 1 else 0 for token_id in input_ids]
token_type_ids = [0 if token_id == 1 else 1 for token_id in attention_mask]


  

print(label_ids)


# zip은 묶어서 내보내는 것
# 단어 단위로 토큰 나이징 하거나 문장을 토크나이징 해도 똑같다
# 단어 단위로 토크나이징을 하면 따로 해야할 것 들이 늘어난다

# 서브 토큰을 안쓰고 메인만 쓴다는 의미로 간다

# += 는 extend 랑 똑같다
# 통째를 넣으면 offset mapping 을 해야한다 --> 케릭터 인덱스로 찾아서 label 맵핑이 필요하다
# 코딩 양 자체를 늘려야한다.....
# 