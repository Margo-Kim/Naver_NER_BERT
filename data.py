from Korpora import Korpora

from transformers import AutoTokenizer

def preprocess():
    corpus = Korpora.load("naver_changwon_ner")
    texts = corpus.train.words
    tags = corpus.train.tags
    words = corpus.train.words

    # word 와 texts 의 차이..? 일단 뜯어본결과 한차례 정제된게 words 인것은 확인
    plm = "klue/roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(plm)
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    unk_token = tokenizer.unk_token
    pad_token_id = tokenizer.pad_token_id
    MAX_TOKEN_LEN = 512

    label_set = set()

    for lists in tags:
        for label in lists:
            label_set.add(label)

    label_set = list(label_set)
    label_to_ids = {l:i for i, l in enumerate(label_set)}
    label_to_ids['[PAD]'] = 29
    print(label_to_ids)
    # print(label_to_ids)
    # 임시 디버그용
    # words = words [ : 5]
    # tags = tags [ : 5]

    all_input_ids = []
    all_attention_masks = []
    all_token_type_ids = []
    all_label = []

    for a_words, a_labels in zip(words,tags):
        
        word_tokens = []
        word_labels = []
        # 리스트 안에 자료형 접근
        for word, label in zip(a_words, a_labels):
            b = tokenizer.tokenize(word)
            
            word_tokens += b
            word_labels.append (label )
            word_labels += ['[PAD]'] * (len(b)-1)
            
        
        # print(word_tokens, word_labels)
        
        word_tokens = [cls_token] + word_tokens + [sep_token]
        word_labels = ['[PAD]'] + word_labels
        
        # append 는 리스트의 끝에 해당 자료를 추가하는 느낌 (리스트 of list 도 가능)
        # += , extend 는 리스트에 리스트 더하기  [ ] + [ ] = [     ] 
        word_tokens += ['[PAD]'] * (MAX_TOKEN_LEN - len(word_tokens))
        word_labels += ['[PAD]'] * (MAX_TOKEN_LEN - len(word_labels))
        
        # total_tokens = []
        # total_labels = []
        
        # total_tokens.append(word_tokens)
        # total_labels.append(word_labels)
        
        # 궁금한점은 자료구조가 궁금함... [ ] [ ] [ ] 이렇게 되어있는것같아서...저런 total_tokens 가 필요한건지..in order to make list in list

        label_idx = [ ]
        # total_label = []
        # print(total_tokens)
    
        for label in word_labels:
            idx = label_to_ids[label]  
            label_idx.append(idx)
        
        
        input_ids = tokenizer.convert_tokens_to_ids(word_tokens)
        attention_mask = [1 if token_id != 1 else 0 for token_id in input_ids]
        token_type_ids = [0 if token_id == 1 else 1 for token_id in attention_mask]
        
        all_input_ids.append(input_ids)
        all_attention_masks.append(attention_mask)
        all_token_type_ids.append(token_type_ids)
        all_label.append(label_idx)
    
    return all_input_ids, all_attention_masks, all_token_type_ids, all_label
        
    
# print(preprocess())
