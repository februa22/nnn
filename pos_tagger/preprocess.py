import re
import hgtk

def preprocess_raw(raw_str):

    raw_str = re.sub('[0-9]+', "Ｎ", raw_str)   #숫자
    raw_str = re.sub('[a-zA-Z]+', "Ａ", raw_str)    #영어
    #raw_str = re.sub('[a-z0-9]+', "Ａ", raw_str)
    #raw_str = re.sub('[!\?]', "Ｐ", raw_str)
    #raw_str = re.sub('[,ㆍ]', "Ｃ", raw_str)
    #raw_str = re.sub('_', "Ｃ", raw_str)
    #raw_str = re.sub('[…‥]', "Ｅ", raw_str)
    #raw_str = re.sub('[~∼×～\-]', "Ｔ", raw_str)
    #raw_str = re.sub('[\'\"()\[\]{}<>―「」【】\“‘’”〈〉\-]', "Ｑ", raw_str)
    #raw_str = re.sub('[\u2018-\u3015]', "Ｗ", raw_str)
    #한자 치환
    raw_str = re.sub('[\u2E80-\u2EFF\u3400-\u4DBF\u4E00-\u9FBF\uF900-\uFAFF]+', "Ｈ", raw_str)
    #일본어 치환
    raw_str = re.sub('[\u3040-\u309F\u30A0-\u30FF\u31F0-\u31FF]+', "Ｊ", raw_str)

    return raw_str

def preprocess_pos(pos_str):
#    pos_str = re.sub(' +', '', pos_str)

    pos_str = re.sub('[0-9]+/sn', "Ｎ/SN", pos_str)
    pos_str = re.sub('[a-z0-9]+/sl', "Ａ/SL", pos_str)
    pos_str = re.sub('[!\?]/sf', "Ｐ/SF", pos_str)
    pos_str = re.sub('[,ㆍ]/sp', "Ｃ/SP", pos_str)
    pos_str = re.sub('_/s[?]', "Ｃ/SP", pos_str)
    pos_str = re.sub('[…‥]/se', "Ｅ/SE", pos_str)
    pos_str = re.sub('[~∼×～\-]/so', "Ｔ/SO", pos_str)
    pos_str = re.sub('[\'\"()\[\]{}<>―「」【】\“‘’”〈〉\-]/ss', "Ｑ/SS", pos_str)
    pos_str = re.sub('[\u2018-\u3015]/s?', "Ｗ/SW", pos_str)
    #한자 치환
    pos_str = re.sub('[\u2E80-\u2EFF\u3400-\u4DBF\u4E00-\u9FBF\uF900-\uFAFF]+/sh', "Ｈ/SH", pos_str)
    #일본어 치환
    pos_str = re.sub('[\u3040-\u309F\u30A0-\u30FF\u31F0-\u31FF]+/sh', "Ｊ/SH", pos_str)
    pos_str = re.sub('[\u3040-\u309F\u30A0-\u30FF\u31F0-\u31FF]+/sl', "Ｊ/SH", pos_str)

# BIO 태깅
def tag_bio(sentence):
    sentence = sentence.strip().split()
    tagged_sentence_list = []
    for word in sentence:
        for index, char in enumerate(word):
            if index == 0:
                #tagged_sentence_list.append('B' + char)
                tagged_sentence_list.append(('B', char))
            else:
                #tagged_sentence_list.append('I' + char)
                tagged_sentence_list.append(('I', char))
    return tagged_sentence_list

def split_tag_and_emj(token):
    if token and (token[0] == 'B' or token[0] == 'I'):
        return token[0], token[1:]
    else:
        return '', token

def preprocess_and_tokenize(sentence, bio=True, emj_split=True):
    sentence = preprocess_raw(sentence.strip())
    splitted_sentence = sentence.split()
    tokens = []
    for eoj in splitted_sentence:
        for index, emj in enumerate(eoj):
            #자소분리
            if hgtk.checker.is_hangul(emj):
                emj = ''.join(hgtk.letter.decompose(emj))
            #BIO태그
            if index == 0:
                tokens.append('B'+emj)
            else:
                tokens.append('I'+emj)
    return tokens

def main():
    sentence = '나는 밥을 먹었다. APPLE 1개를 가지고 있다'
    tagged_list = preprocess_and_tokenize(sentence)
    print(tagged_list)    

if __name__ == '__main__':
    main()