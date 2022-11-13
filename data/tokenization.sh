cat ./data/corpus.shuf.train.ko | mecab -O wakati -b 99999 | python3 ./data/post_tokenize.py ./data/corpus.shuf.train.ko > ./data/corpus.shuf.train.tok.ko
cat ./data/corpus.shuf.train.en | python3 ./data/tokenizer.py | python3 ./data/post_tokenize.py ./data/corpus.shuf.train.en > ./data/corpus.shuf.train.tok.en

cat ./data/corpus.shuf.valid.ko | mecab -O wakati -b 99999 | python3 ./data/post_tokenize.py ./data/corpus.shuf.valid.ko > ./data/corpus.shuf.valid.tok.ko
cat ./data/corpus.shuf.valid.en | python3 ./data/tokenizer.py | python3 ./data/post_tokenize.py ./data/corpus.shuf.valid.en > ./data/corpus.shuf.valid.tok.en

cat ./data/corpus.shuf.test.ko | mecab -O wakati -b 99999 | python3 ./data/post_tokenize.py ./data/corpus.shuf.test.ko > ./data/corpus.shuf.test.tok.ko
cat ./data/corpus.shuf.test.en | python3 ./data/tokenizer.py | python3 ./data/post_tokenize.py ./data/corpus.shuf.test.en > ./data/corpus.shuf.test.tok.en