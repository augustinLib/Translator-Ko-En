cat ./data/corpus.shuf.train.tok.ko | python ./data/subword-nmt/apply_bpe.py -c ./data/bpe.ko.model > ./data/corpus.shuf.train.tok.bpe.ko
cat ./data/corpus.shuf.valid.tok.ko | python ./data/subword-nmt/apply_bpe.py -c ./data/bpe.ko.model > ./data/corpus.shuf.valid.tok.bpe.ko
cat ./data/corpus.shuf.test.tok.ko | python ./data/subword-nmt/apply_bpe.py -c ./data/bpe.ko.model > ./data/corpus.shuf.test.tok.bpe.ko

cat ./data/corpus.shuf.train.tok.en | python ./data/subword-nmt/apply_bpe.py -c ./data/bpe.en.model > ./data/corpus.shuf.train.tok.bpe.en
cat ./data/corpus.shuf.valid.tok.en | python ./data/subword-nmt/apply_bpe.py -c ./data/bpe.en.model > ./data/corpus.shuf.valid.tok.bpe.en
cat ./data/corpus.shuf.test.tok.en | python ./data/subword-nmt/apply_bpe.py -c ./data/bpe.en.model > ./data/corpus.shuf.test.tok.bpe.en
