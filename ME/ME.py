lsb_release -a

sudo apt-get install build-essential git-core pkg-config automake libtool wget
zlib1g-dev python-dev libbz2-dev

git clone <https://github.com/moses-smt/mosesdecoder.git>
cd mosesdecoder

make -f contrib/Makefiles/install-dependencies.gmake

./compile.sh [additional options]

\--prefix=/destination/path --install-scripts 

\--with-mm 

wget <https://dl.bintray.com/boostorg/release/1.64.0/source/boost_1_64_0.tar.gz>

tar zxvf boost_1_64_0.tar.gz

cd boost_1_64_0/

./bootstrap.sh

./b2 -j4 --prefix=\$PWD --libdir=\$PWD/lib64 --layout=system link=static install
\|\| echo FAILURE \

wget
http://www.mirrorservice.org/sites/download.sourceforge.net/pub/sourceforge/c/cm/cmph/cmph/cmph-2.0.tar.gz

tar zxvf cmph-2.0.tar.gz

cd cd cmph-2.0/

./configure --prefix= /usr/local/translation

Make

Make install

wget
https://launchpad.net/ubuntu/+archive/primary/+sourcefiles/xmlrpc-c/1.33.14-8build1/xmlrpc-c_1.33.14.orig.tar.gz

tar zxvf xmlrpc-c_1.33.14.orig.tar.gz

cd xmlrpc-c-1.33.14/

./configure --prefix= /usr/local/xmlrpc-c

Make

Make install

./bjam --with-boost=/home/ubuntu/boost_1_64_0 --with-cmph=/usr/local/cmph
--with-xmlrpc-c=/usr/local/xmlrpc-c -j4

git clone <https://github.com/moses-smt/giza-pp.git>

cd giza-pp

make

cd \~/mosesdecoder

mkdir tools

cp \~/giza-pp/GIZA++-v2/GIZA++ \~/giza-pp/GIZA++-v2/snt2cooc.out \\

\~/giza-pp/mkcls-v2/mkcls tools

train-model.perl -external-bin-dir \$HOME/external-bin-dir

\~/mosesdecoder/scripts/tokenizer/tokenizer.perl -l en \\

\< \~/corpus/training/train.tags.zh-en.en \\

\> \~/corpus/train.tags.zh-en.tok.en


import thulac

thu=thulac.thulac(user_dict="/home/ubuntu/corpus/dict.txt",seg_only=True)


thu.cut_f("/home/ubuntu/corpus/train.tags.zh-en.zh",
"/home/ubuntu/corpus/train.tags.zh-en.zh.fc")
