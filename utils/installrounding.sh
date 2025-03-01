#! /bin/bash

wget https://gitlab.com/MIAOresearch/software/roundingsat/-/archive/master/roundingsat-master.zip
unzip roundingsat-master.zip
cd roundingsat-master
cd build

cmake -DCMAKE_BUILD_TYPE=Release ..
make
cp roundingsat ../../../src/
cd ../..
rm -rf roundingsat-master*

# Note that the soplex version does not work yet.
# wget https://gitlab.com/MIAOresearch/software/roundingsat/-/archive/master/roundingsat-master.zip
# unzip roundingsat-master.zip
# cd roundingsat-master
# cd build
# wget https://github.com/scipopt/soplex/archive/refs/tags/release-710.tar.gz
# cmake -DCMAKE_BUILD_TYPE=Release -Dsoplex=ON -Dsoplex_pkg=./release-710.tar.gz .. 
# make
# cp roundingsat ../../../src/roundingsatsoplex
# cd ../..
# rm -rf roundingsat-master*
