#!/bin/bash
#
cd external/user/src; make lib; cd -
cd build; cmake ../; make; make install; make test
#end
