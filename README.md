# MONA
Matrial of Numerical Algebra
Keneral Of Numerical Algebra

app            外部软件包调用接口
bin            可执行文件夹
build          cmake文件夹
external       外部软件包
include        头文件夹
lib            库文件夹
src            源文件夹
test           测试文件夹，基于app和external

Install:
cd external/user/src; make lib; cd -
cd build
cmake ../
make
make install
make test

如何将ops写好?
