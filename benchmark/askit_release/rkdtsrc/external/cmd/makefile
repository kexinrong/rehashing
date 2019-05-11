SOURCES = CmdLine.cpp CmdOption.cpp general.cpp
HEADERS = CmdLine.h CmdOption.h general.h 

all : libcmd.a

libcmd.a : $(SOURCES)
	g++ -c -O3 $(SOURCES)
	ar -rs libcmd.a *.o
	$(RM) *.o
