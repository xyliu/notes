nullstring :=

ifeq (${nullstring}, ${JAVA_HOME})
JAVA_HOME := '/usr/lib/jvm/java-8-openjdk-amd64'
endif



all: Main.java Main.c
	javac Main.java 
	javah -jni Main
	gcc --shared -fpic -o libMain.so -I${JAVA_HOME}/include -I${JAVA_HOME}/include/linux Main.c
	java -Djava.library.path=. Main
clean:
	rm -f libMain.so Main.class Main.h  
