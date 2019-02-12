#echo "Please input compute capabiliy of your GPU, (52 for example), otherwise 53 is the default value"
#read varname
#if [$varname -gt 50]
#then 
#	sed -i 's/53/$varname/g' src/CMakeLists.txt
#else
#	echo "default"
#fi
sh clean.sh
cmake . && make
