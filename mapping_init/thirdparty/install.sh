    # N.B.: pay attention this will generate a module 'pypangolin' ( it does not have the methods dcam.SetBounds(...) and pangolin.DrawPoints(points, colors)  )
    if [ ! -d pangolin ]; then
        if [[ "$OSTYPE" == "linux-gnu"* ]]; then
            sudo apt-get install -y libglew-dev
        fi     
        git clone https://github.com/stevenlovegrove/Pangolin.git pangolin
        cd pangolin
        git submodule init && git submodule update
        cd ..
    fi
    cd pangolin
    mkdir build 
    if [ ! -f build/src/libpangolin.so ]; then
        cd build
        cmake ../ -DAVFORMAT_INCLUDE_DIR="" -DCPP11_NO_BOOST=ON $EXTERNAL_OPTION
        make -j8
        cd build/src
        ln -s pypangolin.*-linux-gnu.so  pangolin.linux-gnu.so
    fi
