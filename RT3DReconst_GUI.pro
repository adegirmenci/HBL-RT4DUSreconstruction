#-------------------------------------------------
#
# Project created by QtCreator 2016-09-07T02:57:21
#
#-------------------------------------------------

QT       += core gui network

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = RT3DReconst_GUI
CONFIG   += console
CONFIG   -= app_bundle
TEMPLATE = app


SOURCES += main.cpp\
        rt3dreconst_gui.cpp \
    frameserverthread.cpp \
    frameserverwidget.cpp \
    rt3dreconstworker.cpp \
    Volume3D.cpp

HEADERS  += rt3dreconst_gui.h \
    frameserverthread.h \
    frameserverwidget.h \
    rt3dreconstworker.h \
    Volume3D.h

FORMS    += rt3dreconst_gui.ui \
    frameserverwidget.ui

win32 {
    INCLUDEPATH += "C:\\opencv\\build\\include" \
                   "D:\\qhull-2015.2\\src"

    CONFIG(debug,debug|release) {
        LIBS += -L"C:\\opencv\\build\\x86\\vc12\\lib" \
            -lopencv_core2411d \
            -lopencv_highgui2411d \
            -lopencv_imgproc2411d \
            -lopencv_features2d2411d \
            -lopencv_calib3d2411d
    }

    CONFIG(release,debug|release) {
        LIBS += -L"C:\\opencv\\build\\x86\\vc12\\lib" \
            -lopencv_core2411 \
            -lopencv_highgui2411 \
            -lopencv_imgproc2411 \
            -lopencv_features2d2411 \
            -lopencv_calib3d2411
    }

    LIBS += -L"D:\\qhull-2015.2\\lib" \
            -lqhullstatic_r \
            -lqhullcpp
}

# Define output directories
DESTDIR = release
OBJECTS_DIR = release/obj
CUDA_OBJECTS_DIR = release/cuda
win32:{
    DEFINES+ = WIN32
    DEFINES+ = _WIN32
}

#----------------------------------------------------------------
#-------------------------Cuda setup-----------------------------
#----------------------------------------------------------------

#Enter your gencode here!
GENCODE = arch=compute_35,code=sm_35

#We must define this as we get some confilcs in minwindef.h and helper_math.h
DEFINES += NOMINMAX

#set out cuda sources
CUDA_SOURCES = "$$PWD"/RT3DUSkernels.cu

#This is to add our .cu files to our file browser in Qt
SOURCES+=RT3DUSkernels.cu
SOURCES-=RT3DUSkernels.cu

# Path to cuda SDK install
win32:CUDA_DIR = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v7.0"
# Path to cuda toolkit install
#win32:CUDA_SDK = "C:\\ProgramData\\NVIDIA Corporation\CUDA Samples\\v7.0"

#Cuda include paths
INCLUDEPATH += $$CUDA_DIR/include
#INCLUDEPATH += $$CUDA_DIR/common/inc/
#INCLUDEPATH += $$CUDA_DIR/../shared/inc/
#To get some prewritten helper functions from NVIDIA
win32:INCLUDEPATH += $$CUDA_SDK\common\inc

#cuda libs
win32:QMAKE_LIBDIR += $$CUDA_DIR\lib\Win32
#win32:QMAKE_LIBDIR += $$CUDA_SDK\common\lib\Win32
LIBS += -lcudart -lcudadevrt

# join the includes in a line
CUDA_INC = $$join(INCLUDEPATH,'" -I"','-I"','"')

# nvcc flags (ptxas option verbose is always useful)
NVCCFLAGS = --compiler-options -fno-strict-aliasing -use_fast_math --ptxas-options=-v

#On windows we must define if we are in debug mode or not
CONFIG(debug, debug|release) {
#DEBUG
    # MSVCRT link option (static or dynamic, it must be the same with your Qt SDK link option)
    win32:MSVCRT_LINK_FLAG_DEBUG = "/MDd"
    win32:NVCCFLAGS += -D_DEBUG -Xcompiler $$MSVCRT_LINK_FLAG_DEBUG
}
else{
#Release UNTESTED!!!
    win32:MSVCRT_LINK_FLAG_RELEASE = "/MD"
    win32:NVCCFLAGS += -Xcompiler $$MSVCRT_LINK_FLAG_RELEASE
}

#prepare intermediat cuda compiler
cudaIntr.input = CUDA_SOURCES
cudaIntr.output = ${OBJECTS_DIR}${QMAKE_FILE_BASE}.o
#So in windows object files have to be named with the .obj suffix instead of just .o
#God I hate you windows!!
win32:cudaIntr.output = $$OBJECTS_DIR/${QMAKE_FILE_BASE}.obj

## Tweak arch according to your hw's compute capability
cudaIntr.commands = $$CUDA_DIR/bin/nvcc -m32 -g -gencode $$GENCODE -dc $$NVCCFLAGS $$CUDA_INC $$LIBS ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT}

#Set our variable out. These obj files need to be used to create the link obj file
#and used in our final gcc compilation
cudaIntr.variable_out = CUDA_OBJ
cudaIntr.variable_out += OBJECTS
cudaIntr.clean = cudaIntrObj/*.o
win32:cudaIntr.clean = cudaIntrObj/*.obj

QMAKE_EXTRA_UNIX_COMPILERS += cudaIntr

# Prepare the linking compiler step
cuda.input = CUDA_OBJ
cuda.output = ${QMAKE_FILE_BASE}_link.o
win32:cuda.output = ${QMAKE_FILE_BASE}_link.obj

# Tweak arch according to your hw's compute capability
cuda.commands = $$CUDA_DIR/bin/nvcc -m32 -g -gencode $$GENCODE  -dlink    ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT}
cuda.dependency_type = TYPE_C
cuda.depend_command = $$CUDA_DIR/bin/nvcc -g -M $$CUDA_INC $$NVCCFLAGS   ${QMAKE_FILE_NAME}
# Tell Qt that we want add more stuff to the Makefile
QMAKE_EXTRA_UNIX_COMPILERS += cuda
